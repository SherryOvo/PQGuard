#!/usr/bin/env python3
"""
PQGuard LLM Pipeline

Integrates all PQGuard components into a complete LLM inference pipeline:
1. Model integrity verification on load
2. Encrypted session establishment
3. Secure inference execution
4. Audit logging with non-repudiation
"""

import sys
import os
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import torch

# Set model environment before importing transformers
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
try:
    import env_model
except ImportError:
    pass

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModelForVision2Seq

from pqguard.sal import PQSecurityAbstractionLayer
from pqguard.model_integrity import ModelIntegrityVerifier, ModelManifest
from pqguard.session_encryption import SecureSessionManager
from pqguard.audit_log import AuditLogger


class PQGuardLLMPipeline:
    """
    PQGuard LLM Inference Pipeline
    
    Complete end-to-end post-quantum secure LLM inference.
    """
    
    def __init__(self, model_id: str, model_type: str = "qwen",
                 verify_integrity: bool = True,
                 enable_encryption: bool = True,
                 enable_audit: bool = True,
                 manifest_path: Optional[Path] = None,
                 trusted_keys_dir: Optional[Path] = None):
        """
        Initialize PQGuard LLM Pipeline.
        
        Args:
            model_id: HuggingFace model ID or local path
            model_type: Model type ("qwen", "llama", "phi", etc.)
            verify_integrity: Whether to verify model integrity
            enable_encryption: Whether to enable session encryption
            enable_audit: Whether to enable audit logging
            manifest_path: Path to model manifest (auto-detected if None)
            trusted_keys_dir: Directory containing trusted public keys
        """
        self.model_id = model_id
        self.model_type = model_type
        self.verify_integrity = verify_integrity
        self.enable_encryption = enable_encryption
        self.enable_audit = enable_audit
        
        # Initialize PQ-SAL
        self.sal = PQSecurityAbstractionLayer()
        
        # Initialize components
        self.integrity_verifier = ModelIntegrityVerifier(self.sal) if verify_integrity else None
        self.session_manager = SecureSessionManager(self.sal) if enable_encryption else None
        self.audit_logger = AuditLogger(self.sal) if enable_audit else None
        
        # Load trusted keys
        if trusted_keys_dir and self.integrity_verifier:
            self._load_trusted_keys(Path(trusted_keys_dir))
        
        # Model components (loaded after verification)
        self.tokenizer = None
        self.model = None
        self.processor = None  # For multimodal models
        self.model_path: Optional[Path] = None
        self.model_version_hash: Optional[str] = None
        
        # Load and verify model
        self._load_model(manifest_path)
    
    def _load_trusted_keys(self, keys_dir: Path):
        """Load trusted public keys from directory."""
        keys_dir = Path(keys_dir)
        if not keys_dir.exists():
            print(f"Warning: Trusted keys directory not found: {keys_dir}")
            return
        
        for key_file in keys_dir.glob("*.pub"):
            model_id = key_file.stem
            public_key = key_file.read_bytes()
            self.integrity_verifier.add_trusted_key(model_id, public_key)
            print(f"Loaded trusted key for: {model_id}")
    
    def _load_model(self, manifest_path: Optional[Path] = None):
        """Load model with integrity verification."""
        # Resolve model path
        if Path(self.model_id).exists():
            self.model_path = Path(self.model_id)
        else:
            # Try HuggingFace cache
            cache_base = Path(os.environ.get("HF_HOME", "~/.cache/huggingface"))
            hub_path = cache_base / "hub"
            # Look for model in cache
            model_name_encoded = self.model_id.replace("/", "--")
            for model_dir in hub_path.glob(f"models--{model_name_encoded}*"):
                snapshots = list((model_dir / "snapshots").glob("*"))
                if snapshots:
                    self.model_path = snapshots[-1]
                    break

        # If model_path 仍然是 HuggingFace hub 仓库根目录（例如 models--Qwen--Qwen1.5-7B-Chat），
        # 自动跳转到最新的 snapshots 子目录，这里才包含标准的 config.json / 权重文件。
        if self.model_path and self.model_path.is_dir():
            snapshots_dir = self.model_path / "snapshots"
            if snapshots_dir.is_dir():
                snapshot_dirs = sorted(d for d in snapshots_dir.iterdir() if d.is_dir())
                if snapshot_dirs:
                    self.model_path = snapshot_dirs[-1]

        if not self.model_path or not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_id}")
        
        # Verify integrity if enabled
        if self.verify_integrity and self.integrity_verifier:
            print("Verifying model integrity...")
            if manifest_path is None:
                manifest_path = self.model_path / "manifest.json"
            
            if manifest_path.exists():
                is_valid, errors = self.integrity_verifier.verify_complete(
                    self.model_path, manifest_path
                )
                if not is_valid:
                    raise ValueError(f"Model integrity verification failed: {errors}")
                print("✓ Model integrity verified")
            else:
                print(f"Warning: Manifest not found at {manifest_path}, skipping verification")
        
        # Compute model version hash
        self.model_version_hash = self.sal.hash_model_weights(self.model_path)
        
        # Load tokenizer and model
        print(f"Loading model from: {self.model_path}")
        
        if self.model_type in ["qwen2vl", "multimodal"]:
            # Multimodal model
            self.processor = AutoProcessor.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
            )
            self.model = AutoModelForVision2Seq.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else {"": "cpu"},
            )
        else:
            # Text-only model
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else {"": "cpu"},
            )
        
        self.model.eval()
        print("✓ Model loaded successfully")
    
    def get_server_public_key(self) -> bytes:
        """Get server's KEM public key for session establishment."""
        if not self.session_manager:
            raise RuntimeError("Session encryption not enabled")
        return self.session_manager.get_server_public_key()
    
    def establish_session(self, kem_ciphertext: bytes) -> Tuple[str, bytes]:
        """Establish encrypted session (server side)."""
        if not self.session_manager:
            raise RuntimeError("Session encryption not enabled")
        return self.session_manager.create_session(kem_ciphertext)
    
    def generate(self, prompt: str, session_id: Optional[str] = None,
                max_new_tokens: int = 512, temperature: float = 0.7,
                top_p: float = 0.8, user_id: Optional[str] = None,
                images: Optional[List] = None) -> Dict[str, Any]:
        """
        Generate response with PQGuard protection.
        
        Args:
            prompt: Input prompt
            session_id: Session ID (for encrypted sessions)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            user_id: User identifier (for audit)
            images: Optional images (for multimodal models)
            
        Returns:
            Dictionary with response and metadata
        """
        # 当前实现中，服务端假定收到的 prompt 已是明文；
        # 会话加密主要用于响应数据加密，输入链路由上层通道（TLS/隧道）保护。
        # 如需端到端加密输入，可以扩展为接受密文参数并在此处解密。
        request_data = prompt.encode()
        
        # Prepare inputs
        if self.processor:  # Multimodal
            if images:
                messages = [{"role": "user", "content": prompt}]
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.processor(
                    text=text, images=images, return_tensors="pt"
                )
            else:
                messages = [{"role": "user", "content": prompt}]
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.processor(text=text, return_tensors="pt")
        else:  # Text-only
            messages = [{"role": "user", "content": prompt}]
            input_ids = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_tensors="pt"
            )
            inputs = {"input_ids": input_ids}
        
        # Move to device
        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
            )
        
        # Decode response
        if self.processor:
            input_ids_len = inputs["input_ids"].shape[-1]
            generated_ids_trimmed = [
                out_ids[input_ids_len:] for out_ids in outputs
            ]
            response = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        else:
            input_ids_len = inputs["input_ids"].shape[-1]
            response = self.tokenizer.decode(
                outputs[0][input_ids_len:],
                skip_special_tokens=True,
            )
        
        response_data = response.encode()
        
        # Encrypt response if session provided
        if session_id and self.session_manager:
            session = self.session_manager.get_session(session_id)
            if session:
                response_data = self.session_manager.encrypt_response(
                    session.session_key, response_data
                )
                response = response_data.hex()  # Return hex for encrypted data
        
        # Audit logging
        if self.audit_logger:
            entry = self.audit_logger.create_entry(
                session_id=session_id or "no-session",
                model_id=self.model_id,
                model_version_hash=self.model_version_hash,
                request_data=prompt.encode(),
                response_data=response.encode() if isinstance(response, str) else response_data,
                user_id=user_id,
                metadata={
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                },
            )
            self.audit_logger.log(entry)
        
        return {
            "response": response,
            "session_id": session_id,
            "model_id": self.model_id,
            "model_version_hash": self.model_version_hash,
            "encrypted": session_id is not None and self.session_manager is not None,
        }

