#!/usr/bin/env python3
"""
Model Integrity Verification Module

Provides fine-grained integrity verification for:
- Model weights
- LoRA adapters
- Prompt templates
- Tokenizer configurations

Uses Dilithium signatures for authenticity verification.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pqguard.sal import PQSecurityAbstractionLayer


@dataclass
class ModelManifest:
    """Model manifest structure for integrity verification."""
    model_id: str
    model_type: str  # e.g., "llama", "qwen", "phi"
    version: str
    files: List[Tuple[str, str]]  # (relative_path, sha256_hash)
    lora_adapters: Optional[List[Tuple[str, str]]] = None
    prompt_templates: Optional[Dict[str, str]] = None
    signature: Optional[str] = None  # Hex-encoded Dilithium signature
    
    def to_json_bytes(self) -> bytes:
        """Convert manifest to JSON bytes (excluding signature)."""
        data = asdict(self)
        data.pop("signature", None)
        return json.dumps(data, sort_keys=True).encode()
    
    @classmethod
    def from_dict(cls, data: dict):
        """Create manifest from dictionary."""
        return cls(**data)


class ModelIntegrityVerifier:
    """
    Model Integrity Verifier
    
    Verifies model weights, LoRA adapters, and prompt templates using
    Dilithium signatures and SHA-256 hashes.
    """
    
    def __init__(self, sal: Optional[PQSecurityAbstractionLayer] = None):
        """
        Initialize verifier.
        
        Args:
            sal: PQ-SAL instance (creates new one if None)
        """
        self.sal = sal or PQSecurityAbstractionLayer()
        self.trusted_public_keys: Dict[str, bytes] = {}  # model_id -> public_key
    
    def add_trusted_key(self, model_id: str, public_key: bytes):
        """
        Add trusted public key for model verification.
        
        Args:
            model_id: Model identifier
            public_key: Dilithium public key (bytes)
        """
        self.trusted_public_keys[model_id] = public_key
    
    def create_manifest(self, model_path: Path, model_id: str, 
                       model_type: str = "unknown", version: str = "1.0",
                       secret_key: Optional[bytes] = None) -> ModelManifest:
        """
        Create and sign model manifest.
        
        Args:
            model_path: Path to model directory
            model_id: Model identifier
            model_type: Model type (e.g., "qwen", "llama")
            version: Model version
            secret_key: Signer's secret key (if None, manifest won't be signed)
            
        Returns:
            ModelManifest instance
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
        # Collect all model files and compute hashes
        files: List[Tuple[str, str]] = []
        for pattern in ["*.safetensors", "*.bin", "*.pt", "*.json", "*.txt"]:
            for file_path in model_path.rglob(pattern):
                rel_path = file_path.relative_to(model_path)
                # 跳过 manifest 自身，避免“写入 manifest 后哈希失配”
                if rel_path.name == "manifest.json":
                    continue
                file_hash = self.sal.sha256_file(file_path).hex()
                files.append((str(rel_path), file_hash))
        
        # Check for LoRA adapters
        lora_adapters: Optional[List[Tuple[str, str]]] = None
        lora_path = model_path / "lora_adapters"
        if lora_path.exists():
            lora_adapters = []
            for lora_file in lora_path.rglob("*.safetensors"):
                rel_path = lora_file.relative_to(model_path)
                file_hash = self.sal.sha256_file(lora_file).hex()
                lora_adapters.append((str(rel_path), file_hash))
        
        # Load prompt templates if available
        prompt_templates: Optional[Dict[str, str]] = None
        template_file = model_path / "prompt_templates.json"
        if template_file.exists():
            prompt_templates = json.loads(template_file.read_text())
            # Hash prompt templates
            template_hash = self.sal.sha256(template_file.read_bytes()).hex()
            prompt_templates["_hash"] = template_hash
        
        # Create manifest
        manifest = ModelManifest(
            model_id=model_id,
            model_type=model_type,
            version=version,
            files=files,
            lora_adapters=lora_adapters,
            prompt_templates=prompt_templates,
        )
        
        # Sign manifest if secret key provided
        if secret_key:
            manifest_data = manifest.to_json_bytes()
            signature = self.sal.sign(manifest_data, secret_key)
            manifest.signature = signature.hex()
        
        return manifest
    
    def save_manifest(self, manifest: ModelManifest, output_path: Path):
        """
        Save manifest to file.
        
        Args:
            manifest: ModelManifest instance
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = asdict(manifest)
        output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    
    def load_manifest(self, manifest_path: Path) -> ModelManifest:
        """
        Load manifest from file.
        
        Args:
            manifest_path: Manifest file path
            
        Returns:
            ModelManifest instance
        """
        manifest_path = Path(manifest_path)
        data = json.loads(manifest_path.read_text())
        return ModelManifest.from_dict(data)
    
    def verify_manifest(self, manifest: ModelManifest) -> Tuple[bool, Optional[str]]:
        """
        Verify manifest signature.
        
        Args:
            manifest: ModelManifest instance
            
        Returns:
            (is_valid, error_message) tuple
        """
        if not manifest.signature:
            return False, "Manifest is not signed"
        
        public_key = self.trusted_public_keys.get(manifest.model_id)
        if not public_key:
            return False, f"No trusted public key for model: {manifest.model_id}"
        
        # Verify signature
        manifest_data = manifest.to_json_bytes()
        signature_bytes = bytes.fromhex(manifest.signature)
        
        is_valid = self.sal.verify(manifest_data, signature_bytes, public_key)
        if not is_valid:
            return False, "Signature verification failed"
        
        return True, None
    
    def verify_model_files(self, model_path: Path, manifest: ModelManifest) -> Tuple[bool, List[str]]:
        """
        Verify all model files match manifest hashes.
        
        Args:
            model_path: Path to model directory
            manifest: ModelManifest instance
            
        Returns:
            (all_valid, list_of_errors) tuple
        """
        model_path = Path(model_path)
        errors: List[str] = []
        
        # Verify regular files
        file_dict = dict(manifest.files)
        for rel_path, expected_hash in manifest.files:
            file_path = model_path / rel_path
            if not file_path.exists():
                errors.append(f"File not found: {rel_path}")
                continue
            
            actual_hash = self.sal.sha256_file(file_path).hex()
            if actual_hash != expected_hash:
                errors.append(f"Hash mismatch for {rel_path}: expected {expected_hash[:16]}..., got {actual_hash[:16]}...")
        
        # Verify LoRA adapters if present
        if manifest.lora_adapters:
            for rel_path, expected_hash in manifest.lora_adapters:
                file_path = model_path / rel_path
                if not file_path.exists():
                    errors.append(f"LoRA adapter not found: {rel_path}")
                    continue
                
                actual_hash = self.sal.sha256_file(file_path).hex()
                if actual_hash != expected_hash:
                    errors.append(f"LoRA hash mismatch for {rel_path}")
        
        # Verify prompt templates if present
        if manifest.prompt_templates and "_hash" in manifest.prompt_templates:
            template_file = model_path / "prompt_templates.json"
            if template_file.exists():
                actual_hash = self.sal.sha256(template_file.read_bytes()).hex()
                expected_hash = manifest.prompt_templates["_hash"]
                if actual_hash != expected_hash:
                    errors.append(f"Prompt template hash mismatch")
        
        return len(errors) == 0, errors
    
    def verify_complete(self, model_path: Path, manifest_path: Optional[Path] = None) -> Tuple[bool, List[str]]:
        """
        Complete verification: manifest signature + file hashes.
        
        Args:
            model_path: Path to model directory
            manifest_path: Path to manifest file (auto-detected if None)
            
        Returns:
            (is_valid, list_of_errors) tuple
        """
        model_path = Path(model_path)
        
        # Auto-detect manifest
        if manifest_path is None:
            manifest_path = model_path / "manifest.json"
        
        if not manifest_path.exists():
            return False, [f"Manifest not found: {manifest_path}"]
        
        # Load manifest
        try:
            manifest = self.load_manifest(manifest_path)
        except Exception as e:
            return False, [f"Failed to load manifest: {e}"]
        
        errors: List[str] = []
        
        # Verify signature
        sig_valid, sig_error = self.verify_manifest(manifest)
        if not sig_valid:
            errors.append(f"Signature verification failed: {sig_error}")
        
        # Verify files
        files_valid, file_errors = self.verify_model_files(model_path, manifest)
        if not files_valid:
            errors.extend(file_errors)
        
        return len(errors) == 0, errors
    
    def add_watermark(self, model_weights: bytes, watermark_data: bytes,
                     secret_key: bytes) -> bytes:
        """
        Add Dilithium-signed watermark to model weights.
        
        Args:
            model_weights: Model weights bytes
            watermark_data: Watermark information
            secret_key: Signer's secret key
            
        Returns:
            Watermarked weights (with embedded signature)
        """
        # Create watermark payload
        watermark_payload = {
            "data": watermark_data.hex(),
            "hash": self.sal.sha256(model_weights).hex(),
        }
        watermark_bytes = json.dumps(watermark_payload, sort_keys=True).encode()
        
        # Sign watermark
        signature = self.sal.sign(watermark_bytes, secret_key)
        
        # Embed in model metadata (simplified - in practice, use model-specific metadata)
        watermark_info = {
            "watermark": watermark_bytes.hex(),
            "signature": signature.hex(),
        }
        
        # In practice, embed in model config or metadata file
        return model_weights  # Simplified - actual implementation would embed in model format


