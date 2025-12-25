#!/usr/bin/env python3
"""
PQ-SAL: Post-Quantum Security Abstraction Layer

Provides a unified interface for PQC operations:
- Key Encapsulation (Kyber)
- Digital Signatures (Dilithium, SPHINCS+)
- Hybrid Classic-PQC operations for transition period
"""

import os
import hashlib
import json
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes

try:
    import oqs
    PYOQS_AVAILABLE = True
except ImportError:
    PYOQS_AVAILABLE = False
    print("Warning: pyoqs not available. Install with: pip install pyoqs")


class PQSecurityAbstractionLayer:
    """
    Post-Quantum Security Abstraction Layer
    
    Encapsulates:
    - CRYSTALS-Kyber for key encapsulation
    - CRYSTALS-Dilithium for digital signatures
    - SPHINCS+ for stateless audit log signing
    - Hybrid classic-PQC operations
    """
    
    # Algorithm choices (NIST standardized)
    KEM_ALGORITHM = "Kyber768"  # NIST Level 3
    SIG_ALGORITHM = "Dilithium2"  # NIST Level 2
    # Use available SPHINCS+ variant (fallback to Dilithium if not available)
    AUDIT_SIG_ALGORITHM = None  # Will be auto-detected
    
    # AES-GCM key size (256 bits for post-quantum security)
    AES_KEY_SIZE = 32
    
    def __init__(self):
        """Initialize PQ-SAL."""
        if not PYOQS_AVAILABLE:
            raise RuntimeError("pyoqs is required for PQ-SAL. Install with: pip install pyoqs")
        self._verify_algorithms()
    
    def _verify_algorithms(self):
        """Verify that required algorithms are available."""
        with oqs.KeyEncapsulation(self.KEM_ALGORITHM) as _:
            pass  # Test KEM
        
        with oqs.Signature(self.SIG_ALGORITHM) as _:
            pass  # Test signature
        
        # Auto-detect SPHINCS+ algorithm
        if self.AUDIT_SIG_ALGORITHM is None:
            available_sigs = oqs.get_enabled_sig_mechanisms()
            # Try to find a SPHINCS+ variant
            sphincs_variants = [s for s in available_sigs if "SPHINCS" in s.upper()]
            if sphincs_variants:
                self.AUDIT_SIG_ALGORITHM = sphincs_variants[0]
            else:
                # Fallback to Dilithium for audit (less ideal but works)
                self.AUDIT_SIG_ALGORITHM = self.SIG_ALGORITHM
                print(f"Warning: SPHINCS+ not available, using {self.SIG_ALGORITHM} for audit logs")
        
        try:
            with oqs.Signature(self.AUDIT_SIG_ALGORITHM) as _:
                pass  # Test audit signature
        except Exception as e:
            # Fallback to Dilithium
            print(f"Warning: {self.AUDIT_SIG_ALGORITHM} not available, falling back to Dilithium2")
            self.AUDIT_SIG_ALGORITHM = self.SIG_ALGORITHM
    
    # ==================== Key Encapsulation (Kyber) ====================
    
    def generate_kem_keypair(self) -> Tuple[bytes, bytes]:
        """
        Generate Kyber key pair for key encapsulation.
        
        Returns:
            (public_key, secret_key) tuple
        """
        with oqs.KeyEncapsulation(self.KEM_ALGORITHM) as kem:
            public_key = kem.generate_keypair()
            secret_key = kem.export_secret_key()
        return public_key, secret_key
    
    def kem_encapsulate(self, public_key: bytes) -> Tuple[bytes, bytes]:
        """
        Generate shared secret using Kyber encapsulation (client side).
        
        Args:
            public_key: Server's public key
            
        Returns:
            (ciphertext, shared_secret) tuple
        """
        with oqs.KeyEncapsulation(self.KEM_ALGORITHM) as kem:
            ciphertext, shared_secret = kem.encap_secret(public_key)
        return ciphertext, shared_secret
    
    def kem_decapsulate(self, secret_key: bytes, ciphertext: bytes) -> bytes:
        """
        Recover shared secret using Kyber decapsulation (server side).
        
        Args:
            secret_key: Server's secret key
            ciphertext: Encapsulation ciphertext
            
        Returns:
            Shared secret
        """
        with oqs.KeyEncapsulation(self.KEM_ALGORITHM, secret_key=secret_key) as kem:
            shared_secret = kem.decap_secret(ciphertext)
        return shared_secret
    
    def derive_session_key(self, shared_secret: bytes, context: Optional[bytes] = None) -> bytes:
        """
        Derive AES-256-GCM session key from shared secret using HKDF.
        
        Args:
            shared_secret: Kyber shared secret
            context: Optional context for key derivation
            
        Returns:
            32-byte AES key
        """
        context = context or b"PQGuard-Session-Key"
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=self.AES_KEY_SIZE,
            salt=None,
            info=context,
        )
        return hkdf.derive(shared_secret)
    
    # ==================== Digital Signatures (Dilithium) ====================
    
    def generate_signature_keypair(self, algorithm: Optional[str] = None) -> Tuple[bytes, bytes]:
        """
        Generate signature key pair (Dilithium for model signing).
        
        Args:
            algorithm: Signature algorithm (default: Dilithium2)
            
        Returns:
            (public_key, secret_key) tuple
        """
        algorithm = algorithm or self.SIG_ALGORITHM
        with oqs.Signature(algorithm) as sig:
            public_key = sig.generate_keypair()
            secret_key = sig.export_secret_key()
        return public_key, secret_key
    
    def sign(self, data: bytes, secret_key: bytes, algorithm: Optional[str] = None) -> bytes:
        """
        Sign data using Dilithium.
        
        Args:
            data: Data to sign
            secret_key: Signer's secret key
            algorithm: Signature algorithm (default: Dilithium2)
            
        Returns:
            Signature bytes
        """
        algorithm = algorithm or self.SIG_ALGORITHM
        with oqs.Signature(algorithm, secret_key=secret_key) as sig:
            signature = sig.sign(data)
        return signature
    
    def verify(self, data: bytes, signature: bytes, public_key: bytes, 
               algorithm: Optional[str] = None) -> bool:
        """
        Verify signature.
        
        Args:
            data: Original data
            signature: Signature to verify
            public_key: Signer's public key
            algorithm: Signature algorithm (default: Dilithium2)
            
        Returns:
            True if signature is valid, False otherwise
        """
        algorithm = algorithm or self.SIG_ALGORITHM
        try:
            with oqs.Signature(algorithm) as sig:
                return sig.verify(data, signature, public_key)
        except Exception:
            return False
    
    # ==================== Audit Log Signatures (SPHINCS+) ====================
    
    def generate_audit_keypair(self) -> Tuple[bytes, bytes]:
        """
        Generate SPHINCS+ key pair for audit log signing.
        
        Returns:
            (public_key, secret_key) tuple
        """
        algorithm = self.AUDIT_SIG_ALGORITHM or self.SIG_ALGORITHM
        return self.generate_signature_keypair(algorithm)
    
    def sign_audit_log(self, log_data: bytes, secret_key: bytes) -> bytes:
        """
        Sign audit log entry using SPHINCS+ (stateless).
        
        Args:
            log_data: Audit log entry
            secret_key: Audit signer's secret key
            
        Returns:
            SPHINCS+ signature
        """
        return self.sign(log_data, secret_key, self.AUDIT_SIG_ALGORITHM)
    
    def verify_audit_log(self, log_data: bytes, signature: bytes, 
                        public_key: bytes) -> bool:
        """
        Verify audit log signature.
        
        Args:
            log_data: Audit log entry
            signature: SPHINCS+ signature
            public_key: Audit signer's public key
            
        Returns:
            True if signature is valid
        """
        return self.verify(log_data, signature, public_key, self.AUDIT_SIG_ALGORITHM)
    
    # ==================== Symmetric Encryption (AES-256-GCM) ====================
    
    def encrypt(self, key: bytes, plaintext: bytes, 
                associated_data: Optional[bytes] = None) -> bytes:
        """
        Encrypt data using AES-256-GCM.
        
        Args:
            key: AES key (32 bytes)
            plaintext: Data to encrypt
            associated_data: Optional authenticated associated data
            
        Returns:
            Encrypted data (nonce || ciphertext || tag)
        """
        if len(key) != self.AES_KEY_SIZE:
            raise ValueError(f"Key must be {self.AES_KEY_SIZE} bytes")
        
        aes = AESGCM(key)
        nonce = os.urandom(12)  # 96-bit nonce for GCM
        associated_data = associated_data or b""
        ciphertext = aes.encrypt(nonce, plaintext, associated_data)
        return nonce + ciphertext  # Prepend nonce
    
    def decrypt(self, key: bytes, ciphertext: bytes,
                associated_data: Optional[bytes] = None) -> bytes:
        """
        Decrypt data using AES-256-GCM.
        
        Args:
            key: AES key (32 bytes)
            ciphertext: Encrypted data (nonce || ciphertext || tag)
            associated_data: Optional authenticated associated data
            
        Returns:
            Decrypted plaintext
        """
        if len(key) != self.AES_KEY_SIZE:
            raise ValueError(f"Key must be {self.AES_KEY_SIZE} bytes")
        
        if len(ciphertext) < 12:
            raise ValueError("Ciphertext too short")
        
        aes = AESGCM(key)
        nonce = ciphertext[:12]
        encrypted = ciphertext[12:]
        associated_data = associated_data or b""
        return aes.decrypt(nonce, encrypted, associated_data)
    
    # ==================== Hybrid Operations (Transition Support) ====================
    
    def hybrid_kem(self, classic_shared_secret: bytes, 
                   pq_shared_secret: bytes) -> bytes:
        """
        Combine classic and post-quantum shared secrets for hybrid security.
        
        Args:
            classic_shared_secret: ECDH/P-256 shared secret
            pq_shared_secret: Kyber shared secret
            
        Returns:
            Combined key using HKDF
        """
        combined = classic_shared_secret + pq_shared_secret
        return self.derive_session_key(combined, b"PQGuard-Hybrid-Key")
    
    # ==================== Utility Functions ====================
    
    @staticmethod
    def sha256(data: bytes) -> bytes:
        """Compute SHA-256 hash."""
        return hashlib.sha256(data).digest()
    
    @staticmethod
    def sha256_file(filepath: Path) -> bytes:
        """Compute SHA-256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)
        return sha256_hash.digest()
    
    @staticmethod
    def hash_model_weights(weights_path: Path) -> str:
        """
        Compute hash of model weights (for integrity verification).
        
        Args:
            weights_path: Path to model weights directory
            
        Returns:
            Hexadecimal hash string
        """
        hashes = []
        weights_path = Path(weights_path)
        
        # Collect all weight files
        for pattern in ["*.safetensors", "*.bin", "*.pt"]:
            for weight_file in weights_path.rglob(pattern):
                hashes.append(PQSecurityAbstractionLayer.sha256_file(weight_file).hex())
        
        # Sort for deterministic hash
        hashes.sort()
        combined = "".join(hashes).encode()
        return hashlib.sha256(combined).hexdigest()

