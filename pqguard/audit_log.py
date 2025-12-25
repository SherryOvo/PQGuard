#!/usr/bin/env python3
"""
Audit Log System

Provides legally-binding non-repudiation through SPHINCS+-signed audit logs
anchored in immutable storage.

Production requirement:
- Audit signing keys MUST be persistent across process restarts, otherwise
  old log entries cannot be verified.
"""

import json
import time
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pqguard.sal import PQSecurityAbstractionLayer


@dataclass
class AuditEntry:
    """Single audit log entry."""
    timestamp: float
    session_id: str
    model_id: str
    model_version_hash: str
    request_hash: str
    response_hash: str
    user_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    signature: Optional[str] = None  # SPHINCS+ signature (hex)
    
    def to_json_bytes(self, include_signature: bool = False) -> bytes:
        """Convert to JSON bytes for signing."""
        data = asdict(self)
        if not include_signature:
            data.pop("signature", None)
        return json.dumps(data, sort_keys=True).encode()
    
    @classmethod
    def from_dict(cls, data: dict):
        """Create entry from dictionary."""
        return cls(**data)


class AuditLogger:
    """
    Audit Logger with SPHINCS+ Signatures
    
    Creates legally-binding audit logs with non-repudiation guarantees.
    """
    
    def __init__(self, sal: Optional[PQSecurityAbstractionLayer] = None,
                 log_dir: Optional[Path] = None):
        """
        Initialize audit logger.
        
        Args:
            sal: PQ-SAL instance
            log_dir: Directory for audit logs (default: ./audit_logs)
        """
        self.sal = sal or PQSecurityAbstractionLayer()
        self.log_dir = Path(log_dir) if log_dir else Path("./audit_logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Key storage directory (for persistence across restarts)
        self.key_dir = self.log_dir / "keys"
        self.key_dir.mkdir(parents=True, exist_ok=True)

        # Load or generate persistent audit signing key pair.
        # In production this should ideally be managed by HSM/KMS,
        # but file-based persistence is sufficient as a reference implementation.
        self.audit_public_key, self.audit_secret_key = self._load_or_generate_keys(self.key_dir)
        
        # Batch signing (for efficiency)
        self.pending_entries: List[AuditEntry] = []
        self.batch_size = 10  # Sign every N entries

    # ------------------------------------------------------------------
    # Key management
    # ------------------------------------------------------------------

    def _load_or_generate_keys(self, key_dir: Path) -> Tuple[bytes, bytes]:
        """
        Load existing audit key pair from disk, or generate and persist a new one.

        This ensures that audit log signatures remain verifiable across
        process restarts.
        """
        key_dir = Path(key_dir)
        pk_path = key_dir / "audit_public_key.bin"
        sk_path = key_dir / "audit_secret_key.bin"

        if pk_path.exists() and sk_path.exists():
            public_key = pk_path.read_bytes()
            secret_key = sk_path.read_bytes()
            return public_key, secret_key

        # Generate new key pair and persist
        public_key, secret_key = self.sal.generate_audit_keypair()
        pk_path.write_bytes(public_key)
        sk_path.write_bytes(secret_key)

        # Best-effort harden secret key file permissions (POSIX only)
        try:
            os.chmod(sk_path, 0o600)
        except Exception:
            pass

        return public_key, secret_key
    
    def save_public_key(self, output_path: Path):
        """
        Save audit public key for verification.
        
        Args:
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.write_bytes(self.audit_public_key)
    
    def load_public_key(self, key_path: Path) -> bytes:
        """
        Load audit public key for verification.
        
        Args:
            key_path: Key file path
            
        Returns:
            Public key bytes
        """
        return Path(key_path).read_bytes()
    
    def create_entry(self, session_id: str, model_id: str,
                    model_version_hash: str, request_data: bytes,
                    response_data: bytes, user_id: Optional[str] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> AuditEntry:
        """
        Create audit log entry.
        
        Args:
            session_id: Session identifier
            model_id: Model identifier
            model_version_hash: Hash of model version/weights
            request_data: Request data (will be hashed)
            response_data: Response data (will be hashed)
            user_id: Optional user identifier
            metadata: Optional additional metadata
            
        Returns:
            AuditEntry instance
        """
        request_hash = self.sal.sha256(request_data).hex()
        response_hash = self.sal.sha256(response_data).hex()
        
        entry = AuditEntry(
            timestamp=time.time(),
            session_id=session_id,
            model_id=model_id,
            model_version_hash=model_version_hash,
            request_hash=request_hash,
            response_hash=response_hash,
            user_id=user_id,
            metadata=metadata or {},
        )
        
        return entry
    
    def sign_entry(self, entry: AuditEntry) -> AuditEntry:
        """
        Sign audit entry with SPHINCS+.
        
        Args:
            entry: AuditEntry to sign
            
        Returns:
            Signed entry
        """
        entry_data = entry.to_json_bytes(include_signature=False)
        signature = self.sal.sign_audit_log(entry_data, self.audit_secret_key)
        entry.signature = signature.hex()
        return entry
    
    def verify_entry(self, entry: AuditEntry, public_key: Optional[bytes] = None) -> bool:
        """
        Verify audit entry signature.
        
        Args:
            entry: AuditEntry to verify
            public_key: Public key (uses instance key if None)
            
        Returns:
            True if signature is valid
        """
        if not entry.signature:
            return False
        
        public_key = public_key or self.audit_public_key
        entry_data = entry.to_json_bytes(include_signature=False)
        signature_bytes = bytes.fromhex(entry.signature)
        
        return self.sal.verify_audit_log(entry_data, signature_bytes, public_key)
    
    def log(self, entry: AuditEntry, sign: bool = True, 
           immediate_flush: bool = False) -> Path:
        """
        Write audit entry to log file.
        
        Args:
            entry: AuditEntry to log
            sign: Whether to sign entry
            immediate_flush: Whether to flush to disk immediately
            
        Returns:
            Path to log file
        """
        if sign:
            entry = self.sign_entry(entry)
        
        # Add to batch
        self.pending_entries.append(entry)
        
        # Flush if batch size reached or immediate flush requested
        if len(self.pending_entries) >= self.batch_size or immediate_flush:
            return self.flush()
        
        # Return current log file path (for appending)
        return self._get_log_file_path()
    
    def flush(self) -> Path:
        """
        Flush pending entries to log file.
        
        Returns:
            Path to log file
        """
        if not self.pending_entries:
            return self._get_log_file_path()
        
        log_file = self._get_log_file_path()
        
        # Append entries (JSONL format)
        with open(log_file, "a") as f:
            for entry in self.pending_entries:
                entry_dict = asdict(entry)
                f.write(json.dumps(entry_dict, ensure_ascii=False) + "\n")
        
        # Create batch signature (optional - for additional integrity)
        batch_hash = self._hash_batch(self.pending_entries)
        batch_sig = self.sal.sign_audit_log(batch_hash, self.audit_secret_key)
        
        # Write batch signature to separate file
        batch_sig_file = log_file.parent / f"{log_file.stem}.batch_sig"
        batch_sig_file.write_bytes(batch_sig)
        
        self.pending_entries.clear()
        return log_file
    
    def _get_log_file_path(self) -> Path:
        """Get log file path for current date."""
        date_str = time.strftime("%Y-%m-%d")
        return self.log_dir / f"audit_{date_str}.jsonl"
    
    def _hash_batch(self, entries: List[AuditEntry]) -> bytes:
        """Compute hash of batch of entries."""
        combined = b"".join(e.to_json_bytes(include_signature=True) for e in entries)
        return self.sal.sha256(combined)
    
    def verify_log_file(self, log_file: Path, public_key: Optional[bytes] = None) -> Tuple[bool, List[str]]:
        """
        Verify all entries in log file.
        
        Args:
            log_file: Path to log file
            public_key: Public key for verification
            
        Returns:
            (all_valid, list_of_errors) tuple
        """
        public_key = public_key or self.audit_public_key
        errors: List[str] = []
        
        with open(log_file, "r") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    entry = AuditEntry.from_dict(data)
                    
                    if not self.verify_entry(entry, public_key):
                        errors.append(f"Line {line_num}: Invalid signature")
                    
                except Exception as e:
                    errors.append(f"Line {line_num}: {e}")
        
        return len(errors) == 0, errors
    
    def query_logs(self, start_time: Optional[float] = None,
                  end_time: Optional[float] = None,
                  session_id: Optional[str] = None,
                  model_id: Optional[str] = None) -> List[AuditEntry]:
        """
        Query audit logs.
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp
            session_id: Filter by session ID
            model_id: Filter by model ID
            
        Returns:
            List of matching entries
        """
        # Flush pending entries first
        self.flush()
        
        entries: List[AuditEntry] = []
        
        # Search all log files
        for log_file in sorted(self.log_dir.glob("audit_*.jsonl")):
            with open(log_file, "r") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        entry = AuditEntry.from_dict(data)
                        
                        # Apply filters
                        if start_time and entry.timestamp < start_time:
                            continue
                        if end_time and entry.timestamp > end_time:
                            continue
                        if session_id and entry.session_id != session_id:
                            continue
                        if model_id and entry.model_id != model_id:
                            continue
                        
                        entries.append(entry)
                    except Exception:
                        continue
        
        return entries
    
    def anchor_to_immutable_storage(self, log_file: Path, 
                                   storage_path: Optional[Path] = None):
        """
        Anchor log file to immutable storage (simplified - in practice,
        use WORM storage, blockchain, or S3 Object Lock).
        
        Args:
            log_file: Log file to anchor
            storage_path: Immutable storage path (default: ./immutable_audit)
        """
        storage_path = storage_path or Path("./immutable_audit")
        storage_path.mkdir(parents=True, exist_ok=True)
        
        # Compute file hash
        file_hash = self.sal.sha256_file(log_file).hex()
        
        # Create anchor entry
        anchor_entry = {
            "log_file": str(log_file),
            "hash": file_hash,
            "timestamp": time.time(),
        }
        
        # Sign anchor
        anchor_data = json.dumps(anchor_entry, sort_keys=True).encode()
        anchor_sig = self.sal.sign_audit_log(anchor_data, self.audit_secret_key)
        anchor_entry["signature"] = anchor_sig.hex()
        
        # Save anchor
        anchor_file = storage_path / f"anchor_{log_file.stem}.json"
        anchor_file.write_text(json.dumps(anchor_entry, indent=2))
        
        # In production: also copy to WORM storage or create blockchain entry


