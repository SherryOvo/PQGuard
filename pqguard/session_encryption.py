#!/usr/bin/env python3
"""
Secure Session Manager

Establishes encrypted inference sessions using:
- Kyber KEM for key exchange
- AES-256-GCM for symmetric encryption

Provides low-latency encrypted communication for LLM inference.
"""

import time
import secrets
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass, asdict
from pqguard.sal import PQSecurityAbstractionLayer


@dataclass
class SessionInfo:
    """Session information."""
    session_id: str
    server_public_key: bytes
    client_public_key: Optional[bytes]
    session_key: bytes
    created_at: float
    last_used: float
    request_count: int = 0


class SecureSessionManager:
    """
    Secure Session Manager
    
    Manages encrypted inference sessions using Kyber KEM and AES-256-GCM.
    """
    
    def __init__(self, sal: Optional[PQSecurityAbstractionLayer] = None,
                 session_timeout: int = 3600):
        """
        Initialize session manager.
        
        Args:
            sal: PQ-SAL instance
            session_timeout: Session timeout in seconds (default: 1 hour)
        """
        self.sal = sal or PQSecurityAbstractionLayer()
        self.session_timeout = session_timeout
        self.server_public_key, self.server_secret_key = self.sal.generate_kem_keypair()
        self.sessions: Dict[str, SessionInfo] = {}
    
    def get_server_public_key(self) -> bytes:
        """
        Get server's KEM public key for key exchange.
        
        Returns:
            Server's Kyber public key
        """
        return self.server_public_key
    
    def create_session(self, kem_ciphertext: bytes, 
                      client_context: Optional[bytes] = None) -> Tuple[str, bytes]:
        """
        Create new session from client's KEM ciphertext (server side).
        
        Args:
            kem_ciphertext: Client's Kyber encapsulation ciphertext
            client_context: Optional client context for key derivation
            
        Returns:
            (session_id, confirmation_token) tuple
        """
        # Decapsulate shared secret
        shared_secret = self.sal.kem_decapsulate(self.server_secret_key, kem_ciphertext)
        
        # Derive session key
        context = client_context or b"PQGuard-Session"
        session_key = self.sal.derive_session_key(shared_secret, context)
        
        # Create session
        session_id = secrets.token_urlsafe(32)
        session = SessionInfo(
            session_id=session_id,
            server_public_key=self.server_public_key,
            client_public_key=None,  # Not needed for server
            session_key=session_key,
            created_at=time.time(),
            last_used=time.time(),
        )
        self.sessions[session_id] = session
        
        # Create confirmation token (encrypted session_id)
        confirmation = self.sal.encrypt(session_key, session_id.encode(), 
                                       b"session-confirm")
        
        return session_id, confirmation
    
    def establish_session(self, server_public_key: bytes,
                         client_context: Optional[bytes] = None) -> Tuple[bytes, str, bytes]:
        """
        Establish session from server's public key (client side).
        
        Args:
            server_public_key: Server's Kyber public key
            client_context: Optional client context
            
        Returns:
            (kem_ciphertext, session_id, confirmation_token) tuple
        """
        # Encapsulate shared secret
        kem_ciphertext, shared_secret = self.sal.kem_encapsulate(server_public_key)
        
        # Derive session key (client computes same key)
        context = client_context or b"PQGuard-Session"
        session_key = self.sal.derive_session_key(shared_secret, context)
        
        # Create temporary session info (server will create actual session)
        # Client stores session_key locally for decryption
        session_id = secrets.token_urlsafe(32)
        
        return kem_ciphertext, session_id, session_key
    
    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """
        Get session by ID (checks timeout).
        
        Args:
            session_id: Session identifier
            
        Returns:
            SessionInfo if valid, None if expired/not found
        """
        session = self.sessions.get(session_id)
        if not session:
            return None
        
        # Check timeout
        if time.time() - session.last_used > self.session_timeout:
            del self.sessions[session_id]
            return None
        
        session.last_used = time.time()
        session.request_count += 1
        return session
    
    def encrypt_request(self, session_id: str, plaintext: bytes,
                       associated_data: Optional[bytes] = None) -> bytes:
        """
        Encrypt inference request.
        
        Args:
            session_id: Session identifier
            plaintext: Request data
            associated_data: Optional authenticated data
            
        Returns:
            Encrypted request
        """
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Invalid or expired session: {session_id}")
        
        # Store current request count before increment
        req_num = session.request_count - 1
        aad = associated_data or f"req-{req_num}".encode()
        ciphertext = self.sal.encrypt(session.session_key, plaintext, aad)
        # Note: request_count was already incremented in get_session
        return ciphertext
    
    def decrypt_request(self, session_id: str, ciphertext: bytes,
                       associated_data: Optional[bytes] = None) -> bytes:
        """
        Decrypt inference request.
        
        Args:
            session_id: Session identifier
            ciphertext: Encrypted request
            associated_data: Optional authenticated data
            
        Returns:
            Decrypted request
        """
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Invalid session: {session_id}")
        
        # Use request count before increment (since encrypt_request incremented it)
        # We need to match the AAD used during encryption
        req_num = session.request_count - 1
        aad = associated_data or f"req-{req_num}".encode()
        return self.sal.decrypt(session.session_key, ciphertext, aad)
    
    def encrypt_response(self, session_key: bytes, plaintext: bytes,
                        associated_data: Optional[bytes] = None) -> bytes:
        """
        Encrypt inference response (client or server).
        
        Args:
            session_key: Session encryption key
            plaintext: Response data
            associated_data: Optional authenticated data
            
        Returns:
            Encrypted response
        """
        aad = associated_data or b"response"
        return self.sal.encrypt(session_key, plaintext, aad)
    
    def decrypt_response(self, session_key: bytes, ciphertext: bytes,
                        associated_data: Optional[bytes] = None) -> bytes:
        """
        Decrypt inference response (client or server).
        
        Args:
            session_key: Session encryption key
            ciphertext: Encrypted response
            associated_data: Optional authenticated data
            
        Returns:
            Decrypted response
        """
        aad = associated_data or b"response"
        return self.sal.decrypt(session_key, ciphertext, aad)
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions."""
        current_time = time.time()
        expired = [
            sid for sid, session in self.sessions.items()
            if current_time - session.last_used > self.session_timeout
        ]
        for sid in expired:
            del self.sessions[sid]
    
    def get_session_stats(self) -> Dict[str, Any]:
        """
        Get session statistics.
        
        Returns:
            Dictionary with session statistics
        """
        self.cleanup_expired_sessions()
        return {
            "active_sessions": len(self.sessions),
            "total_requests": sum(s.request_count for s in self.sessions.values()),
            "server_public_key_hex": self.server_public_key.hex()[:32] + "...",
        }

