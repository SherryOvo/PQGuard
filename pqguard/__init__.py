"""
PQGuard: Post-Quantum Secure LLM Inference Framework

An end-to-end post-quantum security architecture that integrates NIST-standardized
PQC schemes (CRYSTALS-Kyber, CRYSTALS-Dilithium, SPHINCS+) into the complete LLM
lifecycle: model distribution, loading, inference, and output auditing.
"""

__version__ = "1.0.0"

from pqguard.sal import PQSecurityAbstractionLayer
from pqguard.model_integrity import ModelIntegrityVerifier
from pqguard.session_encryption import SecureSessionManager
from pqguard.audit_log import AuditLogger
from pqguard.llm_pipeline import PQGuardLLMPipeline

__all__ = [
    "PQSecurityAbstractionLayer",
    "ModelIntegrityVerifier",
    "SecureSessionManager",
    "AuditLogger",
    "PQGuardLLMPipeline",
]


