#!/usr/bin/env python3
"""
PQGuard System Test

Quick test to verify all components are working correctly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from pqguard import (
            PQSecurityAbstractionLayer,
            ModelIntegrityVerifier,
            SecureSessionManager,
            AuditLogger,
            PQGuardLLMPipeline,
        )
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_sal():
    """Test PQ-SAL basic functionality."""
    print("\nTesting PQ-SAL...")
    try:
        from pqguard.sal import PQSecurityAbstractionLayer
        sal = PQSecurityAbstractionLayer()
        
        # Test KEM
        pk, sk = sal.generate_kem_keypair()
        ct, ss1 = sal.kem_encapsulate(pk)
        ss2 = sal.kem_decapsulate(sk, ct)
        assert ss1 == ss2, "KEM failed"
        print("  ✓ KEM key exchange works")
        
        # Test signature
        pk_sig, sk_sig = sal.generate_signature_keypair()
        data = b"test data"
        sig = sal.sign(data, sk_sig, sal.SIG_ALGORITHM)
        assert sal.verify(data, sig, pk_sig, sal.SIG_ALGORITHM), "Signature verification failed"
        print("  ✓ Digital signature works")
        
        # Test encryption
        key = b"a" * 32
        plaintext = b"secret message"
        ciphertext = sal.encrypt(key, plaintext)
        decrypted = sal.decrypt(key, ciphertext)
        assert decrypted == plaintext, "Encryption failed"
        print("  ✓ AES-GCM encryption works")
        
        print("✓ PQ-SAL tests passed")
        return True
    except Exception as e:
        print(f"✗ PQ-SAL test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_session_manager():
    """Test session manager."""
    print("\nTesting Session Manager...")
    try:
        from pqguard.session_encryption import SecureSessionManager
        from pqguard.sal import PQSecurityAbstractionLayer
        
        sal = PQSecurityAbstractionLayer()
        server = SecureSessionManager(sal)
        client = SecureSessionManager(sal)
        
        # Get server public key
        server_pk = server.get_server_public_key()
        
        # Client establishes session
        kem_ct, client_sid, client_key = client.establish_session(server_pk)
        
        # Server creates session
        session_id, confirmation = server.create_session(kem_ct)
        server_session = server.get_session(session_id)
        
        assert server_session is not None, "Session creation failed"
        
        # Test encryption/decryption
        plaintext = b"test request"
        ciphertext = server.encrypt_request(session_id, plaintext)
        decrypted = server.decrypt_request(session_id, ciphertext)
        assert decrypted == plaintext, "Request encryption failed"
        
        print("✓ Session Manager tests passed")
        return True
    except Exception as e:
        print(f"✗ Session Manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_audit_logger():
    """Test audit logger."""
    print("\nTesting Audit Logger...")
    try:
        from pqguard.audit_log import AuditLogger
        from pqguard.sal import PQSecurityAbstractionLayer
        import tempfile
        
        sal = PQSecurityAbstractionLayer()
        log_dir = Path(tempfile.mkdtemp())
        logger = AuditLogger(sal, log_dir=log_dir)
        
        # Create entry
        entry = logger.create_entry(
            session_id="test-session",
            model_id="test-model",
            model_version_hash="abc123",
            request_data=b"request",
            response_data=b"response",
        )
        
        # Sign and log
        signed_entry = logger.sign_entry(entry)
        log_file = logger.log(signed_entry, immediate_flush=True)
        
        # Verify
        is_valid, errors = logger.verify_log_file(log_file)
        assert is_valid, f"Audit log verification failed: {errors}"
        
        # Query
        entries = logger.query_logs(session_id="test-session")
        assert len(entries) > 0, "Query failed"
        
        print("✓ Audit Logger tests passed")
        return True
    except Exception as e:
        print(f"✗ Audit Logger test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("PQGuard System Test")
    print("=" * 60)
    print()
    
    tests = [
        ("Imports", test_imports),
        ("PQ-SAL", test_sal),
        ("Session Manager", test_session_manager),
        ("Audit Logger", test_audit_logger),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ {name} test crashed: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name:20} {status}")
    
    all_passed = all(r for _, r in results)
    
    print("=" * 60)
    if all_passed:
        print("All tests PASSED! PQGuard is ready to use.")
        return 0
    else:
        print("Some tests FAILED. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

