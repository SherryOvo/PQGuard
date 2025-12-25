#!/usr/bin/env python3
"""
PQGuard Basic Usage Example

Demonstrates:
1. Model integrity verification
2. Encrypted session establishment
3. Secure inference
4. Audit logging
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pqguard import PQGuardLLMPipeline


def main():
    print("=" * 60)
    print("PQGuard Basic Usage Example")
    print("=" * 60)
    print()
    
    # Initialize PQGuard pipeline
    # Note: Uses Qwen model from migrated cache location
    model_id = "/root/private_data/.cache/huggingface/hub/models--Qwen--Qwen1.5-7B-Chat"
    
    print("1. Initializing PQGuard LLM Pipeline...")
    pipeline = PQGuardLLMPipeline(
        model_id=model_id,
        model_type="qwen",
        verify_integrity=False,  # Set to True if you have manifest
        enable_encryption=True,
        enable_audit=True,
    )
    print("✓ Pipeline initialized\n")
    
    # Get server public key for session establishment
    print("2. Getting server public key for session establishment...")
    server_pk = pipeline.get_server_public_key()
    print(f"✓ Server public key (first 32 bytes): {server_pk.hex()[:64]}...\n")
    
    # Client side: Establish session
    print("3. Client: Establishing encrypted session...")
    from pqguard.session_encryption import SecureSessionManager
    client_session_mgr = SecureSessionManager()
    kem_ct, client_session_id, client_session_key = client_session_mgr.establish_session(server_pk)
    print(f"✓ Session established (client-side session key computed)\n")
    
    # Server side: Create session from client's KEM ciphertext
    print("4. Server: Creating session from client's KEM ciphertext...")
    session_id, confirmation = pipeline.establish_session(kem_ct)
    print(f"✓ Session created: {session_id[:16]}...\n")
    
    # Note: In practice, client would use client_session_key to decrypt responses
    # Server uses session_id to encrypt/decrypt
    
    # Generate with encrypted session
    print("5. Generating response with encrypted session...")
    prompt = "请解释一下后量子密码学（Post-Quantum Cryptography）的核心概念。"
    
    # Encrypt prompt (client side simulation)
    encrypted_prompt = client_session_mgr.encrypt_response(
        client_session_key, prompt.encode(), b"request"
    )
    print(f"✓ Prompt encrypted (length: {len(encrypted_prompt)} bytes)")
    
    # For demo, we'll use unencrypted prompt and let server encrypt response
    result = pipeline.generate(
        prompt=prompt,
        session_id=session_id,  # This enables encryption on server side
        max_new_tokens=256,
        temperature=0.7,
    )
    print(f"✓ Response generated\n")
    
    # Decrypt response (client side)
    if result.get("encrypted"):
        print("6. Client: Decrypting response...")
        encrypted_response = bytes.fromhex(result["response"])
        decrypted_response = client_session_mgr.decrypt_response(
            client_session_key, encrypted_response, b"response"
        )
        response = decrypted_response.decode()
        print(f"✓ Response decrypted\n")
    else:
        response = result["response"]
    
    print("=" * 60)
    print("Response:")
    print("=" * 60)
    print(response)
    print()
    
    print("=" * 60)
    print("Audit Log Entry Created")
    print("=" * 60)
    print(f"Session ID: {session_id[:16]}...")
    print(f"Model ID: {result['model_id']}")
    print(f"Model Version Hash: {result['model_version_hash'][:32]}...")
    print()
    
    # Query audit logs
    print("7. Querying audit logs...")
    audit_logs = pipeline.audit_logger.query_logs(session_id=session_id)
    print(f"✓ Found {len(audit_logs)} audit log entries for this session\n")
    
    print("=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()


