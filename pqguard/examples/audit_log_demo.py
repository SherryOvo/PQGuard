#!/usr/bin/env python3
"""
Audit Log Demo

Demonstrates audit logging with SPHINCS+ signatures and querying.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pqguard.audit_log import AuditLogger
from pqguard.sal import PQSecurityAbstractionLayer


def main():
    print("=" * 60)
    print("Audit Log Demo")
    print("=" * 60)
    print()
    
    # Initialize audit logger
    print("1. Initializing Audit Logger...")
    sal = PQSecurityAbstractionLayer()
    logger = AuditLogger(sal, log_dir=Path("./demo_audit_logs"))
    print("✓ Audit logger initialized")
    print(f"  Audit log directory: {logger.log_dir}")
    print(f"  Public key (first 32 bytes): {logger.audit_public_key.hex()[:64]}...")
    print()
    
    # Save public key
    public_key_file = logger.log_dir / "audit_public_key.pem"
    logger.save_public_key(public_key_file)
    print(f"✓ Public key saved to: {public_key_file}\n")
    
    # Create some audit entries
    print("2. Creating audit log entries...")
    
    entries_data = [
        {
            "session_id": "session-001",
            "model_id": "Qwen/Qwen1.5-7B-Chat",
            "model_version_hash": "abc123...",
            "request": "What is post-quantum cryptography?",
            "response": "Post-quantum cryptography is...",
            "user_id": "user-alice",
        },
        {
            "session_id": "session-001",
            "model_id": "Qwen/Qwen1.5-7B-Chat",
            "model_version_hash": "abc123...",
            "request": "Explain Kyber.",
            "response": "Kyber is a key encapsulation mechanism...",
            "user_id": "user-alice",
        },
        {
            "session_id": "session-002",
            "model_id": "Qwen/Qwen2-VL-7B-Instruct",
            "model_version_hash": "def456...",
            "request": "Analyze this image.",
            "response": "The image shows...",
            "user_id": "user-bob",
        },
    ]
    
    for i, data in enumerate(entries_data, 1):
        entry = logger.create_entry(
            session_id=data["session_id"],
            model_id=data["model_id"],
            model_version_hash=data["model_version_hash"],
            request_data=data["request"].encode(),
            response_data=data["response"].encode(),
            user_id=data["user_id"],
        )
        
        log_file = logger.log(entry, immediate_flush=(i == len(entries_data)))
        print(f"  Entry {i}: {data['session_id']} - {data['model_id']}")
    
    print(f"✓ Created {len(entries_data)} audit log entries")
    print(f"  Log file: {log_file}\n")
    
    # Verify log file
    print("3. Verifying audit log file...")
    is_valid, errors = logger.verify_log_file(log_file)
    
    if is_valid:
        print("✓ All audit log entries verified successfully\n")
    else:
        print(f"✗ Verification failed:")
        for error in errors:
            print(f"  - {error}")
        print()
    
    # Query logs
    print("4. Querying audit logs...")
    
    # Query by session
    print("   Querying session-001...")
    entries = logger.query_logs(session_id="session-001")
    print(f"   ✓ Found {len(entries)} entries\n")
    
    # Query by model
    print("   Querying Qwen/Qwen1.5-7B-Chat...")
    entries = logger.query_logs(model_id="Qwen/Qwen1.5-7B-Chat")
    print(f"   ✓ Found {len(entries)} entries\n")
    
    # Query by time range
    print("   Querying entries from last hour...")
    one_hour_ago = time.time() - 3600
    entries = logger.query_logs(start_time=one_hour_ago)
    print(f"   ✓ Found {len(entries)} entries\n")
    
    # Anchor to immutable storage
    print("5. Anchoring log file to immutable storage...")
    immutable_dir = Path("./demo_immutable_audit")
    logger.anchor_to_immutable_storage(log_file, immutable_dir)
    print(f"✓ Log file anchored to: {immutable_dir}\n")
    
    print("=" * 60)
    print("Audit log demo completed!")
    print("=" * 60)
    print()
    print("Audit log provides:")
    print("- SPHINCS+ signatures for non-repudiation")
    print("- Queryable log entries")
    print("- Immutable storage anchoring")
    print("- Legal binding for compliance")


if __name__ == "__main__":
    main()


