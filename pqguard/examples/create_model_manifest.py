#!/usr/bin/env python3
"""
Create Model Manifest Example

Demonstrates how to create and sign a model manifest for integrity verification.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pqguard.sal import PQSecurityAbstractionLayer
from pqguard.model_integrity import ModelIntegrityVerifier


def main():
    print("=" * 60)
    print("Create Model Manifest Example")
    print("=" * 60)
    print()
    
    # Initialize PQ-SAL and verifier
    print("1. Initializing PQ-SAL...")
    sal = PQSecurityAbstractionLayer()
    verifier = ModelIntegrityVerifier(sal)
    print("✓ PQ-SAL initialized\n")
    
    # Generate signing key pair
    print("2. Generating model signing key pair...")
    public_key, secret_key = sal.generate_signature_keypair()
    print(f"✓ Key pair generated")
    print(f"  Public key (first 32 bytes): {public_key.hex()[:64]}...")
    print(f"  Secret key (first 32 bytes): {secret_key.hex()[:64]}...")
    print()
    
    # Save public key (for distribution)
    keys_dir = Path("./model_keys")
    keys_dir.mkdir(exist_ok=True)
    public_key_file = keys_dir / "qwen1.5-7b-chat.pub"
    public_key_file.write_bytes(public_key)
    print(f"✓ Public key saved to: {public_key_file}\n")
    
    # Create manifest for Qwen model
    model_path = Path("/root/private_data/.cache/huggingface/hub/models--Qwen--Qwen1.5-7B-Chat")
    
    if not model_path.exists():
        print(f"Error: Model path not found: {model_path}")
        print("Please update the model path in the script.")
        return
    
    print(f"3. Creating manifest for model: {model_path}")
    print("   (This may take a while for large models)...")
    
    manifest = verifier.create_manifest(
        model_path=model_path,
        model_id="Qwen/Qwen1.5-7B-Chat",
        model_type="qwen",
        version="1.0",
        secret_key=secret_key,
    )
    print(f"✓ Manifest created")
    print(f"  Model ID: {manifest.model_id}")
    print(f"  Model Type: {manifest.model_type}")
    print(f"  Version: {manifest.version}")
    print(f"  Files: {len(manifest.files)}")
    print(f"  Signature: {manifest.signature[:32]}..." if manifest.signature else "  (not signed)")
    print()
    
    # Save manifest
    manifest_path = model_path / "manifest.json"
    verifier.save_manifest(manifest, manifest_path)
    print(f"✓ Manifest saved to: {manifest_path}\n")
    
    # Verify manifest
    print("4. Verifying manifest...")
    verifier.add_trusted_key("Qwen/Qwen1.5-7B-Chat", public_key)
    is_valid, error = verifier.verify_manifest(manifest)
    
    if is_valid:
        print("✓ Manifest signature verified successfully\n")
    else:
        print(f"✗ Manifest verification failed: {error}\n")
    
    # Verify files
    print("5. Verifying model files...")
    files_valid, errors = verifier.verify_model_files(model_path, manifest)
    
    if files_valid:
        print("✓ All model files verified successfully\n")
    else:
        print(f"✗ File verification failed:")
        for error in errors[:5]:  # Show first 5 errors
            print(f"  - {error}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more errors")
        print()
    
    print("=" * 60)
    print("Model manifest creation completed!")
    print("=" * 60)
    print()
    print("Next steps:")
    print(f"1. Distribute public key: {public_key_file}")
    print(f"2. Distribute manifest: {manifest_path}")
    print(f"3. Use manifest for integrity verification in PQGuard pipeline")


if __name__ == "__main__":
    main()


