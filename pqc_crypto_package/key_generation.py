# pq
# pqc_crypto_package/key_generation.py

import base64
import os

# Import specific algorithm classes from the standalone libraries
try:

    from kyber_py.kyber import Kyber768  # From kyber-py library
except ImportError:
    print("ERROR: Failed to import Kyber768 from kyber. Please install kyber-py: pip install kyber-py")
    Kyber768 = None # Set to None if import fails

try:
    # Corrected import name
    from dilithium_py.dilithium import Dilithium3 
except ImportError:
    print("ERROR: Failed to import Dilithium3 from dilithium. Please install dilithium: pip install dilithium")
    Dilithium3 = None # Set to None if import fails

def generate_kyber_keypair():
    """
    Generates a Kyber public/private key pair using kyber-py (Kyber768).
    Returns:
        tuple: (public_key_b64, private_key_b64) or (None, None) on error.
    """
    if Kyber768 is None:
        print("PQC_MODULE.key_generation: Kyber768 not available due to import error.")
        return None, None
        
    print(f"PQC_MODULE.key_generation: Generating Kyber768 keypair using kyber-py...")
    try:
        # Generate keys using the keygen() method
        public_key_bytes, private_key_bytes = Kyber768.keygen()

        # Encode the keys in base64
        public_key_b64 = base64.b64encode(public_key_bytes).decode('utf-8')
        private_key_b64 = base64.b64encode(private_key_bytes).decode('utf-8')
        
        print(f"PQC_MODULE.key_generation: Successfully generated Kyber768 keypair.")
        return public_key_b64, private_key_b64

    except Exception as e:
        print(f"PQC_MODULE.key_generation: Error generating Kyber768 keypair with kyber-py: {e}")
        return None, None

def generate_dilithium_keypair():
    """
    Generates a Dilithium public/private key pair using dilithium lib (Dilithium3).
    Returns:
        tuple: (public_key_b64, private_key_b64) or (None, None) on error.
    """
    if Dilithium3 is None:
        print("PQC_MODULE.key_generation: Dilithium3 not available due to import error.")
        return None, None
        
    print(f"PQC_MODULE.key_generation: Generating Dilithium3 keypair using dilithium lib...")
    try:
        # Generate keys using the keygen() method
        public_key_bytes, private_key_bytes = Dilithium3.keygen()

        # Encode the keys in base64
        public_key_b64 = base64.b64encode(public_key_bytes).decode('utf-8')
        private_key_b64 = base64.b64encode(private_key_bytes).decode('utf-8')
        
        print(f"PQC_MODULE.key_generation: Successfully generated Dilithium3 keypair.")
        return public_key_b64, private_key_b64

    except Exception as e:
        print(f"PQC_MODULE.key_generation: Error generating Dilithium3 keypair with dilithium lib: {e}")
        return None, None

if __name__ == '__main__':
    print("--- Testing Key Generation (kyber-py & dilithium) ---")
    
    print("\nTesting Kyber key generation (kyber-py):")
    kyber_pk_b64, kyber_sk_b64 = generate_kyber_keypair()
    if kyber_pk_b64 and kyber_sk_b64:
        print(f"  Kyber768 Public Key (b64, first 30 chars): {kyber_pk_b64[:30]}...")
    else:
        print(f"  Kyber768 key generation failed or library not installed.")

    print("\nTesting Dilithium key generation (dilithium lib):")
    dilithium_pk_b64, dilithium_sk_b64 = generate_dilithium_keypair()
    if dilithium_pk_b64 and dilithium_sk_b64:
        print(f"  Dilithium3 Public Key (b64, first 30 chars): {dilithium_pk_b64[:30]}...")
    else:
        print(f"  Dilithium3 key generation failed or library not installed.")


