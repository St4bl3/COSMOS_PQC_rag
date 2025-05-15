# pqc_crypto_package/kem_operations.py
import base64
import json
import os  # Keep for os.urandom if needed by AES placeholders if they return

# Import the specific Kyber class and AES functions
try:

    from kyber_py.kyber import Kyber768
except ImportError:
    print("ERROR: Failed to import Kyber768 from kyber. Please install kyber-py: pip install kyber-py")
    Kyber768 = None # Set to None if import fails

from .symmetric_ciphers import aes_gcm_decrypt, aes_gcm_encrypt


def _kyber_encapsulate_internal(recipient_kyber_pk_b64):

    """
    Internal: Performs KEM encapsulation using kyber-py (Kyber768).
    Returns a Kyber ciphertext (b64) and a derived AES key (32 bytes).
    """
    if Kyber768 is None:
        print("PQC_MODULE_INTERNAL.kem: Kyber768 library not available.")
        return None, None

    # print(f"PQC_MODULE_INTERNAL.kem: Encapsulating with Kyber768 PK (first 10 chars): {recipient_kyber_pk_b64[:10]}...")
    try:
        public_key_bytes = base64.b64decode(recipient_kyber_pk_b64)

        # Call Kyber Encapsulation
        # kyber-py returns (shared_secret, ciphertext)
        actual_shared_secret_bytes, actual_ciphertext_bytes = Kyber768.encaps(public_key_bytes)

        # Verify expected lengths after assignment
        if len(actual_ciphertext_bytes) != 1088: # Expected Kyber768 ciphertext length
            print(f"[ERROR] PQC_MODULE_INTERNAL.kem: Kyber768.encaps() produced actual_ciphertext_bytes of unexpected length: {len(actual_ciphertext_bytes)}. Expected 1088.")
            return None, None
        if len(actual_shared_secret_bytes) != 32: # Expected Kyber shared secret length
            print(f"[ERROR] PQC_MODULE_INTERNAL.kem: Kyber768.encaps() produced actual_shared_secret_bytes of unexpected length: {len(actual_shared_secret_bytes)}. Expected 32.")
            return None, None

        from hashlib import sha256


        # Derive AES key from the ACTUAL shared secret
        aes_key = sha256(actual_shared_secret_bytes).digest()

        # Encode the ACTUAL ciphertext
        encoded_ciphertext_b64 = base64.b64encode(actual_ciphertext_bytes).decode('utf-8')

        return encoded_ciphertext_b64, aes_key

    except base64.binascii.Error as b64_error:
        print(f"PQC_MODULE_INTERNAL.kem: Base64 decoding error for public key: {b64_error}")
        return None, None
    except Exception as e:
        print(f"PQC_MODULE_INTERNAL.kem: Error during Kyber768 encapsulation with kyber-py: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def _kyber_decapsulate_internal(recipient_kyber_sk_b64, kyber_ciphertext_b64):
    """
    Internal: Performs KEM decapsulation using kyber-py (Kyber768).
    Returns the derived AES key (32 bytes).
    """
    if Kyber768 is None:
        print("PQC_MODULE_INTERNAL.kem: Kyber768 library not available for decapsulation.")
        return None
    # print(f"PQC_MODULE_INTERNAL.kem: Decapsulating with Kyber768 SK (first 10): {recipient_kyber_sk_b64[:10]} using CT (first 10): {kyber_ciphertext_b64[:10]}...")
    try:
        secret_key_bytes = base64.b64decode(recipient_kyber_sk_b64)
        ciphertext_bytes = base64.b64decode(kyber_ciphertext_b64)

        # Kyber768.decaps returns the shared_secret
        shared_secret_bytes = Kyber768.decaps(secret_key_bytes, ciphertext_bytes)

        if len(shared_secret_bytes) != 32: # Expected Kyber shared secret length
            print(f"[ERROR] PQC_MODULE_INTERNAL.kem: Kyber768.decaps() produced shared_secret_bytes of unexpected length: {len(shared_secret_bytes)}. Expected 32.")
            return None

        from hashlib import sha256
        aes_key = sha256(shared_secret_bytes).digest()

        return aes_key
    except Exception as e:
        print(f"PQC_MODULE_INTERNAL.kem: Error during Kyber768 decapsulation with kyber-py: {e}")
        return None

def kem_wrap_symmetric_key(symmetric_key_bytes_to_wrap, recipient_kyber_pk_b64):
    """
    Securely wraps an existing symmetric key using the recipient's Kyber public key (kyber-py).
    """
    # print(f"PQC_MODULE.kem_operations: KEM Wrapping {len(symmetric_key_bytes_to_wrap)}-byte key using Kyber768 for PK (first 10): {recipient_kyber_pk_b64[:10]}...")
    try:
        kyber_kem_ciphertext_b64, kem_derived_key_bytes = _kyber_encapsulate_internal(recipient_kyber_pk_b64)
        if kyber_kem_ciphertext_b64 is None or kem_derived_key_bytes is None:
            print("PQC_MODULE.kem_operations: Failed to encapsulate KEM-derived key (_kyber_encapsulate_internal failed).")
            return None
        
        aes_encrypted_package = aes_gcm_encrypt(symmetric_key_bytes_to_wrap, kem_derived_key_bytes)
        if aes_encrypted_package is None:
            print("PQC_MODULE.kem_operations: AES encryption of symmetric key failed.")
            return None
            
        wrapped_package = {
            "kem_ct_b64": kyber_kem_ciphertext_b64, 
            "aes_nonce_b64": aes_encrypted_package["nonce_b64"],
            "aes_encrypted_key_b64": aes_encrypted_package["ciphertext_b64"],
            "aes_tag_b64": aes_encrypted_package["tag_b64"]
        }
        return base64.b64encode(json.dumps(wrapped_package).encode('utf-8')).decode('utf-8')

    except Exception as e:
        print(f"PQC_MODULE.kem_operations: KEM key wrapping failed: {e}")
        return None

def kem_unwrap_symmetric_key(wrapped_key_package_b64, recipient_kyber_sk_b64):
    """
    Securely unwraps a symmetric key using the recipient's Kyber private key (kyber-py).
    """
    # print(f"PQC_MODULE.kem_operations: KEM Unwrapping key for SK (first 10): {recipient_kyber_sk_b64[:10]}...")
    try:
        wrapped_package_decoded = base64.b64decode(wrapped_key_package_b64).decode('utf-8')
        wrapped_package = json.loads(wrapped_package_decoded)
        
        kem_ct_b64 = wrapped_package["kem_ct_b64"]
        
        # Optional: Check length of KEM CT b64 string before decapsulation if needed for debugging
        # print(f"[DEBUG] KEM Ciphertext for decaps (base64) length: {len(kem_ct_b64)}")

        aes_encrypted_package_for_decryption = {
            "nonce_b64": wrapped_package["aes_nonce_b64"],
            "ciphertext_b64": wrapped_package["aes_encrypted_key_b64"],
            "tag_b64": wrapped_package["aes_tag_b64"]
        }

        kem_derived_key_bytes = _kyber_decapsulate_internal(recipient_kyber_sk_b64, kem_ct_b64)
        if kem_derived_key_bytes is None:
            print("PQC_MODULE.kem_operations: KEM decapsulation of derived key failed (_kyber_decapsulate_internal failed).")
            return None

        original_symmetric_key_bytes = aes_gcm_decrypt(
            aes_encrypted_package_for_decryption,
            kem_derived_key_bytes
        )
        if original_symmetric_key_bytes is None:
            print("PQC_MODULE.kem_operations: AES decryption of symmetric key failed.")
            return None
            
        return original_symmetric_key_bytes
        
    except json.JSONDecodeError as json_err:
        print(f"PQC_MODULE.kem_operations: KEM key unwrapping failed - JSON decoding error: {json_err}")
        return None
    except KeyError as key_err:
        print(f"PQC_MODULE.kem_operations: KEM key unwrapping failed - Missing key in JSON package: {key_err}")
        return None
    except Exception as e:
        print(f"PQC_MODULE.kem_operations: KEM key unwrapping failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    print(f"--- Testing KEM Operations with kyber-py (Kyber768) and PyCryptodome AES ---")
    from Crypto.Random import get_random_bytes as crypto_get_random_bytes
    
    try:
        if Kyber768 is None:
             print("Kyber768 not imported, skipping test.")
        else:
            recipient_pk_bytes, recipient_sk_bytes = Kyber768.keygen()
            
            recipient_pk_b64 = base64.b64encode(recipient_pk_bytes).decode('utf-8')
            recipient_sk_b64 = base64.b64encode(recipient_sk_bytes).decode('utf-8')

            print(f"Generated Kyber768 keypair for recipient.")
            print(f"  Recipient PK (raw len): {len(recipient_pk_bytes)}, SK (raw len): {len(recipient_sk_bytes)}")


            original_symmetric_key_to_wrap = crypto_get_random_bytes(32) 
            print(f"Original symmetric key to wrap (hex): {original_symmetric_key_to_wrap.hex()}")

            wrapped_package_b64 = kem_wrap_symmetric_key(original_symmetric_key_to_wrap, recipient_pk_b64)

            if wrapped_package_b64:
                print(f"Wrapped package (b64, first 60): {wrapped_package_b64[:60]}...")
                
                unwrapped_symmetric_key = kem_unwrap_symmetric_key(wrapped_package_b64, recipient_sk_b64)

                if unwrapped_symmetric_key:
                    print(f"Unwrapped symmetric key (hex): {unwrapped_symmetric_key.hex()}")
                    assert original_symmetric_key_to_wrap == unwrapped_symmetric_key, "Test FAILED: Unwrapped key does not match original!"
                    print("SUCCESS: KEM Wrap and Unwrap test passed!")
                else:
                    print("ERROR: Failed to unwrap the symmetric key.")
            else:
                print("ERROR: Failed to wrap the symmetric key.")
    except ImportError as ie:
        print(f"ImportError: {ie}. Make sure kyber-py and pycryptodomex are installed. Skipping KEM operations test.")
    except Exception as e:
        print(f"An error occurred during the KEM operations test: {e}")
        import traceback
        traceback.print_exc()
