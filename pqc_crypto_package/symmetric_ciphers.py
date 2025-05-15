# pqc_crypto_package/symmetric_ciphers.py
import base64
import binascii

from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes


def aes_gcm_encrypt(plaintext_bytes, aes_key_bytes):


    try:
        cipher = AES.new(aes_key_bytes, AES.MODE_GCM)
        nonce_bytes = cipher.nonce 
        ciphertext_bytes, tag_bytes = cipher.encrypt_and_digest(plaintext_bytes)
        
        return {
            "nonce_b64": base64.b64encode(nonce_bytes).decode('utf-8'),
            "ciphertext_b64": base64.b64encode(ciphertext_bytes).decode('utf-8'),
            "tag_b64": base64.b64encode(tag_bytes).decode('utf-8')
        }
    except (TypeError, ValueError) as key_error:
        print(f"PQC_MODULE.symmetric_ciphers: AES-GCM encryption error due to invalid key: {key_error}")
        raise # Re-raises the caught TypeError or ValueError
    except Exception as e:
        print(f"PQC_MODULE.symmetric_ciphers: AES-GCM encryption failed (unexpected): {e}")
        return None

def aes_gcm_decrypt(aes_encrypted_package, aes_key_bytes):
    if not isinstance(aes_encrypted_package, dict):
        print("PQC_MODULE.symmetric_ciphers: AES-GCM decryption failed: Encrypted package is not a dictionary or is None.")
        return None

    required_keys = ["nonce_b64", "ciphertext_b64", "tag_b64"]
    if not all(key in aes_encrypted_package for key in required_keys):
        print("PQC_MODULE.symmetric_ciphers: AES-GCM decryption failed: Missing required keys in package.")
        return None
    
    try:
        nonce_bytes = base64.b64decode(aes_encrypted_package["nonce_b64"])
        ciphertext_bytes = base64.b64decode(aes_encrypted_package["ciphertext_b64"])
        tag_bytes = base64.b64decode(aes_encrypted_package["tag_b64"])

        cipher = AES.new(aes_key_bytes, AES.MODE_GCM, nonce=nonce_bytes)
        plaintext_bytes = cipher.decrypt_and_verify(ciphertext_bytes, tag_bytes)
        return plaintext_bytes
    except (TypeError, ValueError) as crypto_error: 
        print(f"PQC_MODULE.symmetric_ciphers: AES-GCM decryption failed (crypto error or tag mismatch): {crypto_error}")
        return None
    except binascii.Error as b64_error: # Specifically catch base64 decoding errors
        print(f"PQC_MODULE.symmetric_ciphers: AES-GCM decryption failed (Base64 decoding error): {b64_error}")
        return None
    except Exception as e: # Catch other unexpected errors
        print(f"PQC_MODULE.symmetric_ciphers: AES-GCM decryption failed with unexpected error: {e}")
        return None

# Your __main__ block for standalone testing (can be kept as is or updated)
if __name__ == '__main__':
    print("--- Testing Symmetric Ciphers (AES-GCM with PyCryptodome) ---")
    key = get_random_bytes(32) 
    original_plaintext = b"This is a super secret message for AES-GCM!"
    print(f"Original Plaintext: {original_plaintext.decode()}")

    print("\nTesting encryption with invalid key type (string)...")
    try:
        aes_gcm_encrypt(original_plaintext, "thisisnotabyteskey")
    except TypeError as te:
        print(f"Correctly caught TypeError: {te}")

    print("\nTesting encryption with invalid key length (10 bytes)...")
    try:
        aes_gcm_encrypt(original_plaintext, get_random_bytes(10))
    except ValueError as ve:
        print(f"Correctly caught ValueError: {ve}")
    
    print("\nTesting encryption with empty key (b'')...")
    try:
        aes_gcm_encrypt(original_plaintext, b"")
    except ValueError as ve:
        print(f"Correctly caught ValueError for empty key: {ve}")

    encrypted_package = aes_gcm_encrypt(original_plaintext, key)
    if encrypted_package:
        print("\nEncryption successful with valid key.")
        print(f"  Nonce (b64): {encrypted_package['nonce_b64']}")
        print(f"  Ciphertext (b64): {encrypted_package['ciphertext_b64'][:30]}...")
        print(f"  Tag (b64): {encrypted_package['tag_b64']}")

        decrypted_plaintext = aes_gcm_decrypt(encrypted_package, key)
        if decrypted_plaintext is not None: # Check for None
            print(f"Decrypted Plaintext: {decrypted_plaintext.decode()}")
            assert original_plaintext == decrypted_plaintext
            print("SUCCESS: Decryption matched original plaintext!")
        else:
            print("ERROR: Decryption failed.")
            
        wrong_key = get_random_bytes(32)
        while wrong_key == key: # Ensure it's actually different
            wrong_key = get_random_bytes(32)
        print("\nTesting decryption with wrong key...")
        decrypted_with_wrong_key = aes_gcm_decrypt(encrypted_package, wrong_key)
        if decrypted_with_wrong_key is None:
            print("SUCCESS: Decryption correctly failed with wrong key.")
        else:
            print("ERROR: Decryption did not fail as expected with wrong key.")

        print("\nTesting decryption with tampered ciphertext...")
        tampered_package = encrypted_package.copy()
        original_ct_bytes = base64.b64decode(tampered_package["ciphertext_b64"])
        
        if original_ct_bytes: # Ensure there are bytes to tamper
            tampered_ct_bytes = original_ct_bytes[:-1] + bytes([(original_ct_bytes[-1] + 1) % 256])
            tampered_package["ciphertext_b64"] = base64.b64encode(tampered_ct_bytes).decode('utf-8')
        else: # If original ciphertext was empty, make tampered one non-empty
            tampered_package["ciphertext_b64"] = base64.b64encode(b"tampered_ciphertext").decode('utf-8')
        
        decrypted_with_tampered_ct = aes_gcm_decrypt(tampered_package, key)
        if decrypted_with_tampered_ct is None:
            print("SUCCESS: Decryption correctly failed with tampered ciphertext.")
        else:
            print("ERROR: Decryption did not fail as expected with tampered ciphertext.")
    else:
        print("ERROR: Encryption failed with valid key (this should not happen if code is correct).")

