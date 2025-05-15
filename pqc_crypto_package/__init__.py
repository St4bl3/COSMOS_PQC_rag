# pqc_crypto_package/__init__.py

"""
PQC Crypto Package (Using kyber-py, pydilithium, PyCryptodome)
This package provides functionalities for Post-Quantum Cryptography using standalone
Python libraries and symmetric encryption using PyCryptodome, including:
- Key generation for Kyber768 (kyber-py) and Dilithium3 (pydilithium)
- Key Encapsulation Mechanism (KEM) operations using Kyber768 (kyber-py)
- Symmetric encryption/decryption using AES-GCM (PyCryptodome)
- Digital signature generation and verification using Dilithium3 (pydilithium)
"""
import logging

logger = logging.getLogger(__name__)

# Ensure correct submodule names are used in imports
try:
    from .digi_sign import sign_message, verify_signature
    from .kem_operations import kem_unwrap_symmetric_key, kem_wrap_symmetric_key
    from .key_generation import generate_dilithium_keypair, generate_kyber_keypair
    from .symmetric_ciphers import aes_gcm_decrypt, aes_gcm_encrypt
    PQC_CRYPTO_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import submodules for pqc_crypto_package: {e}. PQC functionalities will be unavailable.")
    PQC_CRYPTO_AVAILABLE = False
    # Define dummy functions if imports fail, so the rest of the application doesn't break on import
    def _dummy_func(*args, **kwargs):
        logger.error("PQC Crypto function called, but the package is not properly initialized due to import errors.")
        return None
    
    def _dummy_verify_func(*args, **kwargs):
        logger.error("PQC Crypto verify function called, but the package is not properly initialized.")
        return False

    generate_kyber_keypair = _dummy_func
    generate_dilithium_keypair = _dummy_func
    kem_wrap_symmetric_key = _dummy_func
    kem_unwrap_symmetric_key = _dummy_func
    aes_gcm_encrypt = _dummy_func
    aes_gcm_decrypt = _dummy_func
    sign_message = _dummy_func
    verify_signature = _dummy_verify_func


__all__ = [
    "generate_kyber_keypair",
    "generate_dilithium_keypair",
    "kem_wrap_symmetric_key",
    "kem_unwrap_symmetric_key",
    "aes_gcm_encrypt",
    "aes_gcm_decrypt",
    "sign_message",
    "verify_signature",
    "PQC_CRYPTO_AVAILABLE"
]

if PQC_CRYPTO_AVAILABLE:
    logger.info("PQC Crypto Package Initialized (Using kyber-py, pydilithium, PyCryptodome)")
else:
    logger.warning("PQC Crypto Package partially or fully unavailable due to import errors. Check dependencies.")