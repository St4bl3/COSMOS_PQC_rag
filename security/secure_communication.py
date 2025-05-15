# --- File: security/secure_communication.py ---
import json
import base64
import logging
from typing import Optional, Tuple, Union, Dict, Any
from Crypto.Random import get_random_bytes

from pqc_crypto_package import (
    kem_wrap_symmetric_key, kem_unwrap_symmetric_key,
    aes_gcm_encrypt, aes_gcm_decrypt,
    sign_message, verify_signature,
    PQC_CRYPTO_AVAILABLE
)
from .key_manager import KeyManager
from .secure_envelope import SecureEnvelope

logger = logging.getLogger(__name__)

class SecureCommunicator:
    """
    Handles packing (encrypting and signing) and unpacking (verifying and decrypting)
    messages for secure inter-agent communication using PQC.
    """
    def __init__(self, key_manager: KeyManager):
        self.key_manager = key_manager
        if not PQC_CRYPTO_AVAILABLE:
            logger.critical("SecureCommunicator initialized, but PQC_CRYPTO_AVAILABLE is False. Secure operations will fail or be bypassed.")

    def _get_data_to_sign(self, 
                          wrapped_aes_key_b64: str, 
                          aes_nonce_b64: str, 
                          encrypted_payload_b64: str, 
                          aes_tag_b64: str, 
                          sender_id: str, # Added sender_id to signature
                          recipient_id: str, 
                          metadata: Optional[Dict[str, Any]] = None
                         ) -> bytes:
        """
        Concatenates critical data elements in a consistent order for signing.
        Order is crucial for reproducible signature verification.
        """
        # Using a dictionary and then JSON dumping with sorted keys ensures order.
        data_for_signing_dict = {
            "w": wrapped_aes_key_b64, # wrapped_key
            "n": aes_nonce_b64,       # nonce
            "c": encrypted_payload_b64,# ciphertext
            "t": aes_tag_b64,         # tag
            "s": sender_id,           # sender
            "r": recipient_id,        # recipient
            "m": metadata if metadata else {} # metadata (empty dict if None)
        }
        # Sort keys for consistent string representation before encoding
        return json.dumps(data_for_signing_dict, sort_keys=True, separators=(',', ':')).encode('utf-8')

    def pack_message(self, 
                     payload: Any, # Can be Pydantic model, dict, list, str, bytes
                     sender_id: str, 
                     recipient_id: str,
                     payload_type_hint: Optional[str] = None # Hint for deserialization by recipient
                    ) -> Optional[SecureEnvelope]:
        """
        Encrypts and signs a payload for secure transmission.
        Returns a SecureEnvelope or None on failure.
        """
        if not (self.key_manager and PQC_CRYPTO_AVAILABLE):
            logger.error(f"SECURE PACK ({sender_id} -> {recipient_id}): Failed. PQC unavailable or KeyManager missing.")
            return None

        logger.info(f"SECURE PACK ({sender_id} -> {recipient_id}): Initiating secure packing.")

        sender_dilithium_sk_b64 = self.key_manager.get_private_dilithium_key(sender_id)
        recipient_kyber_pk_b64 = self.key_manager.get_public_kyber_key(recipient_id)

        if not sender_dilithium_sk_b64:
            logger.error(f"SECURE PACK ({sender_id} -> {recipient_id}): Sender Dilithium private key not found.")
            return None
        if not recipient_kyber_pk_b64:
            logger.error(f"SECURE PACK ({sender_id} -> {recipient_id}): Recipient Kyber public key not found.")
            return None

        try:
            # 1. Serialize payload to bytes (JSON for complex types)
            if hasattr(payload, 'model_dump_json'): # Pydantic v2 models
                payload_bytes = payload.model_dump_json(indent=None).encode('utf-8')
                effective_payload_type_hint = payload_type_hint or payload.__class__.__name__
            elif hasattr(payload, 'json'): # Pydantic v1 models
                payload_bytes = payload.json(indent=None).encode('utf-8') # No indent for compactness
                effective_payload_type_hint = payload_type_hint or payload.__class__.__name__
            elif isinstance(payload, (dict, list)):
                payload_bytes = json.dumps(payload, separators=(',', ':')).encode('utf-8')
                effective_payload_type_hint = payload_type_hint or type(payload).__name__
            elif isinstance(payload, str):
                payload_bytes = payload.encode('utf-8')
                effective_payload_type_hint = payload_type_hint or "str"
            elif isinstance(payload, bytes):
                payload_bytes = payload
                effective_payload_type_hint = payload_type_hint or "bytes"
            else:
                logger.error(f"SECURE PACK ({sender_id} -> {recipient_id}): Unsupported payload type: {type(payload)}.")
                return None
            
            # 2. Generate temporary AES key
            temp_aes_key_bytes = get_random_bytes(32) # AES-256

            # 3. Encrypt the payload with AES-GCM
            logger.debug(f"SECURE PACK ({sender_id} -> {recipient_id}): Encrypting payload with AES-GCM.")
            aes_encrypted_package = aes_gcm_encrypt(payload_bytes, temp_aes_key_bytes)
            if not aes_encrypted_package:
                logger.error(f"SECURE PACK ({sender_id} -> {recipient_id}): AES-GCM encryption failed.")
                return None
            
            # 4. Secure the AES key for the recipient (KEM Wrap)
            logger.debug(f"SECURE PACK ({sender_id} -> {recipient_id}): KEM-wrapping AES key for recipient.")
            wrapped_aes_key_b64 = kem_wrap_symmetric_key(temp_aes_key_bytes, recipient_kyber_pk_b64)
            if not wrapped_aes_key_b64:
                logger.error(f"SECURE PACK ({sender_id} -> {recipient_id}): KEM wrapping of AES key failed.")
                return None

            # 5. Prepare metadata (including payload type hint)
            envelope_metadata = {"payload_type": effective_payload_type_hint}

            # 6. Sign the package
            data_to_sign_bytes = self._get_data_to_sign(
                wrapped_aes_key_b64, 
                aes_encrypted_package["nonce_b64"],
                aes_encrypted_package["ciphertext_b64"], 
                aes_encrypted_package["tag_b64"],
                sender_id, # Include sender_id in signature
                recipient_id,
                envelope_metadata
            )
            logger.debug(f"SECURE PACK ({sender_id} -> {recipient_id}): Signing data.")
            signature_b64 = sign_message(data_to_sign_bytes, sender_dilithium_sk_b64)
            if not signature_b64:
                logger.error(f"SECURE PACK ({sender_id} -> {recipient_id}): Failed to sign message.")
                return None
            
            logger.info(f"SECURE PACK ({sender_id} -> {recipient_id}): Successfully packed and signed message. Payload type hint: {effective_payload_type_hint}.")
            return SecureEnvelope(
                sender_id=sender_id,
                recipient_id=recipient_id,
                wrapped_aes_key_b64=wrapped_aes_key_b64,
                aes_nonce_b64=aes_encrypted_package["nonce_b64"],
                aes_tag_b64=aes_encrypted_package["tag_b64"],
                encrypted_payload_b64=aes_encrypted_package["ciphertext_b64"],
                signature_b64=signature_b64,
                metadata=envelope_metadata
            )
        except Exception as e:
            logger.exception(f"SECURE PACK ({sender_id} -> {recipient_id}): Unexpected error during message packing: {e}")
            return None

    def unpack_message(self, 
                       envelope: SecureEnvelope, 
                       expected_recipient_id: str
                      ) -> Optional[Tuple[bytes, str, Dict[str, Any]]]: 
        """
        Verifies and decrypts a SecureEnvelope.
        Returns a tuple: (decrypted_payload_bytes, sender_id, metadata) or None on failure.
        """
        if not (self.key_manager and PQC_CRYPTO_AVAILABLE):
            logger.error(f"SECURE UNPACK (for {expected_recipient_id}): Failed. PQC unavailable or KeyManager missing.")
            return None

        logger.info(f"SECURE UNPACK (for {expected_recipient_id} from {envelope.sender_id}): Initiating unpacking.")

        if envelope.recipient_id != expected_recipient_id:
            logger.error(f"SECURE UNPACK (for {expected_recipient_id}): Recipient ID mismatch! Expected '{expected_recipient_id}', envelope to '{envelope.recipient_id}'. Discarding.")
            return None
        
        recipient_kyber_sk_b64 = self.key_manager.get_private_kyber_key(expected_recipient_id)
        sender_dilithium_pk_b64 = self.key_manager.get_public_dilithium_key(envelope.sender_id)

        if not recipient_kyber_sk_b64:
            logger.error(f"SECURE UNPACK (for {expected_recipient_id}): Recipient Kyber private key not found.")
            return None
        if not sender_dilithium_pk_b64:
            logger.error(f"SECURE UNPACK (for {expected_recipient_id}): Sender '{envelope.sender_id}' Dilithium public key not found.")
            return None

        try:
            # 1. Verify Signature
            data_to_verify_bytes = self._get_data_to_sign(
                envelope.wrapped_aes_key_b64, envelope.aes_nonce_b64, 
                envelope.encrypted_payload_b64, envelope.aes_tag_b64,
                envelope.sender_id, # Use sender_id from envelope for signature consistency
                envelope.recipient_id, 
                envelope.metadata
            )
            logger.debug(f"SECURE UNPACK (for {expected_recipient_id}): Verifying signature from {envelope.sender_id}.")
            if not verify_signature(data_to_verify_bytes, envelope.signature_b64, sender_dilithium_pk_b64):
                logger.error(f"SECURE UNPACK (for {expected_recipient_id}): Signature verification FAILED for message from '{envelope.sender_id}'.")
                return None
            logger.info(f"SECURE UNPACK (for {expected_recipient_id}): Signature VERIFIED for message from '{envelope.sender_id}'.")

            # 2. Unwrap the AES Key (KEM Decapsulate)
            logger.debug(f"SECURE UNPACK (for {expected_recipient_id}): KEM-unwrapping AES key.")
            unwrapped_aes_key_bytes = kem_unwrap_symmetric_key(envelope.wrapped_aes_key_b64, recipient_kyber_sk_b64)
            if not unwrapped_aes_key_bytes:
                logger.error(f"SECURE UNPACK (for {expected_recipient_id}): KEM unwrapping of AES key FAILED.")
                return None
            
            # 3. Decrypt the Message
            aes_package_for_decryption = {
                "nonce_b64": envelope.aes_nonce_b64,
                "ciphertext_b64": envelope.encrypted_payload_b64,
                "tag_b64": envelope.aes_tag_b64
            }
            logger.debug(f"SECURE UNPACK (for {expected_recipient_id}): Decrypting payload with AES-GCM.")
            decrypted_payload_bytes = aes_gcm_decrypt(aes_package_for_decryption, unwrapped_aes_key_bytes)
            if not decrypted_payload_bytes:
                logger.error(f"SECURE UNPACK (for {expected_recipient_id}): AES-GCM decryption FAILED.")
                return None
            
            logger.info(f"SECURE UNPACK (for {expected_recipient_id}): Successfully unpacked and decrypted message from '{envelope.sender_id}'. Payload length: {len(decrypted_payload_bytes)} bytes. Metadata: {envelope.metadata}")
            return decrypted_payload_bytes, envelope.sender_id, envelope.metadata or {}
        except Exception as e:
            logger.exception(f"SECURE UNPACK (for {expected_recipient_id}): Unexpected error during message unpacking: {e}")
            return None

