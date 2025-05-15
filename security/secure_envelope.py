# --- File: security/secure_envelope.py ---
from pydantic import BaseModel, Field
from typing import Dict, Optional, Any

class SecureEnvelope(BaseModel):
    """
    Represents a securely packaged message for inter-agent communication.
    This model itself is what's transferred; its fields contain the crypto components.
    """
    sender_id: str = Field(..., description="Unique ID of the sending agent/component.")
    recipient_id: str = Field(..., description="Unique ID of the intended receiving agent/component.")
    
    # For KEM (Kyber) + DEM (AES-GCM) scheme
    wrapped_aes_key_b64: str = Field(..., description="AES session key, KEM-wrapped with recipient's Kyber public key (base64).")
    
    # For AES-GCM encrypted payload
    aes_nonce_b64: str = Field(..., description="Nonce used for AES-GCM encryption of the payload (base64).")
    aes_tag_b64: str = Field(..., description="Authentication tag from AES-GCM encryption (base64).")
    encrypted_payload_b64: str = Field(..., description="The actual payload, AES-GCM encrypted (base64).")
    
    # For Digital Signature (Dilithium)
    signature_b64: str = Field(..., description="Dilithium signature over critical parts of the envelope (base64).")
    
    # Optional, non-encrypted metadata that is part of the signature.
    # Can include hints like payload_type for deserialization.
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata (e.g., payload_type_hint). Included in signature.")

    # To ensure the model is treated as a data class by FastAPI/Pydantic when used in type hints
    class Config:
        orm_mode = True # For Pydantic v1; for v2, use model_config = {"from_attributes": True}
        # For Pydantic v2, it would be:
        # model_config = {"from_attributes": True}
