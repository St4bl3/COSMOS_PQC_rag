# --- File: security/key_manager.py ---
import json
import os
import logging
from typing import Dict, Tuple, Optional

# Assuming your pqc_crypto_package is in the project root or PYTHONPATH
from pqc_crypto_package import key_generation, PQC_CRYPTO_AVAILABLE

logger = logging.getLogger(__name__)

# Define a list of known agent IDs. This should be comprehensive.
# These IDs will be used to generate and retrieve keys.
KNOWN_AGENT_IDS = [
    "Orchestrator", # The main orchestrator agent
    "SOPAgent",
    "SystemCmdAgent",
    "DisasterAgent",
    "AsteroidAgent",
    "GroundCmdAgent",
    "CodeGenAgent"
    # Add any other components/agents that will communicate securely
]
DEFAULT_KEY_FILE = "agent_pqc_keys.json" # Consider making this configurable via config.py

class KeyManager:
    """
    Manages PQC cryptographic keys for all known agents.
    Loads keys from a file or generates them if the file doesn't exist or an agent is missing.
    """
    def __init__(self, key_file_path: str = DEFAULT_KEY_FILE):
        self.key_file_path = key_file_path
        self.agent_keys: Dict[str, Dict[str, str]] = {}
        self._is_dirty = False # Flag to track if keys were generated/changed and need saving
        self._load_or_generate_keys()

    def _load_or_generate_keys(self):
        """Loads keys from the key file or generates new keys if necessary."""
        if not PQC_CRYPTO_AVAILABLE:
            logger.critical("PQC crypto libraries not available. Key management is disabled. THIS IS INSECURE.")
            return

        if os.path.exists(self.key_file_path):
            try:
                with open(self.key_file_path, 'r') as f:
                    self.agent_keys = json.load(f)
                logger.info(f"Successfully loaded agent keys from {self.key_file_path}")
            except (IOError, json.JSONDecodeError) as e:
                logger.error(f"Error loading keys from {self.key_file_path}: {e}. Will attempt to regenerate all missing keys.")
                self.agent_keys = {} # Reset if loading failed, to ensure regeneration

        # Check for missing keys for any known agent and generate them
        for agent_id in KNOWN_AGENT_IDS:
            if agent_id not in self.agent_keys or not self.agent_keys[agent_id]: # Check if entry exists and is not empty
                logger.warning(f"Keys not found or incomplete for agent '{agent_id}'. Generating new keys.")
                self._generate_and_store_keys_for_agent(agent_id)
        
        if self._is_dirty: # If any new keys were generated or file was created
            self._save_keys()

    def _generate_and_store_keys_for_agent(self, agent_id: str):
        """Generates and stores Kyber and Dilithium key pairs for a given agent ID."""
        if not PQC_CRYPTO_AVAILABLE:
            logger.error(f"Cannot generate keys for '{agent_id}': PQC crypto libraries unavailable.")
            return

        logger.info(f"Generating PQC keys for agent: {agent_id}...")
        kyber_pk_b64, kyber_sk_b64 = key_generation.generate_kyber_keypair()
        dilithium_pk_b64, dilithium_sk_b64 = key_generation.generate_dilithium_keypair()

        if not (kyber_pk_b64 and kyber_sk_b64 and dilithium_pk_b64 and dilithium_sk_b64):
            logger.error(f"Failed to generate one or more PQC keys for agent '{agent_id}'. This agent will have incomplete keys and may not function securely.")
            # Store empty dict to indicate an attempt was made but failed, preventing re-attempt every time
            self.agent_keys[agent_id] = {} 
        else:
            self.agent_keys[agent_id] = {
                "kyber_pk_b64": kyber_pk_b64,
                "kyber_sk_b64": kyber_sk_b64,
                "dilithium_pk_b64": dilithium_pk_b64,
                "dilithium_sk_b64": dilithium_sk_b64,
            }
            logger.info(f"Successfully generated and staged keys for agent '{agent_id}'.")
        self._is_dirty = True # Mark that changes need saving

    def _save_keys(self):
        """Saves the current agent keys to the JSON file."""
        if not PQC_CRYPTO_AVAILABLE:
            logger.warning("Cannot save keys: PQC crypto libraries unavailable.")
            return
        if not self._is_dirty:
            logger.debug("No changes to agent keys, skipping save.")
            return
            
        try:
            # Ensure the directory for the key file exists
            key_dir = os.path.dirname(self.key_file_path)
            if key_dir and not os.path.exists(key_dir):
                os.makedirs(key_dir, exist_ok=True)
                logger.info(f"Created directory for key file: {key_dir}")

            with open(self.key_file_path, 'w') as f:
                json.dump(self.agent_keys, f, indent=4)
            logger.info(f"Agent PQC keys saved successfully to {self.key_file_path}")
            self._is_dirty = False
        except IOError as e:
            logger.error(f"CRITICAL: Error saving agent PQC keys to {self.key_file_path}: {e}")

    def get_public_kyber_key(self, agent_id: str) -> Optional[str]:
        """Retrieves the Kyber public key (base64) for a given agent ID."""
        if not PQC_CRYPTO_AVAILABLE: return None
        keys = self.agent_keys.get(agent_id)
        if keys and keys.get("kyber_pk_b64"):
            return keys["kyber_pk_b64"]
        logger.warning(f"Kyber public key not found for agent_id: {agent_id}")
        return None

    def get_private_kyber_key(self, agent_id: str) -> Optional[str]:
        """Retrieves the Kyber private key (base64) for a given agent ID."""
        if not PQC_CRYPTO_AVAILABLE: return None
        keys = self.agent_keys.get(agent_id)
        if keys and keys.get("kyber_sk_b64"):
            return keys["kyber_sk_b64"]
        logger.warning(f"Kyber private key not found for agent_id: {agent_id}")
        return None

    def get_public_dilithium_key(self, agent_id: str) -> Optional[str]:
        """Retrieves the Dilithium public key (base64) for a given agent ID."""
        if not PQC_CRYPTO_AVAILABLE: return None
        keys = self.agent_keys.get(agent_id)
        if keys and keys.get("dilithium_pk_b64"):
            return keys["dilithium_pk_b64"]
        logger.warning(f"Dilithium public key not found for agent_id: {agent_id}")
        return None

    def get_private_dilithium_key(self, agent_id: str) -> Optional[str]:
        """Retrieves the Dilithium private key (base64) for a given agent ID."""
        if not PQC_CRYPTO_AVAILABLE: return None
        keys = self.agent_keys.get(agent_id)
        if keys and keys.get("dilithium_sk_b64"):
            return keys["dilithium_sk_b64"]
        logger.warning(f"Dilithium private key not found for agent_id: {agent_id}")
        return None

