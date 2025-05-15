# --- File: agents/base_agent.py ---
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Type, Union # Added Union
from core.llm import LLMClient
from core.rag import RAGModule
from core.memory import TaskMemory
from security.secure_communication import SecureCommunicator # Import SecureCommunicator
from security.secure_envelope import SecureEnvelope # Import SecureEnvelope
from pqc_crypto_package import PQC_CRYPTO_AVAILABLE # Import global PQC flag
import logging
from typing import get_type_hints
import json

# Import Pydantic models that agents might expect after decryption
from ingestion.models import DisasterWeatherData, SpaceDebrisData, TextInputData, FileInputData 

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """Abstract base class for all specialized agents."""
    def __init__(
        self,
        agent_id: str, # Changed agent_name to agent_id for clarity in security context
        llm_client: LLMClient,
        rag_module: Optional[RAGModule] = None,
        memory: Optional[TaskMemory] = None,
        secure_communicator: Optional[SecureCommunicator] = None 
    ):
        self.agent_id = agent_id 
        self.agent_name = agent_id # For logging, can be the same as agent_id or a more descriptive name
        self.llm = llm_client
        self.rag = rag_module
        self.memory = memory
        self.secure_communicator = secure_communicator
        
        secure_status = "DISABLED"
        if self.secure_communicator and PQC_CRYPTO_AVAILABLE and config.ENABLE_PQC_SECURITY:
            secure_status = "ENABLED"
        elif self.secure_communicator and PQC_CRYPTO_AVAILABLE and not config.ENABLE_PQC_SECURITY:
            secure_status = "CONFIGURED_OFF"
        elif not PQC_CRYPTO_AVAILABLE:
            secure_status = "PQC_LIBS_MISSING"
            
        logger.info(f"Agent '{self.agent_name}' (ID: {self.agent_id}) initialized. Secure communicator status: {secure_status}.")


    @abstractmethod
    def get_system_message(self) -> str:
        """Returns the specific system message defining the agent's role."""
        pass

    @abstractmethod
    async def process(self, 
                      input_data_or_envelope: Union[Any, SecureEnvelope], # Input can be raw or an envelope
                      task_id: str, 
                      context: Optional[Dict[str, Any]] = None
                     ) -> Union[Any, SecureEnvelope]: # Output can also be raw or an envelope
        """
        Processes the input data and returns the agent's output.
        If secure_communicator is enabled and input is SecureEnvelope, it's unpacked.
        If secure_communicator is enabled, the output should be packed into a SecureEnvelope.
        """
        pass

    async def _generate_response(
        self,
        prompt: str,
        use_rag: bool = False,
        rag_query: Optional[str] = None,
        rag_top_k: int = 3,
        rag_filter: Optional[Dict[str, Any]] = None,
        **llm_kwargs
    ) -> str:
        """Helper method to generate LLM response, optionally using RAG."""
        system_message = self.get_system_message()
        logger.debug(f"Agent '{self.agent_name}': Generating response. Use RAG: {use_rag and bool(self.rag)}")

        if use_rag and self.rag:
            logger.info(f"Agent '{self.agent_name}': Using RAG for generation. Query: '{rag_query}'")
            return await self.rag.agenerate_with_rag(
                original_prompt=prompt, query_for_retrieval=rag_query,
                top_k=rag_top_k, filter=rag_filter,
                system_message=system_message, **llm_kwargs
            )
        else:
            if use_rag and not self.rag:
                logger.warning(f"Agent '{self.agent_name}': RAG requested but no RAG module available.")
            logger.info(f"Agent '{self.agent_name}': Using direct LLM generation.")
            return await self.llm.agenerate(
                prompt=prompt, system_message=system_message, **llm_kwargs
            )

    def _log_step(self, task_id: str, description: str, details: Optional[Dict] = None):
        """Logs a step using the memory module if available."""
        log_message = f"[{self.agent_name}] {description}"
        if self.memory:
            self.memory.log_step(task_id, log_message, details)
        else:
            # Fallback to standard logging if memory module isn't configured/available
            logger.info(f"[Task Log {task_id} via agent {self.agent_name} - No DB]: {log_message} {details or ''}")
    
    def _deserialize_payload(self, payload_bytes: bytes, payload_type_hint: Optional[str]) -> Optional[Any]:
        """
        Deserializes payload bytes based on a type hint.
        Type hint corresponds to the class name of the Pydantic model.
        """
        if not payload_type_hint:
            logger.warning(f"Agent {self.agent_id}: No payload_type_hint provided for deserialization. Attempting generic JSON load.")
            try:
                return json.loads(payload_bytes.decode('utf-8'))
            except Exception as e:
                logger.error(f"Agent {self.agent_id}: Failed to deserialize payload with generic JSON load: {e}. Payload (bytes prefix): {payload_bytes[:100]}")
                return None

        logger.debug(f"Agent {self.agent_id}: Attempting to deserialize payload to type: {payload_type_hint}")
        try:
            payload_dict = json.loads(payload_bytes.decode('utf-8'))
            
            # Map type hint string to actual Pydantic model class
            # This is a simplified mapping; a more robust solution might use a registry.
            model_map = {
                "DisasterWeatherData": DisasterWeatherData,
                "SpaceDebrisData": SpaceDebrisData,
                "TextInputData": TextInputData,
                "FileInputData": FileInputData,
                # Add other Pydantic models your agents might expect
            }

            target_class = model_map.get(payload_type_hint)
            if target_class:
                if hasattr(target_class, 'model_validate'): # Pydantic v2
                    return target_class.model_validate(payload_dict)
                elif hasattr(target_class, 'parse_obj'): # Pydantic v1
                    return target_class.parse_obj(payload_dict)
            
            # Fallback for basic types or if no specific model matches
            if payload_type_hint == "dict": return payload_dict
            if payload_type_hint == "list": return payload_dict # Assuming JSON list was parsed
            if payload_type_hint == "str": return payload_bytes.decode('utf-8') # If it was packed as raw string
            
            logger.warning(f"Agent {self.agent_id}: Unknown payload_type_hint '{payload_type_hint}' for specific Pydantic deserialization. Returning raw dict.")
            return payload_dict
        except json.JSONDecodeError as e:
            logger.error(f"Agent {self.agent_id}: JSON deserialization error for type '{payload_type_hint}': {e}. Payload (bytes prefix): {payload_bytes[:100]}")
            return None
        except Exception as e:
            logger.error(f"Agent {self.agent_id}: Error deserializing payload to type '{payload_type_hint}': {e}. Payload (bytes prefix): {payload_bytes[:100]}")
            return None

# Need to import config for ENABLE_PQC_SECURITY check in BaseAgent __init__
import config
