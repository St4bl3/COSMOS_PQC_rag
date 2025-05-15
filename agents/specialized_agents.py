# --- File: agents/specialized_agents.py ---
from agents.base_agent import BaseAgent 
from core.llm import LLMClient
from core.memory import TaskMemory
from core.rag import RAGModule
from ingestion.models import (
    DisasterWeatherData, SpaceDebrisData, TextInputData, FileInputData 
)
from typing import Dict, Any, Optional, List, Union
import json 
import logging 
import re 

# Security Imports
from security.secure_communication import SecureCommunicator 
from security.secure_envelope import SecureEnvelope 
from pqc_crypto_package import PQC_CRYPTO_AVAILABLE 
import config # To check config.ENABLE_PQC_SECURITY

logger = logging.getLogger(__name__)

# --- Helper for Deserialization (Ideally in BaseAgent) ---
def _deserialize_payload_static(
    payload_bytes: Optional[bytes], 
    payload_type_hint: Optional[str], 
    agent_id_for_logging: str
) -> Optional[Any]:
    if not payload_bytes:
        logger.warning(f"Agent {agent_id_for_logging}: Received empty payload_bytes for deserialization.")
        return None
    if not payload_type_hint:
        logger.warning(f"Agent {agent_id_for_logging}: No payload_type_hint provided for deserialization. Attempting generic JSON load.")
        try:
            return json.loads(payload_bytes.decode('utf-8'))
        except Exception as e:
            logger.error(f"Agent {agent_id_for_logging}: Failed to deserialize payload with generic JSON load: {e}. Payload (bytes prefix): {payload_bytes[:100]}")
            return None

    logger.debug(f"Agent {agent_id_for_logging}: Attempting to deserialize payload to type: {payload_type_hint}")
    payload_str = "" 
    try:
        payload_str = payload_bytes.decode('utf-8') 
        
        if payload_type_hint == "str":
             return payload_str

        payload_data = json.loads(payload_str)

        model_map = {
            "DisasterWeatherData": DisasterWeatherData,
            "SpaceDebrisData": SpaceDebrisData,
            "TextInputData": TextInputData,
            "FileInputData": FileInputData,
            "List[str]": lambda data: data if isinstance(data, list) else None,
            "List[Dict[str,Any]]": lambda data: data if isinstance(data, list) else None,
            "Dict[str,Any]": lambda data: data if isinstance(data, dict) else None,
        }
        target_class_or_loader = model_map.get(payload_type_hint)
        
        if target_class_or_loader:
            if callable(target_class_or_loader) and not isinstance(target_class_or_loader, type): 
                return target_class_or_loader(payload_data)
            elif hasattr(target_class_or_loader, 'model_validate'): 
                return target_class_or_loader.model_validate(payload_data)
            elif hasattr(target_class_or_loader, 'parse_obj'): # Pydantic v1
                return target_class_or_loader.parse_obj(payload_data) # type: ignore
        
        logger.warning(f"Agent {agent_id_for_logging}: Unknown payload_type_hint '{payload_type_hint}' for specific deserialization. Returning raw parsed JSON data: {type(payload_data)}")
        return payload_data
        
    except json.JSONDecodeError as e:
        logger.error(f"Agent {agent_id_for_logging}: JSON deserialization error for type '{payload_type_hint}': {e}. Payload (str prefix): {payload_str[:200]}")
        return None
    except UnicodeDecodeError as e:
        logger.error(f"Agent {agent_id_for_logging}: Unicode decoding error for type '{payload_type_hint}': {e}. Payload (bytes prefix): {payload_bytes[:100]}")
        return None
    except Exception as e:
        logger.error(f"Agent {agent_id_for_logging}: Error deserializing payload to type '{payload_type_hint}': {e}. Payload (str prefix): {payload_str[:200] if payload_str else payload_bytes[:100]}")
        return None


class SOPAgent(BaseAgent):
    def __init__(self, llm_client: LLMClient, rag_module: Optional[RAGModule], memory: Optional[TaskMemory], 
                 secure_communicator: Optional[SecureCommunicator], agent_id: str):
        super().__init__(agent_id, llm_client, rag_module, memory, secure_communicator)

    def get_system_message(self) -> str:
        return """You are an expert assistant specialized in creating clear, concise, and actionable Standard Operating Procedures (SOPs) or To-Do lists.
Input will be a description of a critical event (e.g., natural disaster, system failure, security incident) or a specific request for an SOP.
Your primary goal is to ensure safety, operational efficiency, and clarity in the generated procedures.
Use information provided about the event AND any relevant retrieved documents to generate the SOP.
Output the SOP ONLY as a valid JSON list of strings, where each string is a distinct, actionable step. Do NOT include any other text, commentary, or markdown formatting outside of this JSON list.
Example: ["Step 1: Immediately assess the situation for any immediate threats to personnel.", "Step 2: Notify the designated incident commander and provide a concise situation report.", "Step 3: Activate emergency communication channels as per protocol XYZ."]
Ensure each step is precise and unambiguous.
"""

    async def process(self, 
                      input_data_or_envelope: Union[DisasterWeatherData, TextInputData, SecureEnvelope], 
                      task_id: str, 
                      context: Optional[Dict[str, Any]] = None
                     ) -> Union[List[str], SecureEnvelope]:
        
        actual_input_data: Union[DisasterWeatherData, TextInputData]
        original_sender_id: str = context.get("caller_id", "Orchestrator") 

        if isinstance(input_data_or_envelope, SecureEnvelope):
            if not (self.secure_communicator and PQC_CRYPTO_AVAILABLE and config.ENABLE_PQC_SECURITY):
                logger.error(f"Agent {self.agent_id} received SecureEnvelope but security is not fully enabled/configured.")
                return self._pack_error_response(["Security misconfiguration: Cannot process envelope."], original_sender_id)

            original_sender_id = input_data_or_envelope.sender_id
            logger.info(f"Agent {self.agent_id} received secure envelope from {original_sender_id}. Unpacking...")
            unpacked_result = self.secure_communicator.unpack_message(
                envelope=input_data_or_envelope,
                expected_recipient_id=self.agent_id
            )
            if not unpacked_result:
                logger.error(f"Agent {self.agent_id} failed to unpack message from {original_sender_id}.")
                return self._pack_error_response([f"Failed to decrypt/verify input from {original_sender_id}."], original_sender_id)
            
            decrypted_payload_bytes, _, metadata = unpacked_result
            payload_type_hint = metadata.get("payload_type") if metadata else None
            
            deserialized = _deserialize_payload_static(decrypted_payload_bytes, payload_type_hint, self.agent_id)
            if deserialized is None or not isinstance(deserialized, (DisasterWeatherData, TextInputData)):
                logger.error(f"Agent {self.agent_id} failed to deserialize payload to DisasterWeatherData or TextInputData. Hint: {payload_type_hint}, Got: {type(deserialized)}")
                return self._pack_error_response([f"Payload deserialization error. Expected DWD/TID, got {type(deserialized)}."], original_sender_id)
            actual_input_data = deserialized
            logger.info(f"Agent {self.agent_id} successfully unpacked and deserialized message from {original_sender_id} as type {type(actual_input_data)}.")
        
        elif isinstance(input_data_or_envelope, (DisasterWeatherData, TextInputData)):
            actual_input_data = input_data_or_envelope
            if self.secure_communicator and PQC_CRYPTO_AVAILABLE and config.ENABLE_PQC_SECURITY:
                 logger.warning(f"Agent {self.agent_id} received plaintext {type(actual_input_data).__name__} while in secure mode. Sender: {original_sender_id}.")
        else:
            logger.error(f"Agent {self.agent_id} received unsupported input type: {type(input_data_or_envelope)}")
            return self._pack_error_response([f"Unsupported input type: {type(input_data_or_envelope)}"], original_sender_id)

        prompt_details = ""
        rag_query_context = "general emergency response"
        if isinstance(actual_input_data, DisasterWeatherData):
            self._log_step(task_id, f"Processing DisasterWeatherData for SOP", {"event": actual_input_data.event_type})
            prompt_details = f"Event Type: {actual_input_data.event_type}\nLocation: {actual_input_data.location}\nSeverity: {actual_input_data.severity or 'Not specified'}\nPredicted Impact: {actual_input_data.predicted_impact or 'Not specified'}\nConfidence: {actual_input_data.confidence_score or 'Not specified'}"
            rag_query_context = f"{actual_input_data.event_type} response procedures for {actual_input_data.location} severity {actual_input_data.severity}"
        elif isinstance(actual_input_data, TextInputData):
            self._log_step(task_id, f"Processing TextInputData for SOP", {"text_len": len(actual_input_data.text_content)})
            prompt_details = f"User Request for SOP: \"{actual_input_data.text_content}\""
            rag_query_context = f"SOP for: {actual_input_data.text_content[:150]}"
        
        prompt = f"""Based on the following situation, generate a detailed Standard Operating Procedure (SOP).
Situation Details:
{prompt_details}

Consider standard safety protocols, required immediate actions, communication procedures, and resource management.
If the situation involves multiple aspects (e.g., a cyberattack leading to a physical disaster), ensure the SOP covers all relevant response phases.
Base the SOP on established best practices and any relevant procedures found in retrieved documents.
Output ONLY the JSON list of strings, where each string is a clear, actionable step. Do NOT include any introductory or explanatory text.
"""
        if context and context.get('previous_steps'): prompt += f"\nConsider previous steps taken: {context['previous_steps']}"
        
        rag_query = f"Standard Operating Procedure for {rag_query_context}"
        rag_filter = {"doc_type": {"$in": ["sop", "safety_manual", "checklist", "emergency_procedure", "incident_response_plan"]}}
        response_text = await self._generate_response(prompt=prompt, use_rag=True, rag_query=rag_query, rag_filter=rag_filter, rag_top_k=7, temperature=0.2) 
        
        sop_list = self._parse_json_list_from_llm(response_text, task_id, "SOP")

        if self.secure_communicator and PQC_CRYPTO_AVAILABLE and config.ENABLE_PQC_SECURITY:
            logger.info(f"Agent {self.agent_id} packing SOP response for {original_sender_id}...")
            packed_output = self.secure_communicator.pack_message(
                payload=sop_list, sender_id=self.agent_id, recipient_id=original_sender_id, payload_type_hint="List[str]" 
            )
            return packed_output if packed_output else self._pack_error_response(["Failed to pack secure SOP response."], original_sender_id)
        return sop_list

    def _parse_json_list_from_llm(self, response_text: str, task_id: str, item_name: str) -> List[str]:
        items = []
        cleaned_text = ""
        try:
            logging.debug(f"Agent {self.agent_id}: Raw LLM response for {item_name}: '{response_text}'")
            cleaned_text = response_text.strip()
            match_md = re.search(r"```json\s*(.*?)\s*```", cleaned_text, re.DOTALL | re.IGNORECASE)
            if match_md:
                cleaned_text = match_md.group(1).strip()
            else:
                match_md_generic = re.search(r"```\s*(.*?)\s*```", cleaned_text, re.DOTALL | re.IGNORECASE)
                if match_md_generic:
                    cleaned_text = match_md_generic.group(1).strip()
            
            match_arr = re.search(r'^\s*(\[.*\])\s*$', cleaned_text, re.DOTALL) # Ensure full string is array
            if match_arr: potential_json_str = match_arr.group(1)
            else: potential_json_str = cleaned_text
            
            parsed_output = json.loads(potential_json_str)
            if isinstance(parsed_output, list) and all(isinstance(item, str) for item in parsed_output):
                items = parsed_output
                self._log_step(task_id, f"Successfully parsed {item_name} JSON with {len(items)} items.")
            else:
                items = [f"Error: Parsed {item_name} was not a list of strings, but {type(parsed_output)}"]
                self._log_step(task_id, f"Parsed {item_name} JSON was not a list of strings.", {"type": str(type(parsed_output)), "raw_parsed": str(parsed_output)[:200]})
        except json.JSONDecodeError as e:
            items = [f"Error: Could not parse {item_name} from LLM response - {e}. Cleaned text for parsing: '{cleaned_text[:200]}' Raw snippet: '{response_text[:100]}'"]
            self._log_step(task_id, f"Failed to parse {item_name} JSON.", {"error": str(e), "cleaned_text_attempt": cleaned_text[:200]})
        except Exception as e:
            items = [f"Error processing {item_name} LLM response: {e}"]
            self._log_step(task_id, f"Generic error processing {item_name}.", {"error": str(e)})
        return items

    def _pack_error_response(self, error_payload_list: List[str], recipient_id: str) -> Union[List[str], SecureEnvelope]:
        logger.error(f"Agent {self.agent_id}: Preparing error response for {recipient_id} - Content: {error_payload_list}")
        if self.secure_communicator and PQC_CRYPTO_AVAILABLE and config.ENABLE_PQC_SECURITY:
            packed_error = self.secure_communicator.pack_message(
                payload=error_payload_list, sender_id=self.agent_id, recipient_id=recipient_id, payload_type_hint="List[str]" 
            )
            return packed_error if packed_error else [f"CRITICAL_ERROR: Failed to pack error list. Original: {error_payload_list}"]
        return error_payload_list


class CodeGenAgent(BaseAgent):
    def __init__(self, llm_client: LLMClient, rag_module: Optional[RAGModule], memory: Optional[TaskMemory],
                 secure_communicator: Optional[SecureCommunicator], agent_id: str):
        super().__init__(agent_id, llm_client, rag_module, memory, secure_communicator)

    def get_system_message(self) -> str:
        return """You are an expert code generation assistant. Your task is to write clean, efficient, and correct code snippets based on user requirements.
If a programming language is specified in the request, use that language. Otherwise, default to Python.
The code should be well-commented, explaining its logic, inputs, and outputs.
Focus on fulfilling the requirements accurately and producing runnable code where applicable.
Output ONLY the raw code string. Do NOT wrap it in markdown code fences (e.g., ```python ... ```) or add any explanatory text before or after the code, unless it's part of a comment within the code itself.
If the request is ambiguous or lacks detail for a functional snippet, provide a foundational structure or ask for clarification within code comments.
"""

    async def process(self, 
                      input_data_or_envelope: Union[TextInputData, SecureEnvelope], 
                      task_id: str, 
                      context: Optional[Dict[str, Any]] = None
                     ) -> Union[str, SecureEnvelope]:
        actual_input_data: TextInputData
        original_sender_id: str = context.get("caller_id", "Orchestrator")

        if isinstance(input_data_or_envelope, SecureEnvelope):
            if not (self.secure_communicator and PQC_CRYPTO_AVAILABLE and config.ENABLE_PQC_SECURITY):
                return self._pack_error_response_str("Security misconfiguration for CodeGenAgent.", original_sender_id)
            
            original_sender_id = input_data_or_envelope.sender_id
            logger.info(f"Agent {self.agent_id} received secure envelope from {original_sender_id}. Unpacking...")
            unpacked = self.secure_communicator.unpack_message(input_data_or_envelope, self.agent_id)
            if not unpacked:
                return self._pack_error_response_str(f"Failed to decrypt/verify input from {original_sender_id}.", original_sender_id)
            
            decrypted_payload_bytes, _, metadata = unpacked
            payload_type_hint = metadata.get("payload_type")
            deserialized = _deserialize_payload_static(decrypted_payload_bytes, payload_type_hint, self.agent_id)

            if not isinstance(deserialized, TextInputData):
                logger.error(f"Agent {self.agent_id} failed to deserialize payload to TextInputData. Hint: {payload_type_hint}, Got: {type(deserialized)}")
                return self._pack_error_response_str(f"Payload error. Expected TextInputData, got {type(deserialized)}.", original_sender_id)
            actual_input_data = deserialized
            logger.info(f"Agent {self.agent_id} successfully unpacked message from {original_sender_id} as TextInputData.")
        elif isinstance(input_data_or_envelope, TextInputData):
            actual_input_data = input_data_or_envelope
        else:
            logger.error(f"Agent {self.agent_id} received unsupported input type: {type(input_data_or_envelope)}")
            return self._pack_error_response_str(f"Unsupported input type {type(input_data_or_envelope)} for CodeGenAgent.", original_sender_id)

        self._log_step(task_id, f"Processing input for Code Generation", {"req_len": len(actual_input_data.text_content)})
        language_match = re.search(r"(?:generate|write|create|in|for)\s+(python|javascript|java|c\+\+|c#|typescript|go|rust|bash|html|css|sql)\s+(?:code|script|function|snippet|program|query)", actual_input_data.text_content, re.IGNORECASE)
        language = "python" 
        if language_match: language = language_match.group(1).lower()
        self._log_step(task_id, f"Determined target language: {language}")

        prompt = f"""Generate a functional and well-commented code snippet in {language} based on the following requirements:

Requirements:
{actual_input_data.text_content}

Ensure the code directly addresses all specified requirements.
Include comments explaining complex logic, function parameters, and return values.
If the request implies a complete runnable script or function, provide that.
Output ONLY the raw code string. Do not include any introductory or concluding text, or markdown code fences.
"""
        rag_query = f"{language} code examples for: {actual_input_data.text_content[:120]}" 
        rag_filter = {"doc_type": {"$in": ["code_example", "library_documentation", "api_reference", "tutorial"]}, "language_tag": language}
        response_text = await self._generate_response(prompt=prompt, use_rag=True, rag_query=rag_query, rag_filter=rag_filter, rag_top_k=4, temperature=0.15)
        
        generated_code = self._clean_code_from_llm(response_text, language)
        
        if not generated_code.strip():
            error_code = f"// CodeGenAgent: No code generated or output was empty. LLM response was: {response_text[:200]}..."
            self._log_step(task_id, "Code generation resulted in empty output after cleaning.", {"raw_llm_response": response_text})
            return self._pack_error_response_str(error_code, original_sender_id)

        self._log_step(task_id, "Successfully generated code snippet.", {"code_len": len(generated_code), "lang": language})

        if self.secure_communicator and PQC_CRYPTO_AVAILABLE and config.ENABLE_PQC_SECURITY:
            logger.info(f"Agent {self.agent_id} packing CodeGen response for {original_sender_id}...")
            packed_output = self.secure_communicator.pack_message(
                generated_code, self.agent_id, original_sender_id, "str"
            )
            return packed_output if packed_output else self._pack_error_response_str("Failed to pack secure code response.", original_sender_id)
        return generated_code

    def _clean_code_from_llm(self, response_text: str, language: str) -> str:
        code = response_text.strip()
        # Try to find content within the most specific fence first
        specific_fence_pattern = re.compile(rf"^\s*```{language}\s*\n(.*?)\n```\s*$", re.DOTALL | re.MULTILINE | re.IGNORECASE)
        match = specific_fence_pattern.match(code)
        if match:
            code = match.group(1).strip()
            logging.info(f"Agent {self.agent_id}: Removed specific language markdown fences. Preview: '{code[:100]}...'")
            return code
        
        # Try generic fence if specific one not found
        generic_fence_pattern = re.compile(r"^\s*```\s*\n(.*?)\n```\s*$", re.DOTALL | re.MULTILINE)
        match = generic_fence_pattern.match(code)
        if match:
            code = match.group(1).strip()
            logging.info(f"Agent {self.agent_id}: Removed generic markdown fences. Preview: '{code[:100]}...'")
            return code
        
        # If no fences, assume it's raw code (or LLM didn't follow instructions)
        return code

    def _pack_error_response_str(self, error_message: str, recipient_id: str) -> Union[str, SecureEnvelope]:
        logger.error(f"Agent {self.agent_id}: Preparing error response for {recipient_id} - '{error_message}'")
        if self.secure_communicator and PQC_CRYPTO_AVAILABLE and config.ENABLE_PQC_SECURITY:
            packed_error = self.secure_communicator.pack_message(
                payload=error_message, sender_id=self.agent_id, recipient_id=recipient_id, payload_type_hint="str"
            )
            return packed_error if packed_error else f"CRITICAL_ERROR: Failed to pack error string. Original: {error_message}"
        return error_message


class DisasterAgent(BaseAgent):
    def __init__(self, llm_client: LLMClient, rag_module: Optional[RAGModule], memory: Optional[TaskMemory],
                 secure_communicator: Optional[SecureCommunicator], agent_id: str):
        super().__init__(agent_id, llm_client, rag_module, memory, secure_communicator)

    def get_system_message(self) -> str:
        return """You are a senior disaster analyst. Your task is to provide a comprehensive, structured analysis of a given disaster event or situation.
Focus on:
1.  **Detailed Impact Assessment**: Quantify where possible (e.g., estimated affected population, infrastructure damage extent).
2.  **Immediate and Secondary Risks**: Identify cascading effects, health hazards, security concerns.
3.  **Critical Resource Needs**: Specify types and urgency (e.g., medical teams, shelter units, water purification, specialized rescue equipment).
4.  **Key Stakeholders & Coordination Points**: Identify relevant agencies and organizations.
5.  **Long-term Recovery Considerations**: Outline initial thoughts on rebuilding, psychosocial support, and resilience building.
6.  **Confidence Level**: Provide an overall confidence level (High, Medium, Low) for your analysis based on the input data quality.
Use information from the input event and any relevant retrieved documents (e.g., historical data, geographical information, resource availability reports, similar incident reports).
Output your analysis ONLY as a valid JSON object with clear keys for each section (e.g., "summary", "detailed_impact", "identified_risks", "resource_requirements", "stakeholder_coordination", "recovery_outlook", "analysis_confidence"). Do NOT include any other text, commentary, or markdown formatting outside of this JSON object.
"""

    async def process(self, 
                      input_data_or_envelope: Union[DisasterWeatherData, TextInputData, SecureEnvelope], 
                      task_id: str, 
                      context: Optional[Dict[str, Any]] = None
                     ) -> Union[Dict[str, Any], SecureEnvelope]:
        actual_input_data: Union[DisasterWeatherData, TextInputData]
        original_sender_id: str = context.get("caller_id", "Orchestrator")

        if isinstance(input_data_or_envelope, SecureEnvelope):
            if not (self.secure_communicator and PQC_CRYPTO_AVAILABLE and config.ENABLE_PQC_SECURITY):
                return self._pack_error_response_dict({"error": "Security misconfiguration for DisasterAgent."}, original_sender_id)
            original_sender_id = input_data_or_envelope.sender_id
            unpacked = self.secure_communicator.unpack_message(input_data_or_envelope, self.agent_id)
            if not unpacked: return self._pack_error_response_dict({"error": f"Failed to unpack input from {original_sender_id}."}, original_sender_id)
            decrypted_bytes, _, metadata = unpacked
            payload_type_hint = metadata.get("payload_type")
            deserialized = _deserialize_payload_static(decrypted_bytes, payload_type_hint, self.agent_id)
            if not isinstance(deserialized, (DisasterWeatherData, TextInputData)):
                return self._pack_error_response_dict({"error": f"Payload error. Expected DWD or TID, got {type(deserialized)}."}, original_sender_id)
            actual_input_data = deserialized
        elif isinstance(input_data_or_envelope, (DisasterWeatherData, TextInputData)):
            actual_input_data = input_data_or_envelope
        else:
            return self._pack_error_response_dict({"error": f"Unsupported input type {type(input_data_or_envelope)} for DisasterAgent."}, original_sender_id)

        prompt_details = ""
        rag_query_context = "general disaster impact analysis"
        if isinstance(actual_input_data, DisasterWeatherData):
            self._log_step(task_id, f"Processing DWD for Disaster Analysis", {"event": actual_input_data.event_type})
            prompt_details = f"Event Type: {actual_input_data.event_type}\nLocation: {actual_input_data.location}\nSeverity: {actual_input_data.severity or 'N/A'}\nPredicted Impact: {actual_input_data.predicted_impact or 'N/A'}\nRaw Data: {actual_input_data.raw_inference_data or 'N/A'}"
            rag_query_context = f"impact assessment for {actual_input_data.event_type} in {actual_input_data.location}"
        elif isinstance(actual_input_data, TextInputData):
            self._log_step(task_id, f"Processing TID for Disaster Analysis", {"len": len(actual_input_data.text_content)})
            prompt_details = f"User Request for Disaster Analysis: \"{actual_input_data.text_content}\""
            rag_query_context = f"analysis of situation: {actual_input_data.text_content[:150]}"
        
        prompt = f"""Provide a comprehensive disaster analysis as a JSON object for the following situation:
{prompt_details}
Adhere strictly to the JSON structure and keys defined in the system message, including all specified keys.
Base your analysis on the provided information and any relevant documents retrieved.
Output ONLY the JSON object. No extra text or markdown.
"""
        if context and context.get('related_past_tasks'): prompt += f"\nConsider related past tasks: {str(context['related_past_tasks'])[:300]}..."
        rag_query = f"Detailed analysis, impact assessment, and resource needs for {rag_query_context}"
        rag_filter = {"doc_type": {"$in": ["disaster_report", "case_study", "geographical_data", "resource_map", "impact_assessment", "humanitarian_response_guide"]}}
        response_text = await self._generate_response(prompt=prompt, use_rag=True, rag_query=rag_query, rag_filter=rag_filter, rag_top_k=6, temperature=0.3) # Slightly lower temp for structured JSON
        
        analysis_report = self._parse_json_object_from_llm(response_text, task_id, "DisasterAnalysis")
        
        if self.secure_communicator and PQC_CRYPTO_AVAILABLE and config.ENABLE_PQC_SECURITY:
            return self.secure_communicator.pack_message(
                analysis_report, self.agent_id, original_sender_id, "Dict[str,Any]"
            ) or self._pack_error_response_dict({"error": "Failed to pack DisasterAgent response"}, original_sender_id)
        return analysis_report

    def _parse_json_object_from_llm(self, response_text: str, task_id: str, item_name: str) -> Dict[str, Any]:
        parsed_dict = {}
        cleaned_text = ""
        try:
            logging.debug(f"Agent {self.agent_id}: Raw LLM for {item_name}: '{response_text}'")
            cleaned_text = response_text.strip()
            match_md = re.search(r"```json\s*(.*?)\s*```", cleaned_text, re.DOTALL | re.IGNORECASE)
            if match_md: cleaned_text = match_md.group(1).strip()
            else:
                match_md_generic = re.search(r"```\s*(.*?)\s*```", cleaned_text, re.DOTALL | re.IGNORECASE)
                if match_md_generic: cleaned_text = match_md_generic.group(1).strip()
            
            match_obj = re.search(r'^\s*(\{.*\})\s*$', cleaned_text, re.DOTALL) # Ensure full string is object
            if match_obj: potential_json_str = match_obj.group(1)
            else: potential_json_str = cleaned_text
            
            parsed_output = json.loads(potential_json_str)
            if isinstance(parsed_output, dict):
                parsed_dict = parsed_output
                self._log_step(task_id, f"Successfully parsed {item_name} JSON object.")
            else:
                parsed_dict = {"error": f"Parsed {item_name} was not a dict, but {type(parsed_output)}"}
                self._log_step(task_id, f"Parsed {item_name} JSON was not a dict.", {"type": str(type(parsed_output))})
        except Exception as e:
            parsed_dict = {"error": f"Could not parse {item_name} from LLM - {e}. Cleaned text for parsing: '{cleaned_text[:200]}'. Raw snippet: '{response_text[:100]}'"}
            self._log_step(task_id, f"Failed to parse {item_name} JSON object.", {"error": str(e), "cleaned_text_attempt": cleaned_text[:200]})
        return parsed_dict

    def _pack_error_response_dict(self, error_dict: Dict[str,Any], recipient_id: str) -> Union[Dict[str,Any], SecureEnvelope]:
        if self.secure_communicator and PQC_CRYPTO_AVAILABLE and config.ENABLE_PQC_SECURITY:
            packed_error = self.secure_communicator.pack_message(
                payload=error_dict, sender_id=self.agent_id, recipient_id=recipient_id, payload_type_hint="Dict[str,Any]"
            )
            return packed_error if packed_error else {"error": f"CRITICAL_ERROR: Failed to pack error dict. Original: {error_dict.get('error')}"}
        return error_dict


class SystemCmdAgent(BaseAgent):
    def __init__(self, llm_client: LLMClient, rag_module: Optional[RAGModule], memory: Optional[TaskMemory], 
                 secure_communicator: Optional[SecureCommunicator], agent_id: str):
        super().__init__(agent_id, llm_client, rag_module, memory, secure_communicator)

    def get_system_message(self) -> str:
        return """You are an expert system responsible for generating precise, safe, and formatted System Commands, such as satellite maneuvers or critical infrastructure controls.
Input will be structured data (e.g., asteroid trajectories, system status) or a textual request describing the need for system commands.
You MUST adhere strictly to the required command format. Use provided context and retrieved technical documentation (e.g., command specifications, safety protocols).
Prioritize safety, accuracy, and adherence to operational constraints. If unsure, data is insufficient, or a safe command cannot be formulated, state that clearly by returning an empty JSON list `[]`.
Output ONLY a valid JSON list of command objects. Each command object must have 'command_name' (string), 'target_system' (string), 'parameters' (dictionary), and 'priority' (integer, e.g., 1-High, 2-Medium, 3-Low). Do NOT include any other text, commentary, or markdown formatting outside of this JSON list.
Example: `[{"command_name": "THRUST_EXEC", "target_system": "SAT_XYZ_CONTROL", "parameters": {"axis": "X", "duration_ms": 500, "thrust_level": 0.8, "checksum": "a1b2c3d4"}, "priority": 1}]`
"""

    async def process(self, 
                      input_data_or_envelope: Union[SpaceDebrisData, TextInputData, SecureEnvelope], 
                      task_id: str, 
                      context: Optional[Dict[str, Any]] = None
                     ) -> Union[List[Dict[str, Any]], SecureEnvelope]:
        actual_input_data: Union[SpaceDebrisData, TextInputData]
        original_sender_id: str = context.get("caller_id", "Orchestrator")

        if isinstance(input_data_or_envelope, SecureEnvelope):
            if not (self.secure_communicator and PQC_CRYPTO_AVAILABLE and config.ENABLE_PQC_SECURITY):
                return self._pack_error_response_cmd_list([{"command_name": "SECURITY_ERROR_SYS", "target_system": self.agent_id, "parameters": {"error": "Security misconfiguration"}, "priority": -1}], original_sender_id)
            original_sender_id = input_data_or_envelope.sender_id
            unpacked = self.secure_communicator.unpack_message(input_data_or_envelope, self.agent_id)
            if not unpacked: return self._pack_error_response_cmd_list([{"command_name": "UNPACK_ERROR_SYS", "target_system": self.agent_id, "parameters": {"error": f"Failed to unpack input from {original_sender_id}"}, "priority": -1}], original_sender_id)
            decrypted_bytes, _, metadata = unpacked
            payload_type_hint = metadata.get("payload_type")
            deserialized = _deserialize_payload_static(decrypted_bytes, payload_type_hint, self.agent_id)
            if not isinstance(deserialized, (SpaceDebrisData, TextInputData)):
                return self._pack_error_response_cmd_list([{"command_name": "PAYLOAD_TYPE_ERROR_SYS", "target_system": self.agent_id, "parameters": {"error": f"Expected SDD or TID, got {type(deserialized)}"}, "priority": -1}], original_sender_id)
            actual_input_data = deserialized
        elif isinstance(input_data_or_envelope, (SpaceDebrisData, TextInputData)):
            actual_input_data = input_data_or_envelope
        else:
            return self._pack_error_response_cmd_list([{"command_name": "INPUT_TYPE_ERROR_SYS", "target_system": self.agent_id, "parameters": {"error": f"Unsupported input type {type(input_data_or_envelope)}"}, "priority": -1}], original_sender_id)

        prompt_details = ""
        rag_query_context_main = "general satellite operation"
        if isinstance(actual_input_data, SpaceDebrisData):
            self._log_step(task_id, f"Processing SDD for SystemCmd", {"obj": actual_input_data.object_id})
            prompt_details = f"Object ID: {actual_input_data.object_id}\nSize: {actual_input_data.size_estimate_m or 'N/A'}m\nTrajectory: {str(actual_input_data.trajectory[:2])}...\nRisk: {str(actual_input_data.collision_risk_assessment)}"
            rag_query_context_main = f"commands for space object {actual_input_data.object_id}"
        elif isinstance(actual_input_data, TextInputData):
            self._log_step(task_id, f"Processing TID for SystemCmd", {"len": len(actual_input_data.text_content)})
            prompt_details = f"User Request for System Commands: \"{actual_input_data.text_content}\""
            rag_query_context_main = actual_input_data.text_content[:150]

        prompt = f"""Generate system commands as a JSON list of objects for:
{prompt_details}
Adhere to format: `[{{"command_name": "CMD", "target_system": "SYS_ID", "parameters": {{}}, "priority": 1}}]`.
If no commands are needed or possible, return `[]`. Output ONLY the JSON list. No extra text.
"""
        if context and context.get('satellite_id'): prompt += f"Target Satellite: {context['satellite_id']}\n"
        
        rag_query = f"System command specification for {rag_query_context_main}, target: {context.get('satellite_id', 'generic_satellite')}"
        rag_filter = {"doc_type": {"$in": ["satellite_manual", "command_spec", "safety_protocol", "operational_procedure", "infrastructure_control_doc"]}}
        response_text = await self._generate_response(prompt=prompt, use_rag=True, rag_query=rag_query, rag_filter=rag_filter, temperature=0.05) # Very low temp for commands
        
        commands_list = self._parse_command_list_from_llm(response_text, task_id, "SystemCommand")

        if self.secure_communicator and PQC_CRYPTO_AVAILABLE and config.ENABLE_PQC_SECURITY:
            return self.secure_communicator.pack_message(
                commands_list, self.agent_id, original_sender_id, "List[Dict[str,Any]]"
            ) or self._pack_error_response_cmd_list([{"command_name": "PACK_ERROR_SYS", "target_system":self.agent_id, "parameters":{"error": "Failed to pack SystemCmd response"}}], original_sender_id)
        return commands_list

    def _parse_command_list_from_llm(self, response_text: str, task_id: str, item_name: str) -> List[Dict[str, Any]]:
        parsed_list = []
        cleaned_text = ""
        try:
            logging.debug(f"Agent {self.agent_id}: Raw LLM for {item_name}: '{response_text}'")
            cleaned_text = response_text.strip()
            match_md = re.search(r"```json\s*(.*?)\s*```", cleaned_text, re.DOTALL | re.IGNORECASE)
            if match_md: cleaned_text = match_md.group(1).strip()
            else:
                match_md_generic = re.search(r"```\s*(.*?)\s*```", cleaned_text, re.DOTALL | re.IGNORECASE)
                if match_md_generic: cleaned_text = match_md_generic.group(1).strip()
            
            match_arr = re.search(r'^\s*(\[.*\])\s*$', cleaned_text, re.DOTALL)
            if match_arr: potential_json_str = match_arr.group(1)
            else: potential_json_str = cleaned_text

            parsed_output = json.loads(potential_json_str)
            if isinstance(parsed_output, list):
                validated_commands = []
                required_keys = ['command_name', 'target_system', 'parameters', 'priority']
                for cmd_idx, cmd in enumerate(parsed_output): # Added index for logging
                    if isinstance(cmd, dict) and all(k in cmd for k in required_keys) and isinstance(cmd.get('parameters'), dict):
                        validated_commands.append(cmd)
                    else:
                        logging.warning(f"Agent {self.agent_id}: Invalid {item_name} structure at index {cmd_idx} ignored: {str(cmd)[:100]}")
                parsed_list = validated_commands
                self._log_step(task_id, f"Successfully parsed and validated {len(parsed_list)} of {len(parsed_output)} {item_name} objects.")
            elif isinstance(parsed_output, dict) and parsed_output.get("command_name"): 
                # Check if the single dict is a valid command structure
                required_keys = ['command_name', 'target_system', 'parameters', 'priority']
                if all(k in parsed_output for k in required_keys) and isinstance(parsed_output.get('parameters'), dict):
                    parsed_list = [parsed_output]
                    self._log_step(task_id, f"Parsed single valid {item_name} object, wrapped in list.")
                else:
                    parsed_list = [{"command_name": "INVALID_SINGLE_CMD_STRUCT", "target_system": self.agent_id, "parameters": {"error": f"Single {item_name} object has invalid structure", "received_obj": str(parsed_output)[:200]}, "priority": -1}]
                    self._log_step(task_id, f"Parsed single {item_name} object was not a valid command.", {"raw_obj": str(parsed_output)[:200]})
            else: 
                parsed_list = [{"command_name": "PARSE_FORMAT_ERROR", "target_system": self.agent_id, "parameters": {"error": f"Parsed {item_name} was not a list of command dicts nor a single valid command dict", "raw_type": str(type(parsed_output))}, "priority": -1}]
                self._log_step(task_id, f"Parsed {item_name} JSON was not a list of valid commands.", {"type": str(type(parsed_output))})
        except json.JSONDecodeError as e:
            parsed_list = [{"command_name": "JSON_DECODE_ERROR", "target_system": self.agent_id, "parameters": {"error": f"Could not parse {item_name} from LLM - {e}. Cleaned text: '{cleaned_text[:100]}'. Raw snippet: '{response_text[:100]}'"}, "priority": -1}]
            self._log_step(task_id, f"Failed to parse {item_name} JSON.", {"error": str(e), "cleaned_text_attempt": cleaned_text[:200]})
        except Exception as e:
            parsed_list = [{"command_name": "PROCESSING_ERROR", "target_system": self.agent_id, "parameters": {"error": f"Error processing {item_name} LLM response: {e}"}, "priority": -1}]
            self._log_step(task_id, f"Generic error processing {item_name}.", {"error": str(e)})
        return parsed_list
        
    def _pack_error_response_cmd_list(self, error_payload_list: List[Dict[str,Any]], recipient_id: str) -> Union[List[Dict[str,Any]], SecureEnvelope]:
        if self.secure_communicator and PQC_CRYPTO_AVAILABLE and config.ENABLE_PQC_SECURITY:
            packed_error = self.secure_communicator.pack_message(
                payload=error_payload_list, sender_id=self.agent_id, recipient_id=recipient_id, payload_type_hint="List[Dict[str,Any]]"
            )
            return packed_error if packed_error else [{"command_name": "CRITICAL_PACK_ERROR_SYS", "target_system": self.agent_id, "parameters": {"error": f"Failed to pack error list for {self.agent_id} -> {recipient_id}"}, "priority": -1}]
        return error_payload_list


class AsteroidAgent(BaseAgent):
    def __init__(self, llm_client: LLMClient, rag_module: Optional[RAGModule], memory: Optional[TaskMemory],
                 secure_communicator: Optional[SecureCommunicator], agent_id: str):
        super().__init__(agent_id, llm_client, rag_module, memory, secure_communicator)

    def get_system_message(self) -> str:
        return """You are a space situational awareness analyst. Your task is to provide a detailed threat assessment for a given space object (asteroid or debris) or a textual description of such an event.
Focus on:
1.  **Object Characteristics**: Size (diameter in meters), estimated mass (kg, if calculable or known), composition type (e.g., C-type, S-type, M-type, or 'Unknown Debris').
2.  **Orbital Parameters**: Key elements (e.g., semi-major axis, eccentricity, inclination), trajectory stability, and predictability confidence (High/Medium/Low).
3.  **Collision Risk Assessment**: Closest approach distance (km) to Earth or specified target, probability of impact (if calculable), potential impact energy (TNT equivalent if applicable), and a qualitative risk level (e.g., Negligible, Low, Moderate, High, Critical).
4.  **Observation & Mitigation Recommendations**: Suggested next steps for observation (e.g., optical, radar), tracking priorities, and if applicable, preliminary thoughts on mitigation strategies if risk is high.
5.  **Overall Summary**: A concise summary of the threat assessment.
Output your assessment ONLY as a valid JSON object with keys: "object_summary", "orbital_analysis", "collision_risk_details", "recommendations", "overall_assessment_summary", "confidence_level" (for the entire assessment). Do NOT include any other text, commentary, or markdown formatting outside of this JSON object.
"""

    async def process(self, 
                      input_data_or_envelope: Union[SpaceDebrisData, TextInputData, SecureEnvelope], 
                      task_id: str, 
                      context: Optional[Dict[str, Any]] = None
                     ) -> Union[Dict[str, Any], SecureEnvelope]:
        actual_input_data: Union[SpaceDebrisData, TextInputData]
        original_sender_id: str = context.get("caller_id", "Orchestrator")

        if isinstance(input_data_or_envelope, SecureEnvelope):
            if not (self.secure_communicator and PQC_CRYPTO_AVAILABLE and config.ENABLE_PQC_SECURITY):
                return self._pack_error_response_dict({"error": "Security misconfiguration for AsteroidAgent."}, original_sender_id)
            original_sender_id = input_data_or_envelope.sender_id
            unpacked = self.secure_communicator.unpack_message(input_data_or_envelope, self.agent_id)
            if not unpacked: return self._pack_error_response_dict({"error": f"Failed to unpack input from {original_sender_id}."}, original_sender_id)
            decrypted_bytes, _, metadata = unpacked
            payload_type_hint = metadata.get("payload_type")
            deserialized = _deserialize_payload_static(decrypted_bytes, payload_type_hint, self.agent_id)
            if not isinstance(deserialized, (SpaceDebrisData, TextInputData)):
                return self._pack_error_response_dict({"error": f"Payload error. Expected SDD or TID, got {type(deserialized)}."}, original_sender_id)
            actual_input_data = deserialized
        elif isinstance(input_data_or_envelope, (SpaceDebrisData, TextInputData)):
            actual_input_data = input_data_or_envelope
        else:
            return self._pack_error_response_dict({"error": f"Unsupported input type {type(input_data_or_envelope)} for AsteroidAgent."}, original_sender_id)

        prompt_details = ""
        rag_query_context = "general space object threat assessment"
        if isinstance(actual_input_data, SpaceDebrisData):
            self._log_step(task_id, f"Processing SDD for Asteroid Analysis", {"obj": actual_input_data.object_id})
            prompt_details = f"Object ID: {actual_input_data.object_id}\nSize: {actual_input_data.size_estimate_m or 'N/A'}m\nTrajectory: {str(actual_input_data.trajectory[:2])}...\nRisk: {str(actual_input_data.collision_risk_assessment)}"
            rag_query_context = f"threat assessment for space object {actual_input_data.object_id}"
        elif isinstance(actual_input_data, TextInputData):
            self._log_step(task_id, f"Processing TID for Asteroid Analysis", {"len": len(actual_input_data.text_content)})
            prompt_details = f"User Request for Space Object Assessment: \"{actual_input_data.text_content}\""
            rag_query_context = f"assessment of space object described as: {actual_input_data.text_content[:150]}"

        prompt = f"""Provide a detailed threat assessment as a JSON object for the space object/situation:
{prompt_details}
Adhere strictly to the JSON structure and keys defined in the system message.
Output ONLY the JSON object. No extra text or markdown.
"""
        rag_query = f"Threat assessment, orbital analysis, and mitigation for {rag_query_context}"
        rag_filter = {"doc_type": {"$in": ["asteroid_data", "orbital_mechanics", "impact_study", "observation_log", "space_debris_report", "planetary_defense_conference"]}}
        response_text = await self._generate_response(prompt=prompt, use_rag=True, rag_query=rag_query, rag_filter=rag_filter, rag_top_k=5, temperature=0.3)
        
        assessment_report = self._parse_json_object_from_llm(response_text, task_id, "AsteroidThreatAssessment")
        
        if self.secure_communicator and PQC_CRYPTO_AVAILABLE and config.ENABLE_PQC_SECURITY:
            return self.secure_communicator.pack_message(
                assessment_report, self.agent_id, original_sender_id, "Dict[str,Any]"
            ) or self._pack_error_response_dict({"error": "Failed to pack AsteroidAgent response"}, original_sender_id)
        return assessment_report
    
    _parse_json_object_from_llm = DisasterAgent._parse_json_object_from_llm 
    _pack_error_response_dict = DisasterAgent._pack_error_response_dict


class GroundCmdAgent(BaseAgent):
    def __init__(self, llm_client: LLMClient, rag_module: Optional[RAGModule], memory: Optional[TaskMemory],
                 secure_communicator: Optional[SecureCommunicator], agent_id: str):
        super().__init__(agent_id, llm_client, rag_module, memory, secure_communicator)

    def get_system_message(self) -> str:
        return """You are an expert ground control systems operator. Your task is to generate precise, safe, and formatted Ground Commands.
Input will describe a situation requiring ground system actions (e.g., resource deployment for disasters, sensor adjustments, communication link configurations).
Commands could involve: resource deployment (specifying type, quantity, location), sensor activation/deactivation (specifying sensor ID, parameters), communication system adjustments (e.g., re-routing data, changing bandwidth), or logistics coordination.
You MUST adhere strictly to the required command format. Use provided context and retrieved technical documentation for ground systems and operational protocols. Prioritize safety, efficiency, and accuracy.
If unsure or data is insufficient, or if no ground commands are appropriate, return an empty JSON list `[]`.
Output ONLY a valid JSON list of command objects. Each command object must have 'command_name' (string), 'target_system' (string, e.g., 'GROUND_STATION_ALPHA', 'RESOURCE_DISPATCH_SYSTEM'), 'parameters' (dictionary specific to the command), 'priority' (integer: 1-High, 2-Medium, 3-Low), and 'dependencies' (optional list of other command_names this command depends on). Do NOT include any other text, commentary, or markdown formatting outside of this JSON list.
Example: `[{"command_name": "DEPLOY_MEDICAL_TEAM", "target_system": "EMS_DISPATCH_REGION_A", "parameters": {"team_size": 4, "destination_gps": [34.0522, -118.2437], "equipment_kits": ["trauma", "basic_life_support"]}, "priority": 1, "dependencies": ["AREA_SECURED_CONFIRMATION"]}]`
"""

    async def process(self, 
                      input_data_or_envelope: Union[DisasterWeatherData, TextInputData, Dict[str, Any], SecureEnvelope], 
                      task_id: str, 
                      context: Optional[Dict[str, Any]] = None
                     ) -> Union[List[Dict[str, Any]], SecureEnvelope]:
        actual_input_data: Union[DisasterWeatherData, TextInputData, Dict[str, Any]]
        original_sender_id: str = context.get("caller_id", "Orchestrator")

        if isinstance(input_data_or_envelope, SecureEnvelope):
            if not (self.secure_communicator and PQC_CRYPTO_AVAILABLE and config.ENABLE_PQC_SECURITY):
                return self._pack_error_response_cmd_list([{"command_name": "SECURITY_ERROR_GND", "target_system": self.agent_id, "parameters": {"error": "Security misconfiguration"}, "priority": -1}], original_sender_id)
            original_sender_id = input_data_or_envelope.sender_id
            unpacked = self.secure_communicator.unpack_message(input_data_or_envelope, self.agent_id)
            if not unpacked: return self._pack_error_response_cmd_list([{"command_name": "UNPACK_ERROR_GND", "target_system": self.agent_id, "parameters": {"error": f"Failed to unpack input from {original_sender_id}"}, "priority": -1}], original_sender_id)
            decrypted_bytes, _, metadata = unpacked
            payload_type_hint = metadata.get("payload_type")
            deserialized = _deserialize_payload_static(decrypted_bytes, payload_type_hint, self.agent_id)
            if not isinstance(deserialized, (DisasterWeatherData, TextInputData, dict)):
                return self._pack_error_response_cmd_list([{"command_name": "PAYLOAD_TYPE_ERROR_GND", "target_system": self.agent_id, "parameters": {"error": f"Expected DWD, TID or Dict, got {type(deserialized)}"}, "priority": -1}], original_sender_id)
            actual_input_data = deserialized
        elif isinstance(input_data_or_envelope, (DisasterWeatherData, TextInputData, dict)):
            actual_input_data = input_data_or_envelope
        else:
            return self._pack_error_response_cmd_list([{"command_name": "INPUT_TYPE_ERROR_GND", "target_system": self.agent_id, "parameters": {"error": f"Unsupported input type {type(input_data_or_envelope)}"}, "priority": -1}], original_sender_id)

        input_summary = ""
        prompt_details = ""
        if isinstance(actual_input_data, DisasterWeatherData):
            input_summary = f"Disaster: {actual_input_data.event_type} at {actual_input_data.location}"
            prompt_details = f"Event: {actual_input_data.event_type}, Loc: {actual_input_data.location}, Sev: {actual_input_data.severity or 'N/A'}"
        elif isinstance(actual_input_data, TextInputData):
            input_summary = f"Text Req: {actual_input_data.text_content[:70]}"
            prompt_details = f"User Request for Ground Commands: \"{actual_input_data.text_content}\""
        elif isinstance(actual_input_data, dict):
            input_summary = f"Generic Data: {str(actual_input_data)[:70]}"
            prompt_details = f"Input Data Details: {json.dumps(actual_input_data, indent=2)}"
        self._log_step(task_id, f"Processing for GroundCmd", {"summary": input_summary})

        prompt = f"""Generate ground system commands as a JSON list of objects for:
Situation: {input_summary}
Details: {prompt_details}
Adhere to format: `[{{"command_name": "CMD", "target_system": "SYS_ID", "parameters": {{}}, "priority": 1, "dependencies": []}}]`.
If no commands needed, return `[]`. Output ONLY the JSON list. No extra text.
"""
        if context and context.get('current_resources'): prompt += f"\nResource Status: {str(context['current_resources'])[:200]}\n"
        
        rag_query = f"Ground system commands and operational protocols for situation: {input_summary}"
        rag_filter = {"doc_type": {"$in": ["ground_system_manual", "emergency_protocol", "resource_allocation_guide", "logistics_plan"]}}
        response_text = await self._generate_response(prompt=prompt, use_rag=True, rag_query=rag_query, rag_filter=rag_filter, temperature=0.1) # Low temp for commands
        
        commands_list = self._parse_command_list_from_llm(response_text, task_id, "GroundCommand")

        if self.secure_communicator and PQC_CRYPTO_AVAILABLE and config.ENABLE_PQC_SECURITY:
            return self.secure_communicator.pack_message(
                commands_list, self.agent_id, original_sender_id, "List[Dict[str,Any]]"
            ) or self._pack_error_response_cmd_list([{"command_name": "PACK_ERROR_GND", "target_system":self.agent_id, "parameters":{"error": "Failed to pack GroundCmd response"}}], original_sender_id)
        return commands_list

    _parse_command_list_from_llm = SystemCmdAgent._parse_command_list_from_llm
    _pack_error_response_cmd_list = SystemCmdAgent._pack_error_response_cmd_list

