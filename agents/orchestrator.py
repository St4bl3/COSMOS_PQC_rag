# --- File: agents/orchestrator.py ---
import uuid
from typing import Dict, Any, Optional, List, Type, Union
from datetime import datetime

from ingestion.models import (
    OrchestratorInput, DisasterWeatherData, SpaceDebrisData, 
    TextInputData, FileInputData
)
# Assuming BaseAgent is conceptually updated to handle agent_id and secure_communicator
from agents.base_agent import BaseAgent, get_type_hints 
from agents.specialized_agents import (
    SOPAgent, SystemCmdAgent, DisasterAgent, 
    AsteroidAgent, GroundCmdAgent, CodeGenAgent
)
from core.llm import LLMClient
from core.rag import RAGModule
from core.memory import TaskMemory

# Security Imports
from security.secure_communication import SecureCommunicator 
from security.secure_envelope import SecureEnvelope 
from pqc_crypto_package import PQC_CRYPTO_AVAILABLE
import config # To check config.ENABLE_PQC_SECURITY

import logging 
import json 
import re 

logger = logging.getLogger(__name__)

class OrchestrationAgent(BaseAgent): # BaseAgent conceptually takes agent_id and secure_communicator
    def __init__(
        self,
        llm_client: LLMClient,
        rag_module: RAGModule,
        memory: TaskMemory,
        agent_classes: Dict[str, Type[BaseAgent]],
        secure_communicator: Optional[SecureCommunicator], 
        agent_id: str # Orchestrator's own ID
    ):
        # BaseAgent __init__ now conceptually takes agent_id and secure_communicator
        super().__init__(agent_id, llm_client, rag_module, memory, secure_communicator)
        self.agent_classes = agent_classes
        self.agents: Dict[str, BaseAgent] = self._create_agents(secure_communicator)
        
        secure_status = "DISABLED"
        if self.secure_communicator and PQC_CRYPTO_AVAILABLE and config.ENABLE_PQC_SECURITY:
            secure_status = "ENABLED"
        logger.info(f"Orchestrator (ID: {self.agent_id}) initialized. Agents: {list(self.agents.keys())}. Secure comms: {secure_status}")

    def _create_agents(self, secure_communicator: Optional[SecureCommunicator]) -> Dict[str, BaseAgent]:
        created_agents = {}
        logger.info(f"Orchestrator (ID: {self.agent_id}): Creating specialized agents...")
        for name, agent_cls in self.agent_classes.items():
            try:
                if not self.llm: raise ValueError("LLMClient dependency missing for agent creation")
                created_agents[name] = agent_cls(
                    llm_client=self.llm, 
                    rag_module=self.rag, 
                    memory=self.memory,
                    secure_communicator=secure_communicator, # Pass down
                    agent_id=name # Agent's own ID for security operations
                )
                logging.debug(f"Agent '{name}' created successfully by Orchestrator.")
            except Exception as e:
                logging.exception(f"Error creating agent '{name}': {e}")
        return created_agents

    def get_system_message(self) -> str:
        return "You are a central orchestrator for a multi-agent system. Your role is to understand input, determine necessary specialized agents, and manage workflow. Focus on efficient routing and clear task definition."

    async def process(self, input_payload: OrchestratorInput) -> Dict[str, Any]:
        task_id = input_payload.task_id or str(uuid.uuid4())
        input_payload.task_id = task_id 
        logging.info(f"\n--- Orchestrator (ID: {self.agent_id}) Starting Task: {task_id} ---")
        
        input_data_for_state = {
            "source_id": input_payload.source_id, "timestamp": input_payload.timestamp, 
            "metadata": input_payload.metadata, "data_type": input_payload.data_type,
            "data": input_payload.data.model_dump() if hasattr(input_payload.data, 'model_dump') else input_payload.data, # Pydantic v2
            "task_id": input_payload.task_id
        }
        initial_state = {
            "input_data": input_data_for_state, "status": "processing", "agent_outputs": {},
            "human_feedback": {}, "steps_log": [], "timestamp": input_payload.timestamp
        }
        self.memory.log_step(task_id, f"[{self.agent_name}] Orchestrator received input payload", 
                             {"data_type": input_payload.data_type, "source": input_payload.source_id})
        self.memory.save_task_state(task_id, initial_state)

        context = await self._enrich_context(input_payload, task_id)
        required_agent_names = self._determine_required_agents(input_payload, context, task_id)
        self._log_step(task_id, f"[{self.agent_name}] Determined required agents: {required_agent_names}")

        agent_outputs_raw = {}
        execution_successful = True

        if not required_agent_names:
            logging.warning(f"Task {task_id}: No agents determined for execution by {self.agent_id}.")
            self._log_step(task_id, f"[{self.agent_name}] No agents required for this task.")
        else:
            current_context = context.copy()
            for agent_name in required_agent_names: 
                agent = self.agents.get(agent_name)
                if agent:
                    try:
                        logging.info(f"--- Task {task_id}: Orchestrator (ID: {self.agent_id}) invoking Agent: {agent_name} (ID: {agent.agent_id}) ---")
                        
                        agent_input_data_unsecured = self._get_agent_input(input_payload, agent_name)
                        if agent_input_data_unsecured is None:
                            logging.warning(f"Task {task_id}: Could not determine/prepare input for {agent_name}. Skipping.")
                            self._log_step(task_id, f"[{self.agent_name}] Skipping agent '{agent_name}' due to input preparation failure.")
                            continue

                        agent_call_input: Union[Any, SecureEnvelope] = agent_input_data_unsecured
                        payload_type_hint_for_agent = agent_input_data_unsecured.__class__.__name__ if hasattr(agent_input_data_unsecured, '__class__') else type(agent_input_data_unsecured).__name__


                        if self.secure_communicator and PQC_CRYPTO_AVAILABLE and config.ENABLE_PQC_SECURITY:
                            logging.info(f"Orchestrator ({self.agent_id}) securely packing input for agent '{agent.agent_id}'. Type hint: {payload_type_hint_for_agent}")
                            packed_input_envelope = self.secure_communicator.pack_message(
                                payload=agent_input_data_unsecured, # pack_message handles Pydantic serialization
                                sender_id=self.agent_id, 
                                recipient_id=agent.agent_id,
                                payload_type_hint=payload_type_hint_for_agent
                            )
                            if not packed_input_envelope:
                                logging.error(f"Task {task_id}: Failed to pack secure envelope for agent {agent.agent_id}. Skipping.")
                                agent_outputs_raw[agent_name] = {"error": f"Failed to secure input for {agent.agent_id}"}
                                execution_successful = False
                                continue
                            agent_call_input = packed_input_envelope
                        elif config.ENABLE_PQC_SECURITY: # PQC enabled in config, but communicator missing or PQC libs failed
                             logging.warning(f"Task {task_id}: PQC Security is enabled in config, but SecureCommunicator or PQC libs are not fully available. Sending plaintext to {agent.agent_id}. THIS IS INSECURE.")
                        
                        output_from_agent = await agent.process(agent_call_input, task_id, current_context)
                        actual_output = output_from_agent

                        if self.secure_communicator and PQC_CRYPTO_AVAILABLE and config.ENABLE_PQC_SECURITY and isinstance(output_from_agent, SecureEnvelope):
                            logging.info(f"Orchestrator ({self.agent_id}) received secure envelope from {agent.agent_id}. Unpacking...")
                            unpacked_result = self.secure_communicator.unpack_message(
                                envelope=output_from_agent,
                                expected_recipient_id=self.agent_id
                            )
                            if unpacked_result:
                                decrypted_payload_bytes, received_sender_id, received_metadata = unpacked_result
                                if received_sender_id != agent.agent_id:
                                     logging.error(f"Sender ID mismatch in envelope from {agent.agent_id}! Expected {agent.agent_id}, got {received_sender_id}.")
                                     actual_output = {"error": f"Sender ID mismatch from {agent.agent_id}"}
                                     execution_successful = False
                                else:
                                    # Deserialize based on what this agent is expected to return (complex part)
                                    # For now, try generic JSON deserialization or raw string
                                    try:
                                        actual_output = json.loads(decrypted_payload_bytes.decode('utf-8'))
                                    except (json.JSONDecodeError, UnicodeDecodeError):
                                        actual_output = decrypted_payload_bytes.decode('utf-8', errors='replace') 
                                    logging.info(f"Task {task_id}: Successfully unpacked message from {agent.agent_id}.")
                            else:
                                logging.error(f"Task {task_id}: Failed to unpack secure envelope from agent {agent.agent_id}. Marking agent output as error.")
                                actual_output = {"error": f"Failed to decrypt/verify output from {agent.agent_id}"}
                                execution_successful = False
                        elif config.ENABLE_PQC_SECURITY and not isinstance(output_from_agent, SecureEnvelope) and PQC_CRYPTO_AVAILABLE:
                            logging.warning(f"Task {task_id}: Received plaintext output from {agent.agent_id} while PQC security is ENABLED. This is unexpected.")
                        
                        agent_outputs_raw[agent_name] = actual_output
                        if isinstance(actual_output, dict) and 'error' in actual_output:
                            logging.error(f"Agent '{agent_name}' (ID: {agent.agent_id}) effectively reported an error: {actual_output.get('error')}")
                            execution_successful = False
                        elif isinstance(actual_output, list) and actual_output and isinstance(actual_output[0], dict) and "ERROR" in actual_output[0].get("command_name", ""):
                            logging.error(f"Agent '{agent_name}' (ID: {agent.agent_id}) reported command error: {actual_output[0]}")
                            execution_successful = False

                        current_context[f"{agent_name}_output_summary"] = str(actual_output)[:200]
                        logging.info(f"--- Task {task_id}: Orchestrator (ID: {self.agent_id}) received output from Agent {agent_name} ---")
                    except Exception as e:
                        logging.exception(f"Task {task_id}: Unhandled error by Orchestrator (ID: {self.agent_id}) during processing loop for agent '{agent_name}': {e}")
                        self._log_step(task_id, f"[{self.agent_name}] Unhandled error with agent '{agent_name}'", {"error": str(e)})
                        agent_outputs_raw[agent_name] = {"error": f"Orchestrator failed to process agent {agent_name}: {e}"}
                        execution_successful = False
                else: 
                    logging.error(f"Task {task_id}: Agent '{agent_name}' not found by Orchestrator (ID: {self.agent_id}).")
                    self._log_step(task_id, f"[{self.agent_name}] Agent '{agent_name}' not found.")
                    execution_successful = False
            
            formatted_agent_outputs = self._format_output(agent_outputs_raw)
            final_state = self.memory.get_task_state(task_id) or initial_state
            final_state['agent_outputs'] = formatted_agent_outputs 
            
            if not execution_successful: final_state['status'] = 'failed'
            elif not required_agent_names: final_state['status'] = 'completed_no_action'
            else: final_state['status'] = 'pending_review'
            
            self.memory.save_task_state(task_id, final_state)
            log_msg = f"[{self.agent_name}] Task processing finished with status: {final_state['status']}."
            self._log_step(task_id, log_msg)
            logging.info(f"--- Task {task_id} Finished Processing by {self.agent_id} (Status: {final_state['status']}) ---")
            return {
                "task_id": task_id,
                "status": final_state['status'],
                "results": formatted_agent_outputs
            }

    async def _enrich_context(self, input_payload: OrchestratorInput, task_id: str) -> Dict[str, Any]:
        self._log_step(task_id, f"[{self.agent_name}] Starting context enrichment")
        input_summary_dict = input_payload.model_dump(exclude={'data'}) # Pydantic v2
        if 'timestamp' in input_summary_dict and isinstance(input_summary_dict['timestamp'], datetime):
             input_summary_dict['timestamp'] = input_summary_dict['timestamp'].isoformat()
        context = {"initial_input_summary": str(input_summary_dict)[:300]}

        if input_payload.data_type == "text" and isinstance(input_payload.data, TextInputData):
            prompt = f"""Analyze the following text input. Identify the primary topic (e.g., disaster alert, debris report, SOP request, command request, code request, general query), key entities (locations, object IDs, system names), and the desired output type (e.g., SOP, System Command, Ground Command, Code Snippet, Analysis).

Text: "{input_payload.data.text_content}"

Output your analysis ONLY as a JSON object with keys 'topic', 'entities' (as a dictionary), 'desired_output_type'. 'desired_output_type' should be a single comma-separated string if multiple types are detected (e.g., "Analysis, SOP, System Command"), or a single string if only one type (e.g., "SOP"). If unsure, use "Unknown" for any field.
Example 1 (single): {{"topic": "Debris Report", "entities": {{"object_id": "SATDEB-1234"}}, "desired_output_type": "System Command"}}
Example 2 (multiple): {{"topic": "Complex Incident", "entities": {{"location": "Sector A"}}, "desired_output_type": "Analysis, SOP, Ground Command"}}
"""
            analysis_raw_str = "" 
            extracted_json_str = None 
            try:
                logging.debug(f"Task {task_id}: Performing LLM context enrichment for text input by {self.agent_name}.")
                analysis_raw_str = await self.llm.agenerate(prompt, system_message=self.get_system_message())
                logging.debug(f"LLM analysis response for context enrichment: {analysis_raw_str}")

                match_curly = re.search(r'\{.*\}', analysis_raw_str, re.DOTALL)
                if match_curly: extracted_json_str = match_curly.group(0)
                
                if extracted_json_str:
                    analysis = json.loads(extracted_json_str)
                    if 'desired_output_type' in analysis:
                        dot_value = analysis['desired_output_type']
                        if isinstance(dot_value, dict): analysis['desired_output_type'] = ", ".join(str(v) for v in dot_value.values())
                        elif isinstance(dot_value, list): analysis['desired_output_type'] = ", ".join(str(v) for v in dot_value)
                        elif not isinstance(dot_value, str): analysis['desired_output_type'] = str(dot_value)
                    context.update(analysis)
                    self._log_step(task_id, f"[{self.agent_name}] Enriched context using LLM analysis", analysis)
                else: 
                    self._log_step(task_id, f"[{self.agent_name}] Failed LLM context enrichment (No JSON found)", {"raw_response": analysis_raw_str})
            except json.JSONDecodeError as json_e: 
                self._log_step(task_id, f"[{self.agent_name}] Failed LLM context enrichment (JSON Decode)", {"error": str(json_e), "raw_response": extracted_json_str or analysis_raw_str})
            except Exception as e: 
                 self._log_step(task_id, f"[{self.agent_name}] Failed LLM context enrichment (Exception)", {"error": str(e)})
        
        search_query = f"Task related to {input_payload.data_type}"
        search_detail = ""
        if isinstance(input_payload.data, DisasterWeatherData):
            search_detail = f"{input_payload.data.event_type} at {input_payload.data.location}"
        elif isinstance(input_payload.data, SpaceDebrisData):
            search_detail = f"space object {input_payload.data.object_id}"
        elif isinstance(input_payload.data, TextInputData):
            search_detail = input_payload.data.text_content[:100] 
        search_query += f": {search_detail}" if search_detail else f" source {input_payload.source_id}"

        logging.debug(f"Task {task_id}: Searching for related tasks by {self.agent_name} with query: '{search_query}'")
        related_tasks = self.memory.search_related_tasks(search_query, top_k=2) # search_related_tasks handles its own logging
        if related_tasks:
            serializable_related_tasks = []
            for task_item in related_tasks: 
                task_copy = task_item.copy()
                if 'timestamp' in task_copy and isinstance(task_copy['timestamp'], datetime):
                    task_copy['timestamp'] = task_copy['timestamp'].isoformat()
                if 'input_data' in task_copy: 
                    task_copy['input_data_summary'] = str(task_copy.pop('input_data'))[:150]
                if 'agent_outputs' in task_copy:
                    task_copy['agent_outputs_summary'] = str(task_copy.pop('agent_outputs'))[:150]
                serializable_related_tasks.append(task_copy)
            context['related_past_tasks'] = serializable_related_tasks
            self._log_step(task_id, f"[{self.agent_name}] Retrieved {len(related_tasks)} related task summaries.")
        else:
            logging.debug(f"Task {task_id}: No related tasks found in memory by {self.agent_name}.")
        return context

    def _determine_required_agents(self, input_payload: OrchestratorInput, context: Dict[str, Any], task_id: str) -> List[str]:
        # This is the comprehensive version you provided earlier.
        logging.info(f"Task {task_id} ({self.agent_id}): Determining required agents...")
        data_type = input_payload.data_type
        desired_output_type_str = context.get('desired_output_type', '')
        if not isinstance(desired_output_type_str, str): desired_output_type_str = str(desired_output_type_str)
        desired_output_type_str = desired_output_type_str.lower()
        topic = context.get('topic', '')
        if not isinstance(topic, str): topic = str(topic)
        topic = topic.lower()
        logging.debug(f"Task {task_id} ({self.agent_id}): Input data_type: {data_type}, Desired output: '{desired_output_type_str}', Topic: '{topic}'")
        agents_to_run = []
        def check_desired(keywords: List[str]) -> bool:
            return any(keyword.lower() in desired_output_type_str for keyword in keywords)

        if data_type == 'disaster':
            if check_desired(['sop', 'procedure']) or not desired_output_type_str:
                if "SOPAgent" in self.agents: agents_to_run.append("SOPAgent")
            if check_desired(['analysis', 'report', 'assessment', 'impact analysis']):
                if "DisasterAgent" in self.agents: agents_to_run.append("DisasterAgent")
            if check_desired(['ground command', 'deploy', 'resource deployment']):
                if "GroundCmdAgent" in self.agents: agents_to_run.append("GroundCmdAgent")
        elif data_type == 'debris':
            if check_desired(['system command', 'maneuver', 'satellite command']) or not desired_output_type_str:
                if "SystemCmdAgent" in self.agents: agents_to_run.append("SystemCmdAgent")
            if check_desired(['assessment', 'analysis', 'threat assessment']) or not desired_output_type_str:
                if "AsteroidAgent" in self.agents: agents_to_run.append("AsteroidAgent")
            if check_desired(['sop', 'procedure']):
                if "SOPAgent" in self.agents: agents_to_run.append("SOPAgent")
        elif data_type == 'text' or data_type == 'file':
            logging.debug(f"Task {task_id} ({self.agent_id}): Routing text/file: Topic='{topic}', Desired='{desired_output_type_str}'")
            if check_desired(['sop', 'procedure', 'standard operating procedure']):
                if "SOPAgent" in self.agents: agents_to_run.append("SOPAgent")
            if check_desired(['system command', 'satellite command', 'maneuver']):
                if "SystemCmdAgent" in self.agents: agents_to_run.append("SystemCmdAgent")
            if check_desired(['ground command', 'deploy resource', 'field deployment']):
                if "GroundCmdAgent" in self.agents: agents_to_run.append("GroundCmdAgent")
            if check_desired(['code', 'script', 'program', 'algorithm', 'python script', 'javascript function']):
                if "CodeGenAgent" in self.agents: agents_to_run.append("CodeGenAgent")
            if check_desired(['disaster analysis', 'disaster report', 'impact analysis']):
                if "DisasterAgent" in self.agents: agents_to_run.append("DisasterAgent")
            if check_desired(['asteroid assessment', 'debris analysis', 'space object analysis', 'threat assessment']):
                if "AsteroidAgent" in self.agents: agents_to_run.append("AsteroidAgent")
            
            if topic: 
                if any(kw in topic for kw in ['disaster', 'emergency', 'cyberattack', 'power grid', 'incident response']):
                    if "DisasterAgent" in self.agents and "DisasterAgent" not in agents_to_run : agents_to_run.append("DisasterAgent")
                    if "SOPAgent" in self.agents and "SOPAgent" not in agents_to_run: agents_to_run.append("SOPAgent")
                    if "GroundCmdAgent" in self.agents and "GroundCmdAgent" not in agents_to_run: agents_to_run.append("GroundCmdAgent")
                elif any(kw in topic for kw in ['debris', 'asteroid', 'satellite', 'collision', 'space object', 'orbital']):
                     if "AsteroidAgent" in self.agents and "AsteroidAgent" not in agents_to_run: agents_to_run.append("AsteroidAgent")
                     if "SystemCmdAgent" in self.agents and "SystemCmdAgent" not in agents_to_run: agents_to_run.append("SystemCmdAgent")
                     if "SOPAgent" in self.agents and "SOPAgent" not in agents_to_run : agents_to_run.append("SOPAgent")
                elif any(kw in topic for kw in ['code', 'script', 'program', 'develop', 'python', 'javascript', 'software', 'algorithm']):
                    if "CodeGenAgent" in self.agents and "CodeGenAgent" not in agents_to_run: agents_to_run.append("CodeGenAgent")
                elif any(kw in topic for kw in ['sop', 'procedure', 'checklist', 'guideline']):
                    if "SOPAgent" in self.agents and "SOPAgent" not in agents_to_run: agents_to_run.append("SOPAgent")
            if not agents_to_run: logging.warning(f"Task {task_id} ({self.agent_id}): No agent for text/file, Topic='{topic}', Desired='{desired_output_type_str}'.")
        else: logging.warning(f"Task {task_id} ({self.agent_id}): Unhandled data_type '{data_type}'.")
        
        final_agents_ordered = [] 
        seen_agents = set()
        for agent_name in agents_to_run:
            if agent_name in self.agents and agent_name not in seen_agents:
                final_agents_ordered.append(agent_name)
                seen_agents.add(agent_name)
            elif agent_name not in self.agents:
                 logging.warning(f"Task {task_id} ({self.agent_id}): Determined agent '{agent_name}' not available.")
        logging.info(f"Task {task_id} ({self.agent_id}): Final agent sequence: {final_agents_ordered}")
        return final_agents_ordered

    def _get_agent_input(self, input_payload: OrchestratorInput, agent_name: str) -> Optional[Any]:
        agent_class = self.agent_classes.get(agent_name)
        if not agent_class: 
            logging.error(f"Orchestrator ({self.agent_id}): Cannot get input type for unknown agent: {agent_name}")
            return None
        try:
            process_method = getattr(agent_class, 'process')
            hints = get_type_hints(process_method) 
            expected_input_type = hints.get('input_data_or_envelope') # Match param name in specialized agents
            if not expected_input_type: # Fallback if param name differs or no hint
                 expected_input_type = hints.get('input_data', Any)
            logging.debug(f"Orchestrator ({self.agent_id}): Agent '{agent_name}' expects input type (from hint): {expected_input_type}")
        except Exception as e:
            logging.warning(f"Orchestrator ({self.agent_id}): Could not determine expected input type hint for {agent_name}.process: {e}")
            expected_input_type = Any
        
        actual_data_payload = input_payload.data # This is the Pydantic model (e.g., DisasterWeatherData)

        # If the agent's process method is typed with Union that includes the actual_data_payload type, it's a match.
        if hasattr(expected_input_type, '__origin__') and expected_input_type.__origin__ is Union:
            if any(isinstance(actual_data_payload, arg) for arg in expected_input_type.__args__ if isinstance(arg, type)):
                logging.debug(f"Orchestrator ({self.agent_id}): Providing agent '{agent_name}' with input of type {type(actual_data_payload)} (matches Union).")
                return actual_data_payload
        # Direct type match
        elif expected_input_type and isinstance(actual_data_payload, expected_input_type):
            logging.debug(f"Orchestrator ({self.agent_id}): Providing agent '{agent_name}' with input of type {type(actual_data_payload)}.")
            return actual_data_payload
        # Agent expects Dict, and we have a Pydantic model
        elif expected_input_type == Dict[str, Any] and hasattr(actual_data_payload, 'model_dump'): # Pydantic v2
            logging.debug(f"Orchestrator ({self.agent_id}): Agent '{agent_name}' expects Dict, converting Pydantic model {type(actual_data_payload)} to dict.")
            return actual_data_payload.model_dump()
        elif expected_input_type == Dict[str, Any] and hasattr(actual_data_payload, 'dict'): # Pydantic v1
            logging.debug(f"Orchestrator ({self.agent_id}): Agent '{agent_name}' expects Dict, converting Pydantic model {type(actual_data_payload)} to dict.")
            return actual_data_payload.dict()
        # Agent expects Any, or no specific type hint was resolved
        elif expected_input_type is Any or not expected_input_type:
            logging.debug(f"Orchestrator ({self.agent_id}): Agent '{agent_name}' expects Any or type unknown. Providing raw data payload of type {type(actual_data_payload)}.")
            return actual_data_payload
        
        logging.error(f"Orchestrator ({self.agent_id}): Type mismatch for agent '{agent_name}'. Expected {expected_input_type} but got {type(actual_data_payload)}. Cannot provide suitable input.")
        return None

    def _format_output(self, agent_outputs: Dict[str, Any]) -> Dict[str, Any]:
        logging.debug(f"Orchestrator ({self.agent_id}): Formatting final output from agent outputs: {list(agent_outputs.keys())}")
        formatted = {}
        for agent_name, output in agent_outputs.items():
            is_error = False
            error_message = "Unknown error from agent"
            raw_output_on_error = str(output)[:500] # Default raw output on error

            if isinstance(output, dict) and output.get("status") == "error": # Check our own formatted error
                is_error = True
                error_message = output.get("message", "Agent reported an error.")
                raw_output_on_error = output.get("raw_output", raw_output_on_error)
            elif isinstance(output, list) and output and isinstance(output[0], dict) and "ERROR" in output[0].get("command_name", ""): # Specific for command agents
                is_error = True
                error_message = output[0].get("parameters", {}).get("error", "Command agent reported an error.")
            elif isinstance(output, dict) and 'error' in output: # Generic error dict from agent
                is_error = True
                error_message = str(output['error'])
            
            if is_error: 
                formatted[agent_name] = {"status": "error", "message": error_message, "raw_output": raw_output_on_error}
            else: 
                # This assumes 'output' is the direct result if not an error structure
                formatted[agent_name] = {"status": "success", "output": output}
        return formatted

