# --- File: api/endpoints.py ---
from fastapi import FastAPI, HTTPException, Body, Depends, BackgroundTasks
from typing import Dict, Any, Optional 
from api.models import IngestRequest, IngestResponse, TaskStatusResponse, FeedbackRequest, FeedbackResponse
from ingestion.parser import DataParser
from ingestion.models import OrchestratorInput
# Import classes for type hinting and direct instantiation if needed outside lifespan
from core.llm import LLMClient
from core.embedding import EmbeddingClient
from core.vector_db import VectorDBClient
from core.rag import RAGModule
from core.memory import TaskMemory
from agents.orchestrator import OrchestrationAgent
from agents.specialized_agents import (
    SOPAgent, SystemCmdAgent, DisasterAgent, 
    AsteroidAgent, GroundCmdAgent, CodeGenAgent
)
import config 
import logging 
import uuid 

# --- Dependency Injection Setup ---
# Global instances, primarily set by lifespan in main.py
_embedding_client_instance: Optional[EmbeddingClient] = None
_vector_db_client_instance: Optional[VectorDBClient] = None
_llm_client_instance: Optional[LLMClient] = None
_rag_module_instance: Optional[RAGModule] = None
_task_memory_instance_ref: Optional[TaskMemory] = None # Note: name matches main.py
_orchestrator_instance: Optional[OrchestrationAgent] = None


def get_embedding_client() -> EmbeddingClient:
    global _embedding_client_instance
    if _embedding_client_instance is None:
        logging.warning("EmbeddingClient instance was None, attempting to initialize now (should have been done by lifespan).")
        _embedding_client_instance = EmbeddingClient()
    if not _embedding_client_instance.client: # Check if internal client is okay
        logging.error("EmbeddingClient's internal client is not initialized. RAG and other features might fail.")
        # Optionally raise HTTPException if this is critical for the endpoint using it
    return _embedding_client_instance

def get_llm_client() -> LLMClient:
    global _llm_client_instance
    if _llm_client_instance is None:
        logging.warning("LLMClient instance was None, attempting to initialize now (should have been done by lifespan).")
        _llm_client_instance = LLMClient()
    return _llm_client_instance

def get_vector_db_client(
    embedding_client: EmbeddingClient = Depends(get_embedding_client) # For request-time init
) -> VectorDBClient:
    global _vector_db_client_instance
    if _vector_db_client_instance is None:
        logging.warning("VectorDBClient instance was None, attempting to initialize now (should have been done by lifespan).")
        _vector_db_client_instance = VectorDBClient(embedding_client=embedding_client) # FastAPI resolves embedding_client
    if not _vector_db_client_instance.index:
        logging.error("VectorDBClient's index/collection is not initialized. RAG might fail.")
    return _vector_db_client_instance

def get_task_memory() -> TaskMemory:
    global _task_memory_instance_ref
    if _task_memory_instance_ref is None:
        logging.error("TaskMemory instance was None! This is critical and should have been set by lifespan.")
        # Fallback initialization (less ideal, as VDB client might not be the lifespan one)
        # For a robust fallback, you'd re-resolve dependencies here.
        # temp_ec = get_embedding_client()
        # temp_vdb = get_vector_db_client(embedding_client=temp_ec)
        # _task_memory_instance_ref = TaskMemory(vector_db_client=temp_vdb)
        # logging.warning("Attempted fallback TaskMemory initialization.")
        raise HTTPException(status_code=500, detail="Internal Server Error: Memory service not available (lifespan init failed).")
    return _task_memory_instance_ref

def get_rag_module(
    vector_db_client: VectorDBClient = Depends(get_vector_db_client),
    llm_client: LLMClient = Depends(get_llm_client),
    embedding_client: EmbeddingClient = Depends(get_embedding_client) # Added for robust request-time init
) -> RAGModule:
    global _rag_module_instance
    if _rag_module_instance is None:
        logging.warning("RAGModule instance was None, attempting to initialize now (should have been done by lifespan).")
        _rag_module_instance = RAGModule(
            vector_db_client=vector_db_client, # FastAPI resolves this to an instance
            llm_client=llm_client,             # FastAPI resolves this
            embedding_client=embedding_client  # FastAPI resolves this
        )
    return _rag_module_instance

def get_orchestrator(
    llm_client: LLMClient = Depends(get_llm_client),
    rag_module: RAGModule = Depends(get_rag_module),
    memory: TaskMemory = Depends(get_task_memory)
    # No need for agent_classes here as it's for instantiation, not just retrieval
) -> OrchestrationAgent:
    global _orchestrator_instance
    if _orchestrator_instance is None:
        logging.warning("Orchestrator instance was None, attempting to initialize now (should have been done by lifespan).")
        # This is a fallback, agent_classes should ideally be consistent
        agent_classes_fallback = { 
            "SOPAgent": SOPAgent, "SystemCmdAgent": SystemCmdAgent,
            "DisasterAgent": DisasterAgent, "AsteroidAgent": AsteroidAgent,
            "GroundCmdAgent": GroundCmdAgent, "CodeGenAgent": CodeGenAgent,
        }
        _orchestrator_instance = OrchestrationAgent(
            llm_client=llm_client,
            rag_module=rag_module,
            memory=memory,
            agent_classes=agent_classes_fallback
        )
    return _orchestrator_instance

# --- API Endpoints ---
# (ingest_data, get_task_status, submit_feedback remain the same)
# ... (rest of your endpoints.py file)
async def ingest_data(
    background_tasks: BackgroundTasks,
    ingest_request: IngestRequest = Body(...),
    parser: DataParser = Depends(DataParser), 
    orchestrator: OrchestrationAgent = Depends(get_orchestrator),
    memory: TaskMemory = Depends(get_task_memory) 
):
    logging.info(f"Received ingest request: format={ingest_request.data_format}, task_id={ingest_request.task_id}, source_id={ingest_request.source_id}")
    try:
        orchestrator_input: Optional[OrchestratorInput] = parser.parse(
            input_data=ingest_request.data,
            data_format=ingest_request.data_format
        )
    except Exception as e:
        logging.exception("Error during initial data parsing in ingest endpoint.")
        raise HTTPException(status_code=400, detail=f"Failed to parse input data: {e}")

    if not orchestrator_input:
        raise HTTPException(status_code=400, detail="Failed to parse input data into a valid structure.")

    orchestrator_input.source_id = ingest_request.source_id or orchestrator_input.source_id 
    task_id = ingest_request.task_id or str(uuid.uuid4())
    orchestrator_input.task_id = task_id 
    logging.info(f"Processing task with ID: {task_id} (Source: {orchestrator_input.source_id})")
    try:
        memory.log_step(task_id, "Ingestion request received via API", {"format": ingest_request.data_format, "source_id": orchestrator_input.source_id})
    except Exception as log_e:
        logging.error(f"Failed to log initial receipt for task {task_id}: {log_e}")
    try:
        logging.info(f"Scheduling background processing for task_id: {task_id}")
        background_tasks.add_task(orchestrator.process, orchestrator_input)
    except Exception as e:
        logging.exception(f"Failed to schedule background task for task_id: {task_id}")
        try:
            memory.log_step(task_id, "Failed to schedule background processing", {"error": str(e)})
            memory.save_task_state(task_id, {"status": "failed", "error": "Failed to schedule background task"})
        except Exception as final_log_e:
            logging.error(f"Failed even to log scheduling failure for task {task_id}: {final_log_e}")
        raise HTTPException(status_code=500, detail=f"Failed to schedule background processing: {e}")

    return IngestResponse(
        task_id=task_id,
        status="processing_started",
        message="Data received and processing initiated in the background."
    )

async def get_task_status(
    task_id: str,
    memory: TaskMemory = Depends(get_task_memory)
):
    logging.info(f"Received status request for task_id: {task_id}")
    try:
        task_state = memory.get_task_state(task_id)
    except Exception as e:
        logging.exception(f"Error retrieving task state for {task_id} from memory.")
        raise HTTPException(status_code=500, detail="Internal Server Error: Could not retrieve task status.")

    if not task_state:
        logging.warning(f"Task status request for non-existent task_id: {task_id}")
        raise HTTPException(status_code=404, detail=f"Task with ID '{task_id}' not found.")

    return TaskStatusResponse(
        task_id=task_id,
        status=task_state.get('status', 'unknown'),
        results=task_state.get('agent_outputs'), 
        history=task_state.get('steps_log'), 
        human_feedback_received=task_state.get('human_feedback'),
        timestamp=task_state.get('timestamp') 
    )

async def submit_feedback(
    task_id: str,
    feedback_request: FeedbackRequest = Body(...),
    memory: TaskMemory = Depends(get_task_memory)
):
    logging.info(f"Received feedback submission for task_id: {task_id}")
    feedback_data = feedback_request.feedback
    feedback_data['status_update'] = feedback_request.status_update
    try:
        success = memory.add_human_feedback(task_id, feedback_data)
    except Exception as e:
        logging.exception(f"Error submitting feedback for task {task_id} to memory.")
        raise HTTPException(status_code=500, detail="Internal Server Error: Could not submit feedback.")

    if not success:
        logging.warning(f"Failed to add feedback for task_id: {task_id} (Task not found or update failed).")
        if not memory.get_task_state(task_id):
            raise HTTPException(status_code=404, detail=f"Task with ID '{task_id}' not found.")
        else:
            raise HTTPException(status_code=500, detail=f"Failed to update task {task_id} with feedback.")

    updated_state = memory.get_task_state(task_id)
    new_status = updated_state.get('status', 'unknown') if updated_state else 'unknown'
    return FeedbackResponse(
        task_id=task_id,
        status=new_status,
        message="Feedback submitted successfully."
    )
