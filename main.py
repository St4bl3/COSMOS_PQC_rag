# --- File: main.py ---
from fastapi import FastAPI, Depends, HTTPException, Request, Response 
from fastapi.responses import JSONResponse 
from fastapi.middleware.cors import CORSMiddleware
from api import endpoints 
from api.models import IngestResponse, TaskStatusResponse, FeedbackResponse 
from core.memory import TaskMemory
from core.llm import LLMClient
from core.embedding import EmbeddingClient
from core.vector_db import VectorDBClient
from core.rag import RAGModule
from agents.orchestrator import OrchestrationAgent # Security-aware version
from agents.specialized_agents import ( # Security-aware versions
    SOPAgent, SystemCmdAgent, DisasterAgent, 
    AsteroidAgent, GroundCmdAgent, CodeGenAgent
)
from contextlib import asynccontextmanager
import uvicorn 
import logging 
from typing import Optional 
import os
import config # Your config file

# Security components
from security.key_manager import KeyManager
from security.secure_communication import SecureCommunicator
# PQC_CRYPTO_AVAILABLE is checked via config.ENABLE_PQC_SECURITY which itself checks PQC_LIB_AVAILABLE

# Global instances for DI, set during lifespan
# These are for core application components
# Initialize to None to ensure they are set by lifespan or getters raise errors
if not hasattr(endpoints, '_embedding_client_instance'): endpoints._embedding_client_instance = None
if not hasattr(endpoints, '_llm_client_instance'): endpoints._llm_client_instance = None
if not hasattr(endpoints, '_vector_db_client_instance'): endpoints._vector_db_client_instance = None
if not hasattr(endpoints, '_task_memory_instance_ref'): endpoints._task_memory_instance_ref = None
if not hasattr(endpoints, '_rag_module_instance'): endpoints._rag_module_instance = None
if not hasattr(endpoints, '_orchestrator_instance'): endpoints._orchestrator_instance = None

# Global instances for security components
_key_manager_instance: Optional[KeyManager] = None
_secure_communicator_instance: Optional[SecureCommunicator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    logging.info("Application startup sequence initiated...")
    
    global _key_manager_instance, _secure_communicator_instance
    if config.ENABLE_PQC_SECURITY: 
        logging.info("PQC Security is ENABLED. Initializing security components...")
        if _key_manager_instance is None:
            logging.info("Lifespan: Initializing KeyManager...")
            _key_manager_instance = KeyManager(key_file_path=config.AGENT_KEYS_FILE)
        
        if _secure_communicator_instance is None:
            if not _key_manager_instance: 
                 logging.error("Lifespan: KeyManager failed to initialize prior to SecureCommunicator. Attempting KeyManager init now.")
                 _key_manager_instance = KeyManager(key_file_path=config.AGENT_KEYS_FILE)
                 if not _key_manager_instance: 
                    logging.critical("Lifespan: CRITICAL - KeyManager failed to initialize, cannot create SecureCommunicator.")
            if _key_manager_instance: 
                logging.info("Lifespan: Initializing SecureCommunicator...")
                _secure_communicator_instance = SecureCommunicator(key_manager=_key_manager_instance)
                logging.info("Lifespan: SecureCommunicator initialized.")
            else: # KeyManager still None after re-attempt
                 _secure_communicator_instance = None # Ensure it's None if KeyManager failed
                 logging.error("Lifespan: SecureCommunicator not initialized due to KeyManager failure.")

    else: # PQC Security is disabled
        logging.warning("PQC Security is DISABLED (either by config or missing libraries). Intra-agent communication will be INSECURE.")
        _key_manager_instance = None
        _secure_communicator_instance = None


    logging.info("Initializing core application components...")
    try:
        # 1. Initialize simple singletons first
        if endpoints._embedding_client_instance is None:
            logging.info("Lifespan: Initializing EmbeddingClient...")
            endpoints._embedding_client_instance = EmbeddingClient()
            if not endpoints._embedding_client_instance.client:
                 logging.error("Lifespan: EmbeddingClient's internal client failed to initialize.")
            else:
                logging.info("Lifespan: EmbeddingClient initialized successfully.")
        
        if endpoints._llm_client_instance is None:
            logging.info("Lifespan: Initializing LLMClient...")
            endpoints._llm_client_instance = LLMClient()
            logging.info("Lifespan: LLMClient initialization attempt complete.")

        # 2. Initialize components that depend on the simple ones
        if endpoints._vector_db_client_instance is None:
            logging.info("Lifespan: Initializing VectorDBClient...")
            if not endpoints._embedding_client_instance or not endpoints._embedding_client_instance.client:
                logging.warning("Lifespan: EmbeddingClient not available for VectorDBClient. Attempting re-init.")
                endpoints._embedding_client_instance = EmbeddingClient() 
                if not endpoints._embedding_client_instance.client: # Still failed
                    logging.error("Lifespan: CRITICAL - Failed to initialize EmbeddingClient for VectorDBClient.")
                    # Set VDB to None or handle error to prevent further issues
                    endpoints._vector_db_client_instance = None # type: ignore 
                else: # Embedding client re-init success
                     endpoints._vector_db_client_instance = VectorDBClient(
                        embedding_client=endpoints._embedding_client_instance
                    )
            else: # Embedding client was already fine
                endpoints._vector_db_client_instance = VectorDBClient(
                    embedding_client=endpoints._embedding_client_instance
                )
            
            if endpoints._vector_db_client_instance and not endpoints._vector_db_client_instance.index:
                logging.error("Lifespan: VectorDBClient's index/collection failed to initialize.")
            elif endpoints._vector_db_client_instance:
                logging.info("Lifespan: VectorDBClient initialized successfully.")
            else:
                logging.error("Lifespan: VectorDBClient could not be initialized.")


        if endpoints._task_memory_instance_ref is None:
            logging.info("Lifespan: Initializing TaskMemory...")
            endpoints._task_memory_instance_ref = TaskMemory(
                vector_db_client=endpoints._vector_db_client_instance # Pass potentially None VDB client
            )
            logging.info("Lifespan: TaskMemory initialization attempt complete.")

        # 3. Initialize RAGModule
        if endpoints._rag_module_instance is None:
            logging.info("Lifespan: Initializing RAGModule...")
            # RAGModule constructor expects non-None for these if it's to function fully
            # Check if essential dependencies for RAG are available
            llm_for_rag = endpoints._llm_client_instance
            vdb_for_rag = endpoints._vector_db_client_instance
            emb_for_rag = endpoints._embedding_client_instance

            if not (llm_for_rag and vdb_for_rag and vdb_for_rag.index and emb_for_rag and emb_for_rag.client):
                logging.error("Lifespan: One or more CRITICAL dependencies for RAGModule (LLM, active VDB, active EmbeddingClient) are not initialized. RAG will be impaired.")
                # RAGModule might still be instantiated but log its own warnings if dependencies are problematic.
            
            endpoints._rag_module_instance = RAGModule(
                vector_db_client=vdb_for_rag, # type: ignore
                llm_client=llm_for_rag, # type: ignore
                embedding_client=emb_for_rag # type: ignore
            )
            logging.info("Lifespan: RAGModule initialized (check logs for dependency status).")

        # 4. Initialize Orchestrator - Pass SecureCommunicator
        if endpoints._orchestrator_instance is None:
            logging.info("Lifespan: Initializing Orchestrator...")
            agent_classes = {
                "SOPAgent": SOPAgent, "SystemCmdAgent": SystemCmdAgent,
                "DisasterAgent": DisasterAgent, "AsteroidAgent": AsteroidAgent,
                "GroundCmdAgent": GroundCmdAgent, "CodeGenAgent": CodeGenAgent,
            }
            # Ensure dependencies are actual instances
            if not (endpoints._llm_client_instance and endpoints._rag_module_instance and endpoints._task_memory_instance_ref):
                logging.error("Lifespan: Cannot initialize Orchestrator due to missing LLM, RAG, or Memory instances.")
                # Depending on criticality, you might raise an error to stop the app
                # For now, it will proceed, and Orchestrator might fail if these are None
            
            endpoints._orchestrator_instance = OrchestrationAgent(
                llm_client=endpoints._llm_client_instance, # type: ignore
                rag_module=endpoints._rag_module_instance, # type: ignore
                memory=endpoints._task_memory_instance_ref, # type: ignore
                agent_classes=agent_classes,
                secure_communicator=_secure_communicator_instance, # Pass instance (can be None if security disabled)
                agent_id="Orchestrator" 
            )
            logging.info("Lifespan: Orchestrator initialized successfully.")
        logging.info("All core components pre-initialized via lifespan.")

    except Exception as e:
        logging.exception("FATAL: Error during application startup initialization.")
    
    yield 

    # --- Shutdown ---
    logging.info("Application shutdown sequence initiated...")
    if endpoints._task_memory_instance_ref:
        endpoints._task_memory_instance_ref.close() 
        logging.info("Task Memory resources released.")
    logging.info("Application shutdown complete.")

app = FastAPI(
    title="Agentic LLM System API",
    description="API for ingesting data, securely generating SOPs/Commands, and managing human feedback.",
    version="0.3.0", 
    lifespan=lifespan 
)

origins = os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000").split(",")
logging.info(f"CORS allowed origins: {origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_secure_communicator() -> Optional[SecureCommunicator]:
    if not config.ENABLE_PQC_SECURITY: return None
    if _secure_communicator_instance is None:
        logging.critical("SecureCommunicator instance is None during request, but PQC security is enabled. This indicates a severe startup issue.")
        raise HTTPException(status_code=503, detail="Security system not available or not initialized.")
    return _secure_communicator_instance

# Update endpoint dependency injection in api/endpoints.py to use these getters
# Example: orchestrator: OrchestrationAgent = Depends(endpoints.get_orchestrator)
# where get_orchestrator would use the global instance or init with resolved dependencies.

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logging.error(f"HTTP Exception: Status Code={exc.status_code}, Detail={exc.detail}, Path: {request.url.path}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": f"An error occurred: {exc.detail}"},
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logging.exception(f"Unhandled Exception at Path {request.url.path}: {exc}") 
    return JSONResponse(
        status_code=500,
        content={"message": "An unexpected internal server error occurred. Please check server logs."},
    )

app.post(
    "/ingest", response_model=IngestResponse, summary="Ingest Data and Start Processing",
    tags=["Tasks"], status_code=202 
)(endpoints.ingest_data) 

app.get(
    "/tasks/{task_id}", response_model=TaskStatusResponse, summary="Get Task Status and Results",
    tags=["Tasks"]
)(endpoints.get_task_status)

app.post(
    "/tasks/{task_id}/feedback", response_model=FeedbackResponse, summary="Submit Human Feedback for a Task",
    tags=["Human-in-the-Loop"]
)(endpoints.submit_feedback)

@app.get("/", summary="Root endpoint", tags=["General"], include_in_schema=False) 
async def read_root():
    return {"message": "Welcome to the Secure Agentic LLM System API! See /docs for details."}

if __name__ == "__main__":
    logging.info("Starting Secure Agentic LLM System API server using Uvicorn...")
    if not config.GROQ_API_KEY:
        logging.critical("GROQ_API_KEY environment variable is not set. LLM functionality will fail.")
    if config.ENABLE_PQC_SECURITY:
        logging.info("PQC Security features are configured to be ENABLED.")
    else:
        logging.warning("PQC Security features are configured to be DISABLED.")
    
    log_level = os.getenv("LOG_LEVEL", "debug").lower()
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload_enabled = os.getenv("RELOAD", "true").lower() == "true"

    logging.info(f"Server starting on {host}:{port} with log level {log_level} and reload {'enabled' if reload_enabled else 'disabled'}")
    
    uvicorn.run(
        "main:app", 
        host=host, 
        port=port, 
        log_level=log_level, 
        reload=reload_enabled
    )
