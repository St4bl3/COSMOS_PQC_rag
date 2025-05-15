# --- File: config.py ---
import os
from dotenv import load_dotenv
import logging

load_dotenv()

LOG_LEVEL_FROM_ENV = os.getenv("LOG_LEVEL", "INFO").upper()
numeric_level = getattr(logging, LOG_LEVEL_FROM_ENV, logging.INFO) 

logging.basicConfig(
    level=numeric_level, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- API Keys ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Model Names ---
GROQ_LLM_MODEL = os.getenv("GROQ_LLM_MODEL", "llama3-70b-8192")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2") # Changed default

# --- Vector Database Settings ---
VECTOR_DB_PROVIDER = os.getenv("VECTOR_DB_PROVIDER", "chromadb") 
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-llm-index")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db_storage") # Ensure this path is writable

# --- Memory Settings ---
MEMORY_DB_TYPE = os.getenv("MEMORY_DB_TYPE", "sqlite") 
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "./task_memory.db")

# --- Other Settings ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# --- Security Settings ---
AGENT_KEYS_FILE = os.getenv("AGENT_KEYS_FILE", "agent_pqc_keys.json")
# Fallback for PQC_CRYPTO_AVAILABLE if pqc_crypto_package itself fails to set it
# This allows the app to note that PQC is off even if the package import fails at a higher level.
try:
    from pqc_crypto_package import PQC_CRYPTO_AVAILABLE as PQC_LIB_AVAILABLE
except ImportError:
    PQC_LIB_AVAILABLE = False
# Global flag to easily check if PQC security features should be active
# Can be overridden by an environment variable for explicit disabling even if libs are present
ENABLE_PQC_SECURITY = os.getenv("ENABLE_PQC_SECURITY", "true").lower() == 'true' and PQC_LIB_AVAILABLE


# --- Basic Validation ---
if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY environment variable not set.")
if VECTOR_DB_PROVIDER == "pinecone" and (not PINECONE_API_KEY or not PINECONE_ENVIRONMENT):
    logger.warning("PINECONE_API_KEY or PINECONE_ENVIRONMENT not set for Pinecone provider.")

if not PQC_LIB_AVAILABLE:
    logger.critical("PQC crypto libraries (kyber, dilithium, pycryptodomex) are NOT properly installed or imported. PQC security features will be DISABLED.")
elif not ENABLE_PQC_SECURITY and PQC_LIB_AVAILABLE:
     logger.warning("PQC crypto libraries ARE available, but ENABLE_PQC_SECURITY is set to false. Secure communication will be bypassed.")
elif ENABLE_PQC_SECURITY:
    logger.info("PQC crypto libraries available and security features are ENABLED.")

