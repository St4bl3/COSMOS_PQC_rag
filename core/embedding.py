# --- File: core/embedding.py ---
from typing import List, Optional
import config
import logging # Use logging

# --- Embedding Model Wrapper ---

class EmbeddingClient:
    """
    Wrapper for generating text embeddings.
    Supports different providers (e.g., OpenAI, SentenceTransformers).
    """
    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        self.model_name = model_name
        self.client = self._initialize_client()
        if self.client:
            logging.info(f"Embedding Client initialized for model: {self.model_name}")
        else:
            logging.error(f"Failed to initialize Embedding Client for model: {self.model_name}")


    def _initialize_client(self):
        """Initializes the appropriate embedding client based on configuration."""
        # Example: Using OpenAI (requires 'openai' package)
        if "text-embedding-" in self.model_name:
            try:
                from openai import OpenAI, AsyncOpenAI
                # Ensure OPENAI_API_KEY is set in config or environment
                if not config.OPENAI_API_KEY:
                     logging.warning("OPENAI_API_KEY not set. OpenAI embeddings will fail.")
                     # Fallback or raise error
                     return None # Or initialize a dummy client

                # You might want separate sync/async clients if using both
                # Return sync client for now
                return OpenAI(api_key=config.OPENAI_API_KEY)
            except ImportError:
                logging.error("'openai' package not installed. Install it to use OpenAI embeddings.")
                return None # Or raise error
            except Exception as e:
                logging.error(f"Error initializing OpenAI embedding client: {e}")
                return None

        # Example: Using SentenceTransformers (requires 'sentence-transformers' package)
        elif "MiniLM" in self.model_name or "all-" in self.model_name: # Add other common prefixes
             try:
                 from sentence_transformers import SentenceTransformer
                 # Model will be downloaded on first use if not cached
                 logging.info(f"Initializing SentenceTransformer model: {self.model_name} (may download)...")
                 model = SentenceTransformer(self.model_name)
                 logging.info(f"SentenceTransformer model {self.model_name} loaded.")
                 return model
             except ImportError:
                 logging.error("'sentence-transformers' package not installed. Install it for local embeddings.")
                 return None # Or raise error
             except Exception as e:
                 logging.error(f"Error initializing SentenceTransformer model {self.model_name}: {e}")
                 return None
        else:
            logging.warning(f"Unsupported embedding model type for '{self.model_name}'.")
            # Raise error or return a dummy client
            return None

    def get_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Generates embeddings for a list of texts."""
        if not self.client:
            logging.error("Embedding client not initialized.")
            return None
        if not texts:
            return [] # Return empty list if no texts provided

        try:
            logging.debug(f"Generating embeddings for {len(texts)} texts...")
            if hasattr(self.client, 'embeddings'): # OpenAI client structure (v1+)
                 response = self.client.embeddings.create(
                     input=texts,
                     model=self.model_name
                 )
                 embeddings = [embedding.embedding for embedding in response.data]
            elif hasattr(self.client, 'encode'): # SentenceTransformers structure
                 embeddings = self.client.encode(texts, show_progress_bar=False).tolist() # Added show_progress_bar
            else:
                 logging.error("Unknown embedding client type.")
                 return None
            logging.debug(f"Successfully generated {len(embeddings)} embeddings.")
            return embeddings
        except Exception as e:
            logging.error(f"Error generating embeddings: {e}")
            # Add more specific error handling
            return None

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Generates embedding for a single text."""
        if not text: # Handle empty string case
            return None
        embeddings = self.get_embeddings([text])
        return embeddings[0] if embeddings else None