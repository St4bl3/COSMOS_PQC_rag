# --- File: core/vector_db.py ---
from typing import List, Dict, Any, Optional, Tuple
import config
from core.embedding import EmbeddingClient # Assuming EmbeddingClient is defined
import logging # Use logging

# --- Vector Database Interaction Wrapper ---

class VectorDBClient:
    """
    Wrapper for interacting with the Vector Database (Pinecone or local alternative).
    Handles upserting, querying, and deleting vectors.
    """
    def __init__(
        self,
        provider: str = config.VECTOR_DB_PROVIDER,
        embedding_client: EmbeddingClient = None # Inject embedding client
    ):
        self.provider = provider
        self.embedding_client = embedding_client or EmbeddingClient() # Default instance
        self.index = self._initialize_index()
        if self.index:
             logging.info(f"Vector DB Client initialized using provider: {self.provider}")
        else:
             logging.error(f"Vector DB Client initialization FAILED for provider: {self.provider}")


    def _initialize_index(self):
        """Initializes the connection to the specific vector database index."""
        if self.provider == "pinecone":
            try:
                from pinecone import Pinecone, ServerlessSpec, PodSpec # Updated import
                if not config.PINECONE_API_KEY or not config.PINECONE_ENVIRONMENT:
                    logging.error("Pinecone API key or environment not configured.")
                    return None

                logging.info("Initializing Pinecone connection...")
                pc = Pinecone(api_key=config.PINECONE_API_KEY) 

                index_name = config.PINECONE_INDEX_NAME
                logging.info(f"Checking Pinecone index '{index_name}'...")

                if index_name not in pc.list_indexes().names:
                    logging.info(f"Creating Pinecone index '{index_name}'...")
                    if not self.embedding_client or not self.embedding_client.client:
                        logging.error("Cannot determine embedding dimension without initialized embedding client.")
                        return None
                    try:
                        dummy_embedding = self.embedding_client.get_embedding("dimension_check")
                        if not dummy_embedding:
                            logging.error("Failed to get dummy embedding for dimension calculation.")
                            return None
                        dimension = len(dummy_embedding)
                        logging.info(f"Determined embedding dimension: {dimension}")
                    except Exception as e:
                        logging.error(f"Error getting embedding dimension: {e}")
                        return None

                    if 'serverless' in config.PINECONE_ENVIRONMENT.lower() or 'starter' in config.PINECONE_ENVIRONMENT.lower():
                        spec = ServerlessSpec(cloud='aws', region='us-west-2') # Example, adjust as needed
                        logging.info(f"Using ServerlessSpec for index creation (cloud: {spec.cloud}, region: {spec.region}).")
                    else:
                        spec = PodSpec(environment=config.PINECONE_ENVIRONMENT)
                        logging.info(f"Using PodSpec for index creation (environment: {spec.environment}).")

                    pc.create_index(
                            name=index_name,
                            dimension=dimension,
                            metric="cosine", 
                            spec=spec
                        )
                    import time
                    time.sleep(10) 
                    logging.info(f"Pinecone index '{index_name}' created.")
                else:
                    logging.info(f"Connecting to existing Pinecone index '{index_name}'...")
                return pc.Index(index_name)
            except ImportError:
                logging.error("'pinecone-client' package not installed.")
                return None
            except Exception as e:
                logging.error(f"Error initializing Pinecone: {e}")
                return None

        elif self.provider == "chromadb":
            try:
                import chromadb
                    # Import Settings explicitly for newer versions if needed for clarity or specific options
                from chromadb.config import Settings as ChromaSettings 

                logging.info(f"Initializing ChromaDB client at path: {config.CHROMA_DB_PATH}")
                    
                    # Ensure the directory exists or ChromaDB can create it.
                    # Explicitly use settings to ensure local, persistent mode.
                    # This helps avoid ambiguity that might lead to the "http-only client mode" error.
                    
                client = chromadb.PersistentClient(
                        path=config.CHROMA_DB_PATH,
                        settings=ChromaSettings(anonymized_telemetry=False) 
                                                                    
                    )
                
                # Alternative for some newer versions if PersistentClient is still problematic:
                # settings = ChromaSettings(
                #        chroma_db_impl="duckdb+parquet", # Default for PersistentClient
                #        persist_directory=config.CHROMA_DB_PATH,
                #        anonymized_telemetry=False
                       
                #     )
                # client = chromadb.Client(settings)


                index_name = config.PINECONE_INDEX_NAME # Reuse name for consistency
                logging.info(f"Getting or creating ChromaDB collection '{index_name}'...")
                    
                collection = client.get_or_create_collection(
                    name=index_name,
                        # If using a SentenceTransformer model with ChromaDB's internal embedding function support:
                        # embedding_function=chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(model_name=config.EMBEDDING_MODEL)
                        # However, your current setup generates embeddings externally via EmbeddingClient, which is fine.
                        # So, no embedding_function is needed here if you provide embeddings during upsert.
                    )
                logging.info(f"Connected to ChromaDB collection '{index_name}'.")
                return collection 

            except ImportError:
                logging.error("'chromadb' package not installed.")
                return None
            except Exception as e:
                logging.error(f"Error initializing ChromaDB: {e}", exc_info=True) # Log with traceback
                return None

        else:
            logging.error(f"Unsupported vector database provider: {self.provider}")
            return None

    def upsert(self, items: List[Tuple[str, List[float], Dict[str, Any]]]) -> bool:
        """
        Upserts items into the vector database.
        Each item is a tuple: (id, vector, metadata).
        """
        if not self.index:
            logging.error("Vector DB index not initialized.")
            return False
        if not items:
            logging.warning("No items provided for upsert.")
            return False

        try:
            if self.provider == "pinecone":
                # Pinecone expects vectors in the upsert call
                vectors_to_upsert = [(item[0], item[1], item[2]) for item in items]
                logging.debug(f"Upserting {len(vectors_to_upsert)} items to Pinecone...")
                self.index.upsert(vectors=vectors_to_upsert)
                logging.info(f"Upserted {len(vectors_to_upsert)} items to Pinecone.")

            elif self.provider == "chromadb":
                # ChromaDB expects ids, embeddings, metadatas, and optionally documents
                ids = [item[0] for item in items]
                embeddings = [item[1] for item in items]
                metadatas = [item[2] for item in items]
                 # Optionally include original documents if available/needed
                # documents = [item[2].get('text', '') for item in items] # Example

                logging.debug(f"Upserting {len(ids)} items to ChromaDB...")
                self.index.upsert( # 'index' here is the Chroma collection
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    # documents=documents # Uncomment if adding documents
                )
                logging.info(f"Upserted {len(ids)} items to ChromaDB.")

            return True
        except Exception as e:
            logging.error(f"Error during vector DB upsert: {e}")
            return False

    def query(self, query_text: str, top_k: int = 5, filter: Optional[Dict[str, Any]] = None) -> Optional[List[Dict[str, Any]]]:
        """
        Queries the vector database using text.
        Requires an embedding client to convert text to vector.
        Returns a list of matches with metadata and scores.
        """
        if not self.index or not self.embedding_client:
            logging.error("Vector DB index or embedding client not initialized for query.")
            return None

        try:
            logging.debug(f"Generating embedding for query: '{query_text[:100]}...'")
            query_vector = self.embedding_client.get_embedding(query_text)
            if not query_vector:
                logging.error("Failed to generate query embedding.")
                return None

            logging.debug(f"Querying {self.provider} with top_k={top_k} and filter={filter}")
            if self.provider == "pinecone":
                query_result = self.index.query(
                    vector=query_vector,
                    top_k=top_k,
                    include_metadata=True,
                    filter=filter # Pass Pinecone-specific filter dict if provided
                )
                # Format results consistently
                results = [
                    {"id": match.id, "score": match.score, "metadata": match.metadata}
                    for match in query_result.matches
                ]
                logging.info(f"Pinecone query returned {len(results)} results.")
                return results

            elif self.provider == "chromadb":
                # ChromaDB query structure
                query_result = self.index.query( # 'index' is the Chroma collection
                    query_embeddings=[query_vector], # Note: expects a list of embeddings
                    n_results=top_k,
                    where=filter, # Pass Chroma-specific 'where' filter dict
                    include=['metadatas', 'distances'] # Or 'documents'
                )
                # Format results (distances are often used instead of scores, lower is better)
                # Access results carefully, structure might differ slightly based on version
                results = []
                # Check if results are structured as expected (can vary slightly between versions)
                if query_result and query_result.get('ids') and query_result.get('ids')[0]:
                     ids = query_result['ids'][0]
                     distances = query_result['distances'][0] if query_result.get('distances') else [None] * len(ids)
                     metadatas = query_result['metadatas'][0] if query_result.get('metadatas') else [{}] * len(ids)
                     for i in range(len(ids)):
                         # Calculate similarity score (cosine distance -> similarity)
                         similarity_score = 1.0 - distances[i] if distances[i] is not None else 0.0
                         results.append({
                             "id": ids[i],
                             "score": similarity_score, # Higher is better
                             "distance": distances[i], # Lower is better
                             "metadata": metadatas[i]
                         })
                logging.info(f"ChromaDB query returned {len(results)} results.")
                return results

            else:
                logging.error(f"Query logic not implemented for provider: {self.provider}")
                return None # Should not happen if initialized correctly

        except Exception as e:
            logging.error(f"Error during vector DB query: {e}")
            return None

    # Add delete methods etc. as needed
    # def delete(...)