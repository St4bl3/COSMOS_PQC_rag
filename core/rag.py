
# --- File: core/rag.py ---
from typing import List, Dict, Any, Optional
from core.vector_db import VectorDBClient
from core.llm import LLMClient
from core.embedding import EmbeddingClient
import config
import logging # Use logging

# --- Retrieval Augmented Generation (RAG) Module ---

class RAGModule:
    """
    Handles the RAG process: retrieving relevant context and augmenting prompts.
    """
    def __init__(
        self,
        vector_db_client: VectorDBClient,
        llm_client: LLMClient,
        embedding_client: Optional[EmbeddingClient] = None # Optional if VDB handles embeddings
    ):
        self.vector_db = vector_db_client
        self.llm = llm_client
        # Embedding client might only be needed if VDB doesn't handle query embedding
        self.embedding_client = embedding_client or self.vector_db.embedding_client
        logging.info("RAG Module initialized.")

    def retrieve_context(self, query: str, top_k: int = 5, filter: Optional[Dict[str, Any]] = None) -> List[str]:
        """Retrieves relevant text chunks from the vector database."""
        if not self.vector_db:
            logging.error("Vector DB client not available for retrieval.")
            return []

        try:
            logging.info(f"RAG: Retrieving context for query: '{query[:100]}...' (top_k={top_k}, filter={filter})")
            search_results = self.vector_db.query(query_text=query, top_k=top_k, filter=filter)

            if not search_results:
                logging.warning("No relevant context found in vector DB.")
                return []

            # Extract the text content from metadata (assuming it's stored there)
            # Adjust the key 'text' based on how you store the document chunk
            context_chunks = []
            for result in search_results:
                text = result.get("metadata", {}).get("text")
                if text:
                    context_chunks.append(text)
                else:
                    logging.warning(f"Retrieved chunk ID {result.get('id')} has no 'text' in metadata.")

            logging.info(f"Retrieved {len(context_chunks)} context chunks.")
            return context_chunks

        except Exception as e:
            logging.error(f"Error retrieving context: {e}")
            return []

    def generate_with_rag(
        self,
        original_prompt: str,
        query_for_retrieval: Optional[str] = None,
        top_k: int = 3,
        filter: Optional[Dict[str, Any]] = None,
        system_message: Optional[str] = None,
        **llm_kwargs
    ) -> str:
        """
        Generates an LLM response augmented with retrieved context.
        """
        retrieval_query = query_for_retrieval or original_prompt
        context_chunks = self.retrieve_context(retrieval_query, top_k=top_k, filter=filter)

        if not context_chunks:
            logging.warning("RAG: No context retrieved. Generating response without augmentation.")
            # Fallback to direct generation
            try:
                return self.llm.generate(prompt=original_prompt, system_message=system_message, **llm_kwargs)
            except Exception as e:
                 logging.error(f"Error during fallback LLM generation: {e}")
                 return f"Error: Could not generate response. No context found and fallback failed: {e}"


        # --- Construct the augmented prompt ---
        # This is a crucial step and can be customized significantly.
        # Simple example: Prepend context to the original prompt.
        context_string = "\n\n---\n\n".join(context_chunks) # Separator for clarity

        augmented_prompt = f"""Based on the following relevant information:
<CONTEXT>
{context_string}
</CONTEXT>

Please answer the following question or complete the task:
<TASK>
{original_prompt}
</TASK>
"""
        # You might refine this prompt structure, e.g., instructing the LLM
        # on how to use the context, citing sources, etc.

        logging.info("RAG: Generating response with augmented prompt.")
        # Use the LLM client to generate the final response
        try:
            response = self.llm.generate(
                prompt=augmented_prompt,
                system_message=system_message, # Pass along system message if any
                **llm_kwargs # Pass along other LLM parameters
            )
            return response
        except Exception as e:
            logging.error(f"Error generating RAG response: {e}")
            # Fallback or re-raise
            return f"Error generating response with context: {e}" # Or provide a default error message

    async def agenerate_with_rag(
        self,
        original_prompt: str,
        query_for_retrieval: Optional[str] = None,
        top_k: int = 3,
        filter: Optional[Dict[str, Any]] = None,
        system_message: Optional[str] = None,
        **llm_kwargs
     ) -> str:
        """Asynchronously generates an LLM response augmented with retrieved context."""
        # Note: Assumes vector_db.query and llm.agenerate are async compatible
        # If vector_db query is sync, needs to be run in an executor
        # For simplicity, this example keeps retrieval sync but generation async

        retrieval_query = query_for_retrieval or original_prompt
        logging.info(f"RAG (async): Retrieving context for query: '{retrieval_query[:100]}...'")
        # Assuming retrieve_context is synchronous for now
        # For fully async, vector_db.query would need an async version or run_in_executor
        context_chunks = self.retrieve_context(retrieval_query, top_k=top_k, filter=filter)

        if not context_chunks:
            logging.warning("RAG (async): No context retrieved. Generating response without augmentation.")
            try:
                return await self.llm.agenerate(prompt=original_prompt, system_message=system_message, **llm_kwargs)
            except Exception as e:
                 logging.error(f"Error during async fallback LLM generation: {e}")
                 return f"Error: Could not generate async response. No context found and fallback failed: {e}"


        context_string = "\n\n---\n\n".join(context_chunks) # Separator
        augmented_prompt = f"""Based on the following relevant information:
<CONTEXT>
{context_string}
</CONTEXT>

Please answer the following question or complete the task:
<TASK>
{original_prompt}
</TASK>
"""
        logging.info("RAG (async): Generating response with augmented prompt.")
        try:
            response = await self.llm.agenerate(
                prompt=augmented_prompt,
                system_message=system_message,
                **llm_kwargs
            )
            return response
        except Exception as e:
            logging.error(f"Error generating async RAG response: {e}")
            return f"Error generating async response with context: {e}"
