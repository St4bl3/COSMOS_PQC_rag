# --- File: utils.py ---
import tiktoken # For token counting if needed (requires 'tiktoken')
import config # Import config for chunk settings
from typing import List # Import List
import logging # Use logging

# --- Utility Functions ---

def count_tokens(text: str, model_name: str = "gpt-4") -> int:
    """Estimates the number of tokens in a text string for a given model."""
    if not text: return 0
    try:
        # Attempt to get encoding for the specified model
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Fallback to a common encoding if the specific model is unknown
        logging.debug(f"Model '{model_name}' not found for token counting. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception as e:
         logging.error(f"Error getting tiktoken encoding: {e}")
         # Fallback: estimate based on character count (very rough)
         return len(text) // 4 # Rough estimate

    try:
        token_count = len(encoding.encode(text))
        return token_count
    except Exception as e:
        logging.error(f"Error encoding text for token count: {e}")
        # Fallback: estimate based on character count
        return len(text) // 4


def chunk_text(text: str, chunk_size: int = config.CHUNK_SIZE, chunk_overlap: int = config.CHUNK_OVERLAP) -> List[str]:
    """
    Splits text into chunks based on character count with overlap.
    More sophisticated methods (e.g., using token count or sentence boundaries) exist.
    """
    if not text:
        return []
    if chunk_size <= 0:
         logging.error("Chunk size must be positive.")
         return [text] # Return original text if chunk size is invalid
    if chunk_overlap < 0:
         logging.warning("Chunk overlap should be non-negative. Setting to 0.")
         chunk_overlap = 0
    if chunk_overlap >= chunk_size:
         logging.warning("Chunk overlap is greater than or equal to chunk size. Setting overlap to chunk_size // 4.")
         chunk_overlap = chunk_size // 4


    chunks = []
    start_index = 0
    text_length = len(text)

    while start_index < text_length:
        end_index = start_index + chunk_size
        # Get the chunk
        chunk = text[start_index:end_index]
        chunks.append(chunk)

        # Move to the next starting point
        start_index += chunk_size - chunk_overlap

        # Optimization: If the remaining text is smaller than overlap, just break
        if text_length - start_index <= chunk_overlap and start_index < text_length:
            # Add the last bit if it wasn't fully covered and overlap is large
            # This logic might need refinement depending on desired behavior for small remainders
            final_chunk_start = start_index - chunk_overlap + chunk_size # Where the previous chunk ended
            if final_chunk_start < text_length:
                 final_chunk = text[final_chunk_start:]
                 if final_chunk and (not chunks or chunks[-1] != final_chunk): # Avoid adding empty or duplicate last chunk
                      # Check if this final bit is already fully contained in the last chunk due to overlap
                      if not chunks[-1].endswith(final_chunk):
                           chunks.append(final_chunk)
            break


    # Post-processing: remove potential empty strings if logic allows
    chunks = [c for c in chunks if c]
    logging.debug(f"Chunked text into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap}).")
    return chunks

