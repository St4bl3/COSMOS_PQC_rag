# --- File: ingest_rag_docs.py ---
import os
import argparse
import uuid
from typing import List, Dict, Any, Optional

# Import necessary components from your project structure
import config 
from core.embedding import EmbeddingClient
from core.vector_db import VectorDBClient
from utils import chunk_text
from ingestion.parser import DataParser, EBOOKLIB_AVAILABLE 

import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def get_file_paths(directory: str, supported_extensions: List[str] = None) -> List[str]:
    """Recursively finds all files with supported extensions in a directory."""
    if supported_extensions is None:
        supported_extensions = ['.txt', '.pdf', '.md', '.epub']
    
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in supported_extensions):
                file_paths.append(os.path.join(root, file))
    logger.info(f"Found {len(file_paths)} files in '{directory}' with extensions {supported_extensions}.")
    return file_paths

def process_document(
    file_path: str,
    embedding_client: EmbeddingClient,
    vector_db_client: VectorDBClient,
    data_parser: DataParser, # Pass DataParser instance
    doc_type: str = "general_document", 
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
    custom_metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Reads a document using DataParser, chunks it, creates embeddings, and upserts to Vector DB.
    """
    logger.info(f"Processing document: {file_path} with doc_type: '{doc_type}'")
    
    # 1. Read content using DataParser
    file_extension = data_parser._guess_file_type(file_path) # Use parser's guess method
    
    if not file_extension:
        logger.warning(f"Could not determine file type for {file_path}. Skipping.")
        return False
        
    try:
        # DataParser._read_file_content will now handle EPUBs
        content = data_parser._read_file_content(file_path, file_extension)
        if not content: # _read_file_content now returns None if content is empty or error
            logger.warning(f"No content extracted or error during extraction from {file_path}. Skipping.")
            return False
        logger.info(f"Successfully extracted content from {file_path} (length: {len(content)} chars).")
    except Exception as e: # Catch any unexpected errors from parser interaction
        logger.error(f"Error reading or parsing document {file_path} via DataParser: {e}", exc_info=True)
        return False

    # 2. Chunk the text
    text_chunks = chunk_text(content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not text_chunks:
        logger.warning(f"No text chunks generated for {file_path}. Skipping.")
        return False
    logger.info(f"Document {file_path} split into {len(text_chunks)} chunks.")

    # 3. Generate embeddings for chunks
    try:
        embeddings = embedding_client.get_embeddings(text_chunks)
        if not embeddings or len(embeddings) != len(text_chunks):
            logger.error(f"Failed to generate embeddings or mismatch in count for {file_path}.")
            return False
        logger.info(f"Generated {len(embeddings)} embeddings for chunks from {file_path}.")
    except Exception as e:
        logger.error(f"Error generating embeddings for {file_path}: {e}")
        return False

    # 4. Prepare items for upserting
    items_to_upsert = []
    doc_id = str(uuid.uuid4()) 
    for i, chunk_text_content in enumerate(text_chunks):
        chunk_id = f"{doc_id}-chunk-{i}"
        metadata = {
            "source_document_id": doc_id,
            "source_file_name": os.path.basename(file_path),
            "original_file_path": file_path, 
            "chunk_id": chunk_id,
            "text": chunk_text_content, 
            "doc_type": doc_type, 
            "chunk_index": i,
            "total_chunks_in_doc": len(text_chunks)
        }
        if custom_metadata:
            metadata.update(custom_metadata)
            
        items_to_upsert.append((chunk_id, embeddings[i], metadata))

    # 5. Upsert to Vector DB
    try:
        success = vector_db_client.upsert(items_to_upsert)
        if success:
            logger.info(f"Successfully upserted {len(items_to_upsert)} chunks for {file_path} to {vector_db_client.provider}.")
            return True
        else:
            logger.error(f"Failed to upsert chunks for {file_path} to {vector_db_client.provider}.")
            return False
    except Exception as e:
        logger.error(f"Error upserting chunks for {file_path} to {vector_db_client.provider}: {e}")
        return False

# --- Main Script Logic ---
def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG system's vector database.")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing documents to ingest.")
    parser.add_argument("--doc_type", type=str, default="general_document", help="The 'doc_type' metadata tag for these documents (e.g., 'sop', 'safety_manual', 'technical_spec', 'ebook').")
    parser.add_argument("--chunk_size", type=int, default=config.CHUNK_SIZE, help="Character chunk size for splitting documents.")
    parser.add_argument("--chunk_overlap", type=int, default=config.CHUNK_OVERLAP, help="Character overlap between chunks.")
    
    args = parser.parse_args()

    logger.info("--- Starting RAG Document Ingestion ---")
    logger.info(f"Input directory: {args.dir}")
    logger.info(f"Document type for metadata: {args.doc_type}")
    logger.info(f"Chunk size: {args.chunk_size}, Chunk overlap: {args.chunk_overlap}")


    # Initialize core components
    try:
        embedding_client = EmbeddingClient()
        vector_db_client = VectorDBClient(embedding_client=embedding_client)
        data_parser = DataParser() # Instantiate DataParser
    except Exception as e:
        logger.error(f"Failed to initialize core components: {e}")
        logger.error("Ensure your API keys and configurations in .env and config.py are correct.")
        return

    # Check if embedding client is initialized
    if not embedding_client.client:
        logger.error("Embedding client failed to initialize. Cannot proceed.")
        return
        
    # Check if Ebooklib is available if EPUBs are being processed
    # A simple way to check if EPUBs might be in the target directory is to scan for .epub extension
    # This is not foolproof if --dir points to a single file, but good for directory processing.
    processing_epubs = False
    if os.path.isdir(args.dir):
        if any(fname.lower().endswith('.epub') for fname in os.listdir(args.dir)): # Quick check in top-level dir
            processing_epubs = True
    elif args.dir.lower().endswith('.epub'): # If --dir is a single epub file
        processing_epubs = True
    
    if processing_epubs and not EBOOKLIB_AVAILABLE:
        logger.error("Attempting to process EPUB files, but 'ebooklib' and/or 'BeautifulSoup4' are not installed or failed to import.")
        logger.error("Please install them: pip install ebooklib beautifulsoup4")
        return

    if not vector_db_client.index:
        logger.error("Vector DB client or index/collection failed to initialize. Cannot proceed.")
        return

    file_paths = get_file_paths(args.dir) # get_file_paths now includes .epub
    if not file_paths:
        logger.info("No documents found in the specified directory. Exiting.")
        return

    successful_ingestions = 0
    failed_ingestions = 0

    for file_path in file_paths:
        if process_document(
            file_path,
            embedding_client,
            vector_db_client,
            data_parser, # Pass the DataParser instance
            doc_type=args.doc_type,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        ):
            successful_ingestions += 1
        else:
            failed_ingestions += 1
        logger.info("-" * 50)

    logger.info("--- RAG Document Ingestion Complete ---")
    logger.info(f"Successfully ingested documents: {successful_ingestions}")
    logger.info(f"Failed to ingest documents: {failed_ingestions}")

if __name__ == "__main__":
    main()
