# --- File: ingestion/parser.py ---
from typing import Optional, Dict, Any, Union
from ingestion.models import DisasterWeatherData, SpaceDebrisData, TextInputData, FileInputData, OrchestratorInput
import PyPDF2 # Example for PDF parsing (requires 'pypdf2')
import json
import requests # For fetching files from URLs
import logging # Use logging
import uuid # For generating default IDs
from urllib.parse import urlparse
from io import BytesIO

# Added for EPUB support
try:
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup 
    EBOOKLIB_AVAILABLE = True
except ImportError:
    EBOOKLIB_AVAILABLE = False
    logging.warning("'ebooklib' or 'BeautifulSoup4' not installed. EPUB parsing will not be available.")

# --- Data Parsing and Structuring Logic ---

class DataParser:
    """Parses various input formats into structured Pydantic models."""

    def parse(self, input_data: Union[Dict[str, Any], str], data_format: str = 'json') -> Optional[OrchestratorInput]:
        """
        Parses input data into the OrchestratorInput model.
        Handles different formats like JSON, raw text, file paths/URLs.
        """
        logging.info(f"Parsing input data (format: {data_format})...")
        try:
            if data_format == 'json':
                if isinstance(input_data, str):
                    logging.debug("Parsing JSON string.")
                    parsed_dict = json.loads(input_data)
                elif isinstance(input_data, dict):
                    logging.debug("Parsing dictionary.")
                    parsed_dict = input_data
                else:
                    raise ValueError("Invalid input type for JSON format. Expected dict or JSON string.")
                return self._parse_dict(parsed_dict)

            elif data_format == 'text':
                if not isinstance(input_data, str):
                    raise ValueError("Input must be a string for 'text' format.")
                logging.debug("Parsing raw text input.")
                text_data = TextInputData(text_content=input_data)
                default_id = "text_input_" + str(hash(input_data))[:10]
                return OrchestratorInput(
                    source_id=default_id,
                    data_type="text",
                    data=text_data
                )

            elif data_format == 'file' or data_format == 'file_url':
                if not isinstance(input_data, str):
                    raise ValueError("Input must be a file path or URL string for 'file'/'file_url' format.")
                logging.debug(f"Parsing file input: {input_data}")

                file_type = self._guess_file_type(input_data)
                if not file_type:
                    raise ValueError(f"Could not determine file type for: {input_data}")
                logging.debug(f"Guessed file type: {file_type}")

                file_content = self._read_file_content(input_data, file_type)
                if file_content is None:
                    raise ValueError(f"Could not read content from: {input_data}")
                logging.info(f"Successfully read content from {input_data} ({len(file_content)} chars).")

                text_data = TextInputData(text_content=file_content)
                default_id = "file_content_" + str(hash(input_data))[:10]
                return OrchestratorInput(
                    source_id=default_id,
                    data_type="text", # Treat parsed content as text for downstream agents
                    data=text_data,
                    metadata={"original_source": input_data, "original_format": file_type}
                )

            else:
                logging.error(f"Unsupported data format '{data_format}'")
                return None

        except json.JSONDecodeError as e:
            logging.error(f"Error parsing JSON input: {e}")
            return None
        except ValueError as e:
            logging.error(f"Error parsing input: {e}")
            return None
        except Exception as e:
            logging.exception(f"An unexpected error occurred during parsing: {e}")
            return None

    def _parse_dict(self, data_dict: Dict[str, Any]) -> Optional[OrchestratorInput]:
        """Parses a dictionary, attempting to fit it into known data models."""
        logging.debug(f"Parsing dictionary payload: {str(data_dict)[:200]}...")
        data_type = data_dict.get('data_type')
        data_payload = data_dict.get('data', data_dict)
        if not isinstance(data_payload, dict):
            logging.error(f"Expected 'data' field to be a dictionary or the top level input to be one, but got {type(data_payload)}")
            if isinstance(data_dict.get('data'), dict):
                data_payload = data_dict
            else:
                return None

        if not data_type:
            logging.debug("Attempting to infer data_type from keys...")
            if 'event_type' in data_payload and 'location' in data_payload:
                data_type = 'disaster'
            elif 'object_id' in data_payload and 'trajectory' in data_payload:
                data_type = 'debris'
            elif 'text_content' in data_payload:
                data_type = 'text'
            elif 'file_path' in data_payload and 'file_type' in data_payload:
                data_type = 'file'
            else:
                logging.warning("Could not determine data type from keys. Treating as generic text.")
                try:
                    text_content = json.dumps(data_payload)
                    data_type = "text"
                    data_payload = {"text_content": text_content}
                except Exception as json_e:
                    logging.error(f"Could not serialize unknown data payload to JSON for text fallback: {json_e}")
                    return None
            logging.debug(f"Inferred data_type: {data_type}")

        try:
            logging.debug(f"Validating data against Pydantic model for type: {data_type}")
            if data_type == 'disaster':
                specific_data = DisasterWeatherData(**data_payload)
            elif data_type == 'debris':
                specific_data = SpaceDebrisData(**data_payload)
            elif data_type == 'text':
                specific_data = TextInputData(**data_payload)
            elif data_type == 'file': # This case is less likely if we parse file content immediately
                specific_data = FileInputData(**data_payload)
            else:
                raise ValueError(f"Unsupported final data_type '{data_type}' during validation.")

            source_id = data_dict.get('source_id', "dict_input_" + str(uuid.uuid4())[:8])
            task_id = data_dict.get('task_id')
            timestamp = data_dict.get('timestamp')
            metadata = data_dict.get('metadata', {})

            logging.info(f"Successfully parsed and validated input as '{data_type}' for source_id: {source_id}")
            return OrchestratorInput(
                source_id=source_id,
                timestamp=timestamp,
                metadata=metadata,
                data_type=data_type,
                data=specific_data,
                task_id=task_id
            )
        except Exception as e:
            logging.error(f"Pydantic validation failed for type '{data_type}': {e}", exc_info=True)
            return None

    def _guess_file_type(self, file_path_or_url: str) -> Optional[str]:
        """Guesses file type based on extension."""
        try:
            parsed_path = urlparse(file_path_or_url).path
            extension = parsed_path.split('.')[-1].lower() if '.' in parsed_path else ''

            if extension == "pdf":
                return "pdf"
            elif extension == "txt":
                return "txt"
            elif extension == "json":
                return "json"
            elif extension == "epub": # <<< ADDED EPUB
                return "epub"
            # Add more types as needed (docx, etc.)
            else:
                logging.warning(f"Could not guess file type from extension '{extension}' for {file_path_or_url}")
                return None
        except Exception as e:
            logging.error(f"Error guessing file type for {file_path_or_url}: {e}")
            return None

    def _read_epub_content(self, file_bytes: BytesIO) -> Optional[str]:
        """Reads and extracts text content from EPUB file bytes."""
        if not EBOOKLIB_AVAILABLE:
            logging.error("EbookLib is not available. Cannot parse EPUB.")
            return None
        try:
            book = epub.read_epub(file_bytes)
            text_content = []
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT: # ITEM_DOCUMENT for HTML content
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    # Remove script and style elements
                    for script_or_style in soup(["script", "style"]):
                        script_or_style.decompose()
                    # Get text
                    page_text = soup.get_text(separator='\n', strip=True)
                    if page_text:
                        text_content.append(page_text)
            
            full_text = "\n\n---\n\n".join(text_content) # Join chapters/sections with a separator
            logging.info(f"Extracted {len(full_text)} characters from EPUB.")
            return full_text.strip() if full_text else None
        except Exception as e:
            logging.error(f"Error reading EPUB content: {e}", exc_info=True)
            return None


    def _read_file_content(self, file_path_or_url: str, file_type: str) -> Optional[str]:
        """Reads content from a local file or URL."""
        content = None
        is_url = file_path_or_url.startswith("http://") or file_path_or_url.startswith("https://")
        logging.debug(f"Reading {'URL' if is_url else 'local file'}: {file_path_or_url} (type: {file_type})")

        try:
            if is_url:
                headers = {'User-Agent': 'AgenticLLMSystem/1.0'}
                response = requests.get(file_path_or_url, stream=True, headers=headers, timeout=30)
                response.raise_for_status()
                
                file_bytes = BytesIO(response.content) # Read content into BytesIO for uniform handling

                if file_type == "pdf":
                    content_type = response.headers.get('Content-Type', '').lower()
                    if 'application/pdf' not in content_type:
                        logging.warning(f"URL content-type is '{content_type}', not 'application/pdf'. PDF Parsing might fail.")
                    # PyPDF2 needs bytes
                    pdf_reader = PyPDF2.PdfReader(file_bytes)
                    text_pages = [page.extract_text() for page in pdf_reader.pages if page.extract_text()]
                    content = "\n".join(text_pages).strip()
                elif file_type == "epub": # <<< ADDED EPUB HANDLING FOR URL
                    if not EBOOKLIB_AVAILABLE: return None
                    content = self._read_epub_content(file_bytes)
                elif file_type in ["txt", "json"]: # For text-based files from URL
                    # Let requests handle decoding based on headers, fallback to utf-8
                    # For BytesIO, we need to decode it ourselves
                    try:
                        content = file_bytes.getvalue().decode(response.apparent_encoding or 'utf-8')
                    except UnicodeDecodeError:
                        logging.warning(f"UTF-8/Apparent encoding failed for URL {file_path_or_url}, trying latin-1.")
                        content = file_bytes.getvalue().decode('latin-1')
                else:
                    logging.warning(f"URL reading for file type '{file_type}' not fully implemented for direct text extraction, attempting as binary.")
                    # For other types, you might just return raw bytes or handle specifically
                    # For RAG, we generally need text. If it's not text, what to do?
                    # For now, we'll assume if it's not PDF/EPUB/TXT/JSON, it's not directly processable into text here.
                    return None # Or try to decode as plain text as a last resort
            
            else: # Local file
                if file_type == "pdf":
                    logging.debug("Reading PDF content from local file...")
                    with open(file_path_or_url, 'rb') as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        text_pages = [page.extract_text() for page in pdf_reader.pages if page.extract_text()]
                        content = "\n".join(text_pages).strip()
                elif file_type == "epub": # <<< ADDED EPUB HANDLING FOR LOCAL FILE
                    if not EBOOKLIB_AVAILABLE: return None
                    logging.debug("Reading EPUB content from local file...")
                    with open(file_path_or_url, 'rb') as f:
                        file_bytes = BytesIO(f.read())
                    content = self._read_epub_content(file_bytes)
                elif file_type in ["txt", "json"]:
                    logging.debug(f"Reading {file_type} content from local file...")
                    try:
                        with open(file_path_or_url, 'r', encoding='utf-8') as f:
                            content = f.read()
                    except UnicodeDecodeError:
                        logging.warning(f"UTF-8 decoding failed for {file_path_or_url}, trying latin-1.")
                        with open(file_path_or_url, 'r', encoding='latin-1') as f:
                            content = f.read()
                else:
                    logging.warning(f"Local file reading for type '{file_type}' not implemented.")
                    return None

        except ImportError as ie: # Specifically for PyPDF2 or Ebooklib if not installed
            logging.error(f"Import error for parsing {file_type}: {ie}. Ensure necessary libraries are installed.")
            return None
        except FileNotFoundError:
            logging.error(f"Local file not found at {file_path_or_url}")
            return None
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching URL {file_path_or_url}: {e}")
            return None
        except PyPDF2.errors.PdfReadError as e: # Catch specific PyPDF2 errors
            logging.error(f"Error reading PDF file {file_path_or_url} (possibly corrupted or password-protected): {e}")
            return None
        except Exception as e: # General catch-all
            logging.exception(f"Error reading file {file_path_or_url} (type: {file_type}): {e}")
            return None

        return content if content and content.strip() else None
