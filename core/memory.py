# --- File: core/memory.py ---
import sqlite3
import json
from datetime import datetime, timezone # Import timezone
from typing import List, Dict, Any, Optional, Tuple
import config
from core.vector_db import VectorDBClient # For potential vector-based memory
import logging # Use logging

# --- Memory Management Module ---

class TaskMemory:
    """
    Manages task state, history, and human feedback.
    Can use different storage backends (SQLite, Vector DB, Hybrid).
    """
    def __init__(
        self,
        memory_type: str = config.MEMORY_DB_TYPE,
        vector_db_client: Optional[VectorDBClient] = None # Needed for vector/hybrid
    ):
        self.memory_type = memory_type
        self.vector_db = vector_db_client
        self.conn = None # For SQLite
        self._initialize_memory()
        logging.info(f"Task Memory initialized using type: {self.memory_type}")

    def _datetime_to_iso(self, dt: Optional[datetime]) -> Optional[str]:
        """Converts datetime object to ISO 8601 string, ensuring UTC if naive."""
        if dt is None:
            return None
        if dt.tzinfo is None: # If datetime is naive, assume UTC
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()

    def _iso_to_datetime(self, iso_str: Optional[str]) -> Optional[datetime]:
        """Converts ISO 8601 string back to datetime object."""
        if iso_str is None:
            return None
        try:
            return datetime.fromisoformat(iso_str)
        except (ValueError, TypeError):
            logging.warning(f"Could not parse ISO string to datetime: {iso_str}")
            return None # Or handle as an error

    def _preprocess_state_for_json(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure datetime objects in state are converted to ISO strings for JSON serialization."""
        processed_dict = state_dict.copy()
        if 'timestamp' in processed_dict and isinstance(processed_dict['timestamp'], datetime):
            processed_dict['timestamp'] = self._datetime_to_iso(processed_dict['timestamp'])
        
        # Handle timestamp within input_data if it's from OrchestratorInput
        if 'input_data' in processed_dict and isinstance(processed_dict['input_data'], dict):
            input_data_payload = processed_dict['input_data']
            if 'timestamp' in input_data_payload and isinstance(input_data_payload['timestamp'], datetime):
                input_data_payload['timestamp'] = self._datetime_to_iso(input_data_payload['timestamp'])
        
        # Handle timestamps within steps_log
        if 'steps_log' in processed_dict and isinstance(processed_dict['steps_log'], list):
            for step in processed_dict['steps_log']:
                if isinstance(step, dict) and 'timestamp' in step and isinstance(step['timestamp'], datetime):
                    step['timestamp'] = self._datetime_to_iso(step['timestamp'])
        return processed_dict

    def _postprocess_state_from_json(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Convert ISO strings back to datetime objects after loading from JSON."""
        if 'timestamp' in state_dict and isinstance(state_dict['timestamp'], str):
            state_dict['timestamp'] = self._iso_to_datetime(state_dict['timestamp'])
        
        if 'input_data' in state_dict and isinstance(state_dict['input_data'], dict):
            input_data_payload = state_dict['input_data']
            if 'timestamp' in input_data_payload and isinstance(input_data_payload['timestamp'], str):
                input_data_payload['timestamp'] = self._iso_to_datetime(input_data_payload['timestamp'])

        if 'steps_log' in state_dict and isinstance(state_dict['steps_log'], list):
            for step in state_dict['steps_log']:
                if isinstance(step, dict) and 'timestamp' in step and isinstance(step['timestamp'], str):
                    step['timestamp'] = self._iso_to_datetime(step['timestamp'])
        return state_dict


    def _initialize_memory(self):
        """Initializes the chosen memory storage."""
        if self.memory_type == "sqlite" or self.memory_type == "hybrid":
            self._initialize_memory_sqlite() 

        if self.memory_type == "vector" or self.memory_type == "hybrid":
            if not self.vector_db:
                logging.error("Vector DB client required for 'vector' or 'hybrid' memory type but not provided.")
                if self.memory_type == "vector":
                    self.memory_type = "none" 
            else:
                logging.info("Vector DB component for memory enabled.")
        
        if self.memory_type == "none":
            logging.warning("Memory system is not functional due to configuration errors.")

    def _initialize_memory_sqlite(self):
        """Helper to initialize SQLite component."""
        try:
            self.conn = sqlite3.connect(config.SQLITE_DB_PATH, check_same_thread=False) 
            cursor = self.conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS task_memory (
                    task_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL, -- Stored as ISO string
                    input_data TEXT,
                    agent_outputs TEXT, 
                    human_feedback TEXT, 
                    status TEXT, 
                    steps_log TEXT 
                )
            """)
            self.conn.commit()
            logging.info(f"SQLite memory database initialized at {config.SQLITE_DB_PATH}")
        except Exception as e:
            logging.error(f"Error initializing SQLite memory: {e}")
            self.conn = None
            if self.memory_type == "sqlite": self.memory_type = "none"


    def save_task_state(self, task_id: str, state: Dict[str, Any]):
        """Saves the current state of a task."""
        if self.memory_type == "none":
            logging.error("Memory system not initialized. Cannot save task state.")
            return False

        # Ensure timestamp is set and correctly formatted as string
        current_dt = state.get('timestamp')
        if isinstance(current_dt, datetime):
            timestamp_str = self._datetime_to_iso(current_dt)
        elif isinstance(current_dt, str):
            timestamp_str = current_dt # Assume already ISO string
        else: # If None or other type, generate new
            timestamp_str = self._datetime_to_iso(datetime.now(timezone.utc))
        
        state['timestamp'] = timestamp_str # Update state with string version

        # Preprocess the whole state for JSON serialization (handles nested datetimes)
        state_for_json = self._preprocess_state_for_json(state.copy()) # Use a copy

        saved_sqlite = False
        saved_vector = False

        if self.memory_type == "sqlite" or self.memory_type == "hybrid":
            if not self.conn:
                logging.error("SQLite connection not available for saving task state.")
            else:
                try:
                    cursor = self.conn.cursor()
                    cursor.execute("""
                        INSERT OR REPLACE INTO task_memory
                        (task_id, timestamp, input_data, agent_outputs, human_feedback, status, steps_log)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        task_id,
                        state_for_json.get('timestamp'), # Already string
                        json.dumps(state_for_json.get('input_data', {})),
                        json.dumps(state_for_json.get('agent_outputs', {})),
                        json.dumps(state_for_json.get('human_feedback', {})),
                        state_for_json.get('status', 'unknown'),
                        json.dumps(state_for_json.get('steps_log', []))
                    ))
                    self.conn.commit()
                    logging.info(f"Task state saved for task_id: {task_id} (SQLite)")
                    saved_sqlite = True
                except Exception as e:
                    logging.error(f"Error saving task state to SQLite for task_id {task_id}: {e}")
        
        if self.memory_type == "vector" or self.memory_type == "hybrid":
            if not self.vector_db:
                logging.error("Vector DB client not available for saving task state embedding.")
            else:
                # Pass the original state (with datetime objects if any, _save_task_embedding will handle them)
                # or state_for_json if _save_task_embedding expects strings.
                # For consistency, let _save_task_embedding also handle datetime conversion.
                if self._save_task_embedding(task_id, state): # Pass original state
                    saved_vector = True
        return saved_sqlite or saved_vector

    def _save_task_embedding(self, task_id: str, state: Dict[str, Any]) -> bool:
        """Helper to create and save task embedding to Vector DB."""
        if not self.vector_db or not self.vector_db.embedding_client:
            logging.error("Cannot save task embedding without Vector DB and Embedding Client.")
            return False

        text_representation = f"Task ID: {task_id}\nStatus: {state.get('status')}\n"
        input_summary = "N/A"
        input_data = state.get('input_data', {})
        if isinstance(input_data, dict):
            data_payload = input_data.get('data', {})
            if isinstance(data_payload, dict):
                input_summary = f"Type: {input_data.get('data_type')}, Event: {data_payload.get('event_type') or data_payload.get('object_id') or 'Unknown'}"
            else: 
                input_summary = str(data_payload)[:200] 
        else:
            input_summary = str(input_data)[:200] 

        output_summary = str(state.get('agent_outputs', {}))[:200] 
        feedback_summary = str(state.get('human_feedback', {}))[:200] 

        text_representation += f"Input Summary: {input_summary}\n"
        text_representation += f"Output Summary: {output_summary}\n"
        text_representation += f"Feedback: {feedback_summary}"
        
        logging.debug(f"Generating embedding for task state: {task_id}")
        embedding = self.vector_db.embedding_client.get_embedding(text_representation)
        if not embedding:
            logging.error(f"Failed to generate embedding for task {task_id}")
            return False

        # Metadata for vector DB should use stringified datetime
        metadata_timestamp = state.get('timestamp')
        if isinstance(metadata_timestamp, datetime):
            metadata_timestamp_str = self._datetime_to_iso(metadata_timestamp)
        elif isinstance(metadata_timestamp, str):
            metadata_timestamp_str = metadata_timestamp # Assume already ISO
        else:
            metadata_timestamp_str = self._datetime_to_iso(datetime.now(timezone.utc))


        metadata = {
            "task_id": task_id, 
            "timestamp": metadata_timestamp_str, # Use stringified timestamp
            "status": state.get('status', 'unknown'),
            "memory_type": "task_state", 
            "input_summary": input_summary, 
        }
        if isinstance(input_data, dict):
            data_payload = input_data.get('data', {})
            if isinstance(data_payload, dict):
                metadata["event_type"] = data_payload.get('event_type')
                metadata["object_id"] = data_payload.get('object_id')

        logging.debug(f"Upserting task state embedding for task_id: {task_id}")
        success = self.vector_db.upsert([(task_id, embedding, metadata)])
        if success:
            logging.info(f"Task state embedding saved for task_id: {task_id} (Vector DB)")
        else:
            logging.error(f"Error saving task state embedding for task_id: {task_id}")
        return success

    def get_task_state(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves the state of a specific task by ID (primarily from SQLite)."""
        if self.memory_type == "none":
            logging.error("Memory system not initialized. Cannot get task state.")
            return None

        if self.memory_type == "sqlite" or self.memory_type == "hybrid":
            if not self.conn:
                logging.error("SQLite connection not available for getting task state.")
                return None
            try:
                cursor = self.conn.cursor()
                cursor.execute("SELECT * FROM task_memory WHERE task_id = ?", (task_id,))
                row = cursor.fetchone()
                if row:
                    columns = [description[0] for description in cursor.description]
                    state = dict(zip(columns, row))
                    # Deserialize JSON fields and convert timestamps
                    try:
                        state['input_data'] = json.loads(state.get('input_data', '{}'))
                    except json.JSONDecodeError:
                        logging.warning(f"Could not decode input_data JSON for task {task_id}")
                        state['input_data'] = {}
                    try:
                        state['agent_outputs'] = json.loads(state.get('agent_outputs', '{}'))
                    except json.JSONDecodeError:
                        logging.warning(f"Could not decode agent_outputs JSON for task {task_id}")
                        state['agent_outputs'] = {}
                    try:
                        state['human_feedback'] = json.loads(state.get('human_feedback', '{}'))
                    except json.JSONDecodeError:
                        logging.warning(f"Could not decode human_feedback JSON for task {task_id}")
                        state['human_feedback'] = {}
                    try:
                        state['steps_log'] = json.loads(state.get('steps_log', '[]'))
                    except json.JSONDecodeError:
                        logging.warning(f"Could not decode steps_log JSON for task {task_id}")
                        state['steps_log'] = []
                    
                    # Convert ISO strings back to datetime objects
                    state = self._postprocess_state_from_json(state)
                    logging.debug(f"Retrieved task state for {task_id} from SQLite.")
                    return state
                else:
                    logging.info(f"Task state for {task_id} not found in SQLite.")
                    return None
            except Exception as e:
                logging.error(f"Error getting task state from SQLite for task_id {task_id}: {e}")
                return None
        elif self.memory_type == "vector":
            logging.warning("Getting full task state by ID is not supported efficiently with vector-only memory.")
            return None 
        else: 
            logging.error("Memory type is invalid or uninitialized.")
            return None

    def search_related_tasks(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Searches for tasks semantically related to the query using Vector DB."""
        if self.memory_type == "none" or self.memory_type == "sqlite":
            logging.warning("Semantic task search requires 'vector' or 'hybrid' memory configuration.")
            return [] 

        if not self.vector_db:
            logging.error("Vector DB client required for semantic search but not available.")
            return []

        try:
            logging.info(f"Searching related tasks with query: '{query[:100]}...' (top_k={top_k})")
            mem_filter = {"memory_type": "task_state"}
            results = self.vector_db.query(query_text=query, top_k=top_k, filter=mem_filter)

            if not results:
                logging.info("No semantically related tasks found.")
                return []

            if self.memory_type == "hybrid" and self.conn:
                task_ids = [r['id'] for r in results if r.get('id')]
                logging.debug(f"Found related task IDs via vector search: {task_ids}. Retrieving full state from SQLite.")
                full_states = []
                retrieved_ids = set() 
                for task_id_from_vector in task_ids: # Renamed to avoid conflict
                    if task_id_from_vector not in retrieved_ids:
                        state = self.get_task_state(task_id_from_vector) 
                        if state:
                            score = next((r['score'] for r in results if r['id'] == task_id_from_vector), None)
                            state['_relevance_score'] = score
                            full_states.append(state)
                            retrieved_ids.add(task_id_from_vector)
                logging.info(f"Retrieved full state for {len(full_states)} related tasks.")
                return full_states
            else:
                logging.info(f"Returning metadata for {len(results)} related tasks (vector-only mode).")
                # Ensure metadata timestamps are also handled if needed by consumer
                processed_results = []
                for r in results:
                    metadata = r.get('metadata', {})
                    if 'timestamp' in metadata and isinstance(metadata['timestamp'], str):
                        metadata['timestamp'] = self._iso_to_datetime(metadata['timestamp'])
                    processed_results.append(metadata)
                return processed_results
        except Exception as e:
            logging.error(f"Error searching related tasks: {e}")
            return []

    def add_human_feedback(self, task_id: str, feedback: Dict[str, Any]) -> bool:
        """Adds human feedback to a specific task (updates SQLite primarily)."""
        if self.memory_type == "none":
            logging.error("Memory system not initialized. Cannot add feedback.")
            return False

        current_state = self.get_task_state(task_id)
        if not current_state:
            logging.error(f"Task {task_id} not found to add feedback.")
            return False

        current_feedback = current_state.get('human_feedback', {})
        current_feedback.update(feedback) 
        current_state['human_feedback'] = current_feedback

        new_status = feedback.get('status_update') 
        if new_status:
            current_state['status'] = new_status
        else:
            current_state['status'] = feedback.get('status', current_state.get('status', 'review'))

        logging.info(f"Adding feedback to task {task_id}. New status: {current_state['status']}")
        return self.save_task_state(task_id, current_state)

    def log_step(self, task_id: str, step_description: str, details: Optional[Dict] = None):
        """Logs a step in the task execution process (updates SQLite primarily)."""
        if self.memory_type == "none":
            logging.info(f"[Task Log {task_id} - No DB]: {step_description} {details or ''}")
            return

        current_state = self.get_task_state(task_id)
        if not current_state:
            logging.warning(f"Task {task_id} not found in memory to log step. Creating new entry.")
            current_state = {
                'input_data': {}, 'agent_outputs': {}, 'human_feedback': {},
                'status': 'processing', 'steps_log': [],
                'timestamp': datetime.now(timezone.utc) # Ensure new tasks also get a datetime object initially
            }
        
        # Ensure steps_log is a list
        steps_log = current_state.get('steps_log', [])
        if not isinstance(steps_log, list): # Handle case where it might be None or other type
            steps_log = []

        log_entry = {
            "timestamp": datetime.now(timezone.utc), # Use datetime object here
            "description": step_description,
            "details": details or {}
        }
        steps_log.append(log_entry)
        current_state['steps_log'] = steps_log

        logging.debug(f"Logging step for task {task_id}: {step_description}")
        self.save_task_state(task_id, current_state)

    def close(self):
        """Closes database connections."""
        if (self.memory_type == "sqlite" or self.memory_type == "hybrid") and self.conn:
            try:
                self.conn.close()
                logging.info("SQLite memory connection closed.")
                self.conn = None
            except Exception as e:
                logging.error(f"Error closing SQLite connection: {e}")