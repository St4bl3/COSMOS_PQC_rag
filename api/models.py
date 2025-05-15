from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Union # **Import Union here**
from datetime import datetime # Datetime needed for some fields

# --- API Request/Response Models (using Pydantic) ---

class IngestRequest(BaseModel):
    """Request model for the data ingestion endpoint."""
    source_id: Optional[str] = Field(None, description="Optional source ID, will be generated if missing")
    data_format: str = Field(default='json', description="Format of the data ('json', 'text', 'file', 'file_url')")
    # Using Union[Dict[str, Any], str] for flexibility, parsing happens in the endpoint/parser
    data: Union[Dict[str, Any], str] = Field(..., description="The input data (JSON object, raw text, file path for 'file', or URL for 'file_url')")
    task_id: Optional[str] = Field(None, description="Optional existing task ID to append to or reference")

class IngestResponse(BaseModel):
    """Response model for the data ingestion endpoint."""
    task_id: str = Field(..., description="The unique ID assigned to this processing task")
    status: str = Field(..., description="Initial status (e.g., 'received', 'processing_started')")
    message: str = Field(..., description="A message indicating the outcome of the ingestion request")

class TaskStatusResponse(BaseModel):
    """Response model for checking task status."""
    task_id: str
    status: str
    results: Optional[Dict[str, Any]] = Field(None, description="Generated results if available (SOPs, commands, etc.)")
    history: Optional[List[Dict[str, Any]]] = Field(None, description="Log of steps taken for the task")
    human_feedback_received: Optional[Dict[str, Any]] = Field(None, description="Feedback already submitted")
    timestamp: Optional[datetime] = Field(None, description="Timestamp of the last update") # Added timestamp

class FeedbackRequest(BaseModel):
    """Request model for submitting human feedback."""
    feedback: Dict[str, Any] = Field(..., description="Feedback content (e.g., {'approved': true, 'comments': 'Looks good', 'rating': 5})")
    # Explicitly allow status update via feedback
    status_update: Optional[str] = Field(None, description="Optional new status for the task (e.g., 'approved', 'rejected', 'needs_revision')")

class FeedbackResponse(BaseModel):
    """Response model for submitting feedback."""
    task_id: str
    status: str # The status *after* applying feedback
    message: str

