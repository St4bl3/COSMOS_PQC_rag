# --- File: ingestion/models.py ---
from pydantic import BaseModel, Field, FilePath, HttpUrl
from typing import Optional, Dict, Any, Union, List
from datetime import datetime # Import datetime

# --- Input Data Models (using Pydantic) ---

class BaseInputData(BaseModel):
    """Base model for common input fields."""
    source_id: str = Field(..., description="Unique identifier for the data source or event")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now, description="Time data was received/processed")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Any additional metadata")

class DisasterWeatherData(BaseModel):
    """Model for Natural Disaster/Weather prediction data."""
    event_type: str = Field(..., description="Type of event (e.g., Hurricane, Earthquake, Flood, Wildfire)")
    location: str = Field(..., description="Geographic location or area affected")
    severity: Optional[str] = Field(None, description="Predicted severity (e.g., Category 4, Magnitude 6.5)")
    predicted_impact: Optional[str] = Field(None, description="Text description of predicted impact")
    confidence_score: Optional[float] = Field(None, ge=0, le=1, description="Confidence in the prediction")
    raw_inference_data: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Raw data from prediction model")

class SpaceDebrisData(BaseModel):
    """Model for Asteroid/Space Debris information."""
    object_id: str = Field(..., description="Identifier for the debris/asteroid")
    trajectory: List[Dict[str, Any]] = Field(..., description="List of predicted positions/velocities over time")
    size_estimate_m: Optional[float] = Field(None, gt=0, description="Estimated diameter in meters")
    collision_risk_assessment: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Assessment of collision risk (e.g., probability, target)")
    raw_inference_data: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Raw data from tracking model")

class TextInputData(BaseModel):
    """Model for simple text input."""
    text_content: str = Field(..., description="The raw text input")

class FileInputData(BaseModel):
    """Model for file-based input (e.g., PDF)."""
    # In a real API, this would likely be handled via file uploads.
    # For internal processing, we might just pass the path.
    file_path: Union[FilePath, HttpUrl] = Field(..., description="Path or URL to the input file")
    file_type: str = Field(..., description="Type of file (e.g., 'pdf', 'txt')")


# Combined Input Model for Orchestrator
class OrchestratorInput(BaseInputData):
    """Input model for the Central Orchestration Agent."""
    data_type: str = Field(..., description="Indicator of the data type ('disaster', 'debris', 'text', 'file')")
    # Use Union to allow different data types based on 'data_type'
    # Pydantic v2 supports discriminated unions more elegantly, but this works for v1
    data: Union[DisasterWeatherData, SpaceDebrisData, TextInputData, FileInputData] = Field(..., description="The actual data payload")
    task_id: Optional[str] = Field(None, description="Optional existing task ID to continue or reference")
