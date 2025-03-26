from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime

class MetadataResponse(BaseModel):
    points: int
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    days_requested: Optional[int] = None
    frequency: Optional[str] = None

class LoadDataResponse(BaseModel):
    timestamp: str
    data: List[Dict[str, Any]]
    metadata: MetadataResponse

class ForecastMetadata(BaseModel):
    points: int
    start_time: datetime
    end_time: datetime
    hours_requested: Optional[int] = None

class ForecastResponse(BaseModel):
    timestamp: datetime
    data: List[Dict[str, float]]
    metadata: ForecastMetadata

class ErrorResponse(BaseModel):
    detail: str