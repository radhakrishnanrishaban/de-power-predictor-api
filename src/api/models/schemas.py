from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional

class LoadData(BaseModel):
    timestamp: datetime
    actual_load: Optional[float]
    forecast_load: Optional[float]

class ForecastResponse(BaseModel):
    timestamps: List[datetime]
    values: List[float]
    model_version: str