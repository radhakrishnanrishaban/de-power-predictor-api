from pydantic import BaseModel, Field, field_validator, ConfigDict, model_validator
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
import math
import pandas as pd
from pydantic.json_schema import JsonSchemaValue

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

class LoadMetadata(BaseModel):
    points: int
    start_time: datetime
    end_time: datetime
    days_requested: Optional[int] = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        ser_json_timedelta='iso8601',
        ser_json_bytes='base64',
        validate_assignment=True
    )

class LoadResponse(BaseModel):
    timestamp: datetime
    data: List[Dict[str, Optional[float]]]
    metadata: LoadMetadata

    @model_validator(mode='before')
    @classmethod
    def validate_data(cls, values):
        if isinstance(values, dict) and 'data' in values:
            if isinstance(values['data'], pd.DataFrame):
                # Convert DataFrame to records and handle special values
                df = values['data'].copy()
                for col in df.select_dtypes(include=['float64', 'float32']).columns:
                    df[col] = df[col].where(~(pd.isna(df[col]) | np.isinf(df[col])), None)
                    df[col] = df[col].apply(lambda x: float(x) if x is not None else None)
                values['data'] = df.to_dict(orient='records')
            elif isinstance(values['data'], list):
                # Clean dictionary values
                values['data'] = [
                    {k: (None if isinstance(v, (float, int)) and (pd.isna(v) or math.isinf(v)) else float(v))
                     for k, v in item.items()}
                    for item in values['data']
                ]
        return values

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        ser_json_timedelta='iso8601',
        ser_json_bytes='base64',
        validate_assignment=True
    )

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