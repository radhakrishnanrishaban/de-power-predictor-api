from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
import pandas as pd

from src.data.clients.entsoe_client import EntsoeClient
from src.api.models.schemas import ForecastResponse  # If you have schemas defined

router = APIRouter()
client = EntsoeClient()

@router.get("/latest", response_model=ForecastResponse)
async def get_latest_forecast():
    """Get latest load forecast"""
    try:
        forecast_data = client.get_load_forecast()
        if forecast_data is None or forecast_data.empty:
            raise HTTPException(status_code=404, detail="No forecast data available")
            
        # Convert to DataFrame if Series
        if isinstance(forecast_data, pd.Series):
            forecast_data = forecast_data.to_frame('Load Forecast')
            
        return {
            "timestamp": datetime.now(),
            "data": forecast_data.to_dict(orient='records'),
            "metadata": {
                "points": len(forecast_data),
                "start_time": forecast_data.index.min(),
                "end_time": forecast_data.index.max(),
                "hours_requested": None
            }
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/range/{hours}", response_model=ForecastResponse)
async def get_forecast_range(hours: int):
    """Get load forecast for specified number of hours"""
    try:
        if hours <= 0:
            raise HTTPException(
                status_code=400,
                detail="Invalid hours parameter. Must be a positive integer."
            )
        
        end_time = datetime.now() + timedelta(hours=hours)
        forecast_data = client.get_load_forecast(end_time=end_time)
        
        if forecast_data is None or forecast_data.empty:
            raise HTTPException(status_code=404, detail="No forecast data available")
            
        return {
            "timestamp": datetime.now(),
            "data": forecast_data.to_dict(orient='records'),
            "metadata": {
                "points": len(forecast_data),
                "start_time": forecast_data.index.min(),
                "end_time": forecast_data.index.max(),
                "hours_requested": hours
            }
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))