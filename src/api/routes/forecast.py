from fastapi import APIRouter, HTTPException
from src.api.models.schemas import ForecastResponse
from datetime import datetime, timedelta

router = APIRouter(prefix="/forecast", tags=["forecast"])

@router.get("/", response_model=ForecastResponse)
async def get_forecast(hours: int = 24):
    try:
        # Your forecasting logic here
        return ForecastResponse(
            timestamps=[datetime.now() + timedelta(hours=i) for i in range(hours)],
            values=[0.0] * hours,  # Replace with actual forecast
            model_version="0.1"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))