from fastapi import APIRouter, HTTPException, Query
from datetime import datetime, timedelta
import pandas as pd
import pytz
import logging
from typing import Optional

from src.data.clients.entsoe_client import EntsoeClient
from src.api.models.schemas import ForecastResponse, ForecastMetadata
from src.deployment.model_deployer import ModelDeployer
from src.deployment.prediction_service import PredictionService

# Configure logger
logger = logging.getLogger(__name__)

router = APIRouter()
client = EntsoeClient()
prediction_service = PredictionService()

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
            
        return ForecastResponse(
            timestamp=datetime.now(pytz.UTC),
            data=forecast_data.to_dict(orient='records'),
            metadata=ForecastMetadata(
                points=len(forecast_data),
                start_time=forecast_data.index.min(),
                end_time=forecast_data.index.max(),
                hours_requested=None
            )
        )
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

@router.get("/forecast")
async def get_forecast(hours: Optional[int] = Query(24, ge=1, le=168, description="Number of hours to forecast (1-168)")):
    """Get load forecast for specified number of hours"""
    try:
        if not prediction_service.initialized:
            prediction_service.initialize()
        
        forecast = prediction_service.get_forecast(hours=hours)
        
        if forecast is None:
            raise HTTPException(status_code=500, detail="Failed to generate forecast")
        
        forecast_data = [
            {
                "timestamp": ts.isoformat(),
                "load": float(value)  # Ensure values are JSON serializable
            }
            for ts, value in forecast.items()
        ]
        
        return {
            "timestamp": datetime.now(pytz.UTC).isoformat(),
            "forecasts": forecast_data
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/historical")
async def get_historical():
    """Get historical load data"""
    try:
        if not prediction_service.initialized:
            prediction_service.initialize()
        
        historical_data = prediction_service.deployer.pipeline.get_historical_data()
        
        if historical_data is None or historical_data.empty:
            raise HTTPException(status_code=500, detail="No historical data available")
        
        historical_list = [
            {
                "timestamp": ts.isoformat(),
                "actual_load": float(row["Actual Load"]),
                "forecasted_load": float(row["Forecasted Load"]) if "Forecasted Load" in row else None
            }
            for ts, row in historical_data.iterrows()
        ]
        
        return {
            "timestamp": datetime.now(pytz.UTC).isoformat(),
            "historical_data": historical_list
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics")
async def get_metrics():
    """Get prediction metrics"""
    try:
        if not prediction_service.initialized:
            prediction_service.initialize()
        
        # Get metrics from the metrics collector
        metrics_collector = prediction_service.metrics_collector
        prediction_stats = metrics_collector.get_prediction_stats() if metrics_collector else {}
        
        return {
            "prediction_accuracy": {
                "mean_prediction": prediction_stats.get('mean_prediction', 0),
                "std_prediction": prediction_stats.get('std_prediction', 0),
                "avg_prediction_time_ms": prediction_stats.get('avg_prediction_time_ms', 0),
                "cache_hit_rate": prediction_stats.get('cache_hit_rate', 0),
                "error_count": prediction_stats.get('error_count', 0)
            },
            "last_update": datetime.now(pytz.UTC).isoformat(),
            "model_version": "1.0.0"  # You can get this from your settings or model metadata
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        return {
            "prediction_accuracy": None,
            "last_update": datetime.now(pytz.UTC).isoformat(),
            "model_version": "unknown",
            "error": str(e)
        }

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@router.get("/debug/routes")
async def list_routes():
    """List all available routes (for debugging)"""
    routes = []
    for route in router.routes:
        routes.append({
            "path": route.path,
            "name": route.name,
            "methods": route.methods
        })
    return {"routes": routes}