import pytest
from fastapi.testclient import TestClient
import pandas as pd
from datetime import datetime, timedelta
import pytz
import logging

from src.api import app

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_prediction_service():
    """Initialize prediction service before running tests"""
    from src.deployment.prediction_service import PredictionService
    service = PredictionService()
    service.initialize()
    return service

def test_health_check():
    """Test the health check endpoint"""
    logger.debug("Testing health check endpoint")
    response = client.get("/api/v1/health")
    logger.debug(f"Response status: {response.status_code}")
    logger.debug(f"Response body: {response.text}")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_get_forecast():
    """Test the forecast endpoint"""
    logger.debug("Testing forecast endpoint")
    response = client.get("/api/v1/forecast")
    logger.debug(f"Response status: {response.status_code}")
    logger.debug(f"Response body: {response.text}")
    assert response.status_code == 200
    
    data = response.json()
    assert "forecasts" in data
    assert "timestamp" in data
    
    forecasts = pd.DataFrame(data["forecasts"])
    assert len(forecasts) == 96  # 24 hours * 4 (15-min intervals)
    assert all(30000 <= value <= 80000 for value in forecasts["load"])
    
    # Convert forecasts back to DataFrame for validation
    forecasts = pd.DataFrame(data["forecasts"])
    
    # Validate forecast properties
    assert len(forecasts) == 96  # 24 hours * 4 (15-min intervals)
    assert all(30000 <= value <= 80000 for value in forecasts["load"])  # Reasonable load values
    
    # Validate timestamps are in correct format and timezone
    timestamps = pd.to_datetime(forecasts["timestamp"])
    assert all(ts.tzinfo is not None for ts in timestamps)  # All timestamps should have timezone
    assert timestamps.is_monotonic_increasing  # Timestamps should be in order
    
    # Validate time intervals
    time_diffs = timestamps.diff()[1:]  # Skip first diff which is NaT
    assert all(td == timedelta(minutes=15) for td in time_diffs)

def test_get_forecast_with_hours():
    """Test the forecast endpoint with custom hours parameter"""
    hours = 48
    response = client.get(f"/api/v1/forecast?hours={hours}")
    assert response.status_code == 200
    
    data = response.json()
    forecasts = pd.DataFrame(data["forecasts"])
    assert len(forecasts) == hours * 4  # hours * 4 (15-min intervals)

def test_get_historical():
    """Test the historical data endpoint"""
    response = client.get("/api/v1/historical")
    assert response.status_code == 200
    
    data = response.json()
    assert "historical_data" in data
    assert "timestamp" in data
    
    historical = pd.DataFrame(data["historical_data"])
    
    # Validate historical data properties
    assert len(historical) > 0
    assert all(30000 <= value <= 80000 for value in historical["actual_load"])
    
    # Validate timestamps
    timestamps = pd.to_datetime(historical["timestamp"])
    assert all(ts.tzinfo is not None for ts in timestamps)
    assert timestamps.is_monotonic_increasing

def test_invalid_hours():
    """Test the forecast endpoint with invalid hours parameter"""
    # Test for hours > 168 (7 days)
    response = client.get("/api/v1/forecast?hours=169")
    assert response.status_code == 422  # FastAPI validation error
    assert "less than or equal to 168" in response.json()["detail"][0]["msg"].lower()
    
    # Test for hours = 0
    response = client.get("/api/v1/forecast?hours=0")
    assert response.status_code == 422
    assert "greater than or equal to 1" in response.json()["detail"][0]["msg"].lower()
    
    # Test for negative hours
    response = client.get("/api/v1/forecast?hours=-1")
    assert response.status_code == 422
    assert "greater than or equal to 1" in response.json()["detail"][0]["msg"].lower()

def test_get_metrics():
    """Test the metrics endpoint"""
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    
    data = response.json()
    assert "prediction_accuracy" in data
    assert "last_update" in data
    assert "model_version" in data
    
    # Check prediction accuracy structure
    accuracy = data["prediction_accuracy"]
    if accuracy is not None:  # If metrics are available
        assert "mean_prediction" in accuracy
        assert "std_prediction" in accuracy
        assert "avg_prediction_time_ms" in accuracy
        assert "cache_hit_rate" in accuracy
        assert "error_count" in accuracy

if __name__ == "__main__":
    pytest.main(["-v"]) 