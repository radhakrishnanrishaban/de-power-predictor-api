import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
import pandas as pd
import pytz

from src.api import app
from src.data.clients.entsoe_client import EntsoeClient
from src.data.preprocess.preprocessor import LoadDataPreprocessor

client = TestClient(app)

@pytest.fixture
def mock_forecast_data():
    """Create sample forecast data for testing"""
    tz = pytz.timezone('Europe/Berlin')
    dates = pd.date_range(
        start=datetime.now(tz),
        end=datetime.now(tz) + timedelta(days=2),
        freq='15min'
    )
    return pd.DataFrame({
        'Load Forecast': range(len(dates))
    }, index=dates)

@pytest.fixture
def mock_preprocessor(mocker):
    """Mock preprocessor to return same data"""
    def mock_preprocess(data):
        return data
    
    mocker.patch.object(
        LoadDataPreprocessor,
        'preprocess',
        side_effect=mock_preprocess
    )

def test_get_latest_forecast(mocker, mock_forecast_data, mock_preprocessor):
    """Test latest forecast endpoint"""
    mocker.patch.object(
        EntsoeClient,
        'get_load_forecast',
        return_value=mock_forecast_data
    )
    
    response = client.get("/api/v1/forecast/latest")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "timestamp" in data
    assert "data" in data
    assert "metadata" in data
    assert data["metadata"]["points"] == len(mock_forecast_data)

def test_get_forecast_range(mocker, mock_forecast_data, mock_preprocessor):
    """Test forecast range endpoint"""
    mocker.patch.object(
        EntsoeClient,
        'get_load_forecast',
        return_value=mock_forecast_data
    )
    
    response = client.get("/api/v1/forecast/range/48")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["metadata"]["hours_requested"] == 48
    assert data["metadata"]["points"] == len(mock_forecast_data)

def test_forecast_error_handling(mocker):
    """Test forecast error handling"""
    mocker.patch.object(
        EntsoeClient,
        'get_load_forecast',
        side_effect=Exception("Forecast error")
    )
    
    response = client.get("/api/v1/forecast/latest")
    assert response.status_code == 500
    assert "Forecast error" in response.json()["detail"]

def test_forecast_range_invalid_hours(mocker):
    """Test forecast range with invalid hours parameter"""
    response = client.get("/api/v1/forecast/range/-1")
    assert response.status_code == 400
    assert "Invalid hours parameter" in response.json()["detail"]

    response = client.get("/api/v1/forecast/range/0")
    assert response.status_code == 400
    assert "Invalid hours parameter" in response.json()["detail"]

def test_get_forecast_empty_data(mocker):
    """Test forecast endpoint with empty data"""
    empty_df = pd.DataFrame(columns=['Load Forecast'])
    
    mocker.patch.object(
        EntsoeClient,
        'get_load_forecast',
        return_value=empty_df
    )
    
    response = client.get("/api/v1/forecast/latest")
    assert response.status_code == 404
    assert "No forecast data available" in response.json()["detail"] 