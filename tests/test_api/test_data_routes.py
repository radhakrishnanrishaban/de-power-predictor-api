import pytest
from fastapi.testclient import TestClient
from datetime import datetime
import pandas as pd
import pytz

from src.api import app  # Updated import
from src.data.clients.entsoe_client import EntsoeClient
from src.data.preprocess.preprocessor import LoadDataPreprocessor

client = TestClient(app)

@pytest.fixture
def mock_load_data():
    """Create sample load data for testing"""
    tz = pytz.timezone('Europe/Berlin')
    dates = pd.date_range(
        start=datetime.now(tz) - pd.Timedelta(days=1),
        end=datetime.now(tz),
        freq='15min'
    )
    return pd.DataFrame({
        'Actual Load': range(len(dates))
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

def test_get_latest_load(mocker, mock_load_data, mock_preprocessor):
    """Test latest load endpoint"""
    mocker.patch.object(
        EntsoeClient,
        'get_latest_load',
        return_value=mock_load_data
    )
    
    response = client.get("/api/v1/data/load/latest")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "timestamp" in data
    assert "data" in data
    assert "metadata" in data
    assert data["metadata"]["points"] == len(mock_load_data)

def test_get_historical_load(mocker, mock_load_data, mock_preprocessor):
    """Test historical load endpoint"""
    mocker.patch.object(
        EntsoeClient,
        'fetch_load_data',
        return_value=mock_load_data
    )
    
    response = client.get("/api/v1/data/load/historical/7")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["metadata"]["days_requested"] == 7
    assert data["metadata"]["points"] == len(mock_load_data)

def test_error_handling(mocker):
    """Test error handling"""
    mocker.patch.object(
        EntsoeClient,
        'get_latest_load',
        side_effect=Exception("Test error")
    )
    
    response = client.get("/api/v1/data/load/latest")
    assert response.status_code == 500
    assert "Test error" in response.json()["detail"]

def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_get_historical_load_invalid_days():
    """Test historical load endpoint with invalid days parameter"""
    # Test negative days
    response = client.get("/api/v1/data/load/historical/-1")
    assert response.status_code == 400
    assert "Invalid days parameter" in response.json()["detail"]

    # Test zero days
    response = client.get("/api/v1/data/load/historical/0")
    assert response.status_code == 400
    assert "Invalid days parameter" in response.json()["detail"]

def test_get_historical_load_error(mocker):
    """Test historical load endpoint error handling"""
    mocker.patch.object(
        EntsoeClient,
        'fetch_load_data',
        side_effect=Exception("Historical data fetch error")
    )
    
    response = client.get("/api/v1/data/load/historical/7")
    assert response.status_code == 500
    assert "Historical data fetch error" in response.json()["detail"]

def test_get_latest_load_empty_data(mocker):
    """Test latest load endpoint with empty data"""
    # Create an empty DataFrame with the correct column
    empty_df = pd.DataFrame(columns=['Actual Load'])
    
    # Mock the client to return empty DataFrame
    mocker.patch.object(
        EntsoeClient,
        'get_latest_load',
        return_value=empty_df
    )
    
    response = client.get("/api/v1/data/load/latest")
    assert response.status_code == 404
    assert "No data available" in response.json()["detail"]