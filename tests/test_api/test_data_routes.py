import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
import pandas as pd
import pytz
import numpy as np

from src.api import app  # Updated import
from src.data.clients.entsoe_client import EntsoeClient
from src.data.preprocess.preprocessor import LoadDataPreprocessor

client = TestClient(app)

@pytest.fixture
def mock_load_data():
    """Create mock load data with edge cases"""
    tz = pytz.timezone('Europe/Berlin')
    dates = pd.date_range(
        start=datetime.now(tz) - timedelta(days=1),
        periods=96,
        freq='15min',
        tz='Europe/Berlin'
    )
    
    data = {
        'Actual Load': [50000.0] * 96,
        'Forecasted Load': [51000.0] * 96
    }
    
    df = pd.DataFrame(data, index=dates)
    
    # Add edge cases
    df.iloc[0, 0] = np.nan  # NaN value
    df.iloc[1, 0] = np.inf  # Infinity
    df.iloc[2, 0] = -np.inf  # Negative infinity
    
    return df

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
    # Mock both the client and data manager
    mocker.patch('src.api.routes.data.data_manager.get_latest_load', 
                 return_value=mock_load_data)
    
    response = client.get("/api/v1/data/load/latest")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "timestamp" in data
    assert "data" in data
    assert "metadata" in data
    assert data["metadata"]["points"] == len(mock_load_data)

def test_get_historical_load(mocker, mock_load_data, mock_preprocessor):
    """Test historical load endpoint"""
    # Mock both the client and data manager
    mocker.patch('src.api.routes.data.data_manager.get_load_data', 
                 return_value=mock_load_data)
    
    response = client.get("/api/v1/data/load/historical/7")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["metadata"]["days_requested"] == 7
    assert data["metadata"]["points"] == len(mock_load_data)

def test_error_handling(mocker):
    """Test error handling"""
    # Mock data_manager.get_latest_load instead of EntsoeClient
    mocker.patch(
        'src.api.routes.data.data_manager.get_latest_load',
        side_effect=Exception("Test error")
    )
    
    response = client.get("/api/v1/data/load/latest")
    assert response.status_code == 500
    data = response.json()
    assert "Test error" in data["detail"]

def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_get_historical_load_invalid_days():
    """Test historical load endpoint with invalid days parameter"""
    # Test negative days
    response = client.get("/api/v1/data/load/historical/-1")
    assert response.status_code == 422
    error_detail = response.json()["detail"][0]
    assert error_detail["type"] == "greater_than_equal"  # FastAPI's actual error type
    assert error_detail["loc"] == ["path", "days"]
    assert "greater than or equal to" in error_detail["msg"].lower()

def test_get_historical_load_error(mocker):
    """Test historical load endpoint error handling"""
    # Mock data_manager.get_load_data instead of EntsoeClient
    mocker.patch(
        'src.api.routes.data.data_manager.get_load_data',
        side_effect=Exception("Historical data fetch error")
    )
    
    response = client.get("/api/v1/data/load/historical/7")
    assert response.status_code == 500
    data = response.json()
    assert "Historical data fetch error" in data["detail"]

def test_get_latest_load_empty_data(mocker):
    """Test latest load endpoint with empty data"""
    empty_df = pd.DataFrame(columns=['Actual Load'])
    
    # Mock data_manager.get_latest_load
    mocker.patch(
        'src.api.routes.data.data_manager.get_latest_load',
        return_value=empty_df
    )
    
    response = client.get("/api/v1/data/load/latest")
    assert response.status_code == 404
    assert "No data available" in response.json()["detail"]

def test_get_historical_load_with_cache(mocker):
    """Test historical load endpoint with cache enabled"""
    mock_data = pd.DataFrame({
        'Actual Load': [50000.0] * 96,
        'Forecasted Load': [51000.0] * 96
    })
    
    mocker.patch(
        'src.api.routes.data.data_manager.get_load_data',
        return_value=mock_data
    )
    
    # Test with cache enabled (default)
    response = client.get("/api/v1/data/load/historical/7")
    assert response.status_code == 200
    
    # Test with cache disabled
    response = client.get("/api/v1/data/load/historical/7?use_cache=false")
    assert response.status_code == 200

def test_get_historical_load_invalid_data(mocker):
    """Test historical load endpoint with invalid data"""
    # Test with None return value
    mocker.patch(
        'src.api.routes.data.data_manager.get_load_data',
        return_value=None
    )
    
    response = client.get("/api/v1/data/load/historical/7")
    assert response.status_code == 404
    assert "No historical data available" in response.json()["detail"]

def test_get_historical_load_max_days():
    """Test historical load endpoint with maximum days"""
    # Test with days > 30
    response = client.get("/api/v1/data/load/historical/31")
    assert response.status_code == 422
    error_detail = response.json()["detail"][0]
    assert error_detail["type"] == "less_than_equal"
    assert error_detail["loc"] == ["path", "days"]

def test_sanitize_dataframe_edge_cases():
    """Test sanitize_dataframe function with edge cases"""
    from src.api.routes.data import sanitize_dataframe
    import numpy as np

    # Create test data with various edge cases
    df = pd.DataFrame({
        'Actual Load': [
            np.nan,  # NaN
            np.inf,  # Infinity
            -np.inf,  # Negative infinity
            50000.0,  # Normal value
            None,    # None value
        ]
    })

    result = sanitize_dataframe(df)

    # Check results using pd.isna() for proper None/NaN comparison
    assert pd.isna(result['Actual Load'].iloc[0])  # NaN -> None
    assert pd.isna(result['Actual Load'].iloc[1])  # Inf -> None
    assert pd.isna(result['Actual Load'].iloc[2])  # -Inf -> None
    assert result['Actual Load'].iloc[3] == 50000.0  # Normal value unchanged
    assert pd.isna(result['Actual Load'].iloc[4])  # None -> None

    # Verify the values can be JSON serialized
    import json
    data_dict = result.to_dict(orient='records')
    try:
        json.dumps(data_dict)
        serializable = True
    except TypeError:
        serializable = False
    
    assert serializable, "Data should be JSON serializable"