import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from unittest.mock import patch, MagicMock
import os
import logging

from src.deployment.model_deployer import ModelDeployer
from src.deployment.prediction_service import PredictionService
from src.deployment.pipeline import DataPipeline

# Constants for test data
SAMPLE_DATA_PATH = "data/raw/load_data_20250316_20250326.csv"

@pytest.fixture(autouse=True)
def setup_logging():
    """Set up logging for tests."""
    logging.basicConfig(level=logging.INFO)
    yield

def create_mock_load_data(start_time, end_time):
    """Create mock load data with proper columns and format"""
    dates = pd.date_range(
        start=start_time,
        end=end_time,
        freq='15min',
        tz='Europe/Berlin'
    )
    
    # Create base load with daily and weekly patterns
    hourly_pattern = np.sin(np.pi * dates.hour / 24) + 1
    weekly_pattern = np.sin(np.pi * dates.dayofweek / 7) * 0.3
    
    base_load = 50000 + 10000 * hourly_pattern + 5000 * weekly_pattern
    noise = np.random.normal(0, 1000, size=len(dates))
    
    # Create DataFrame with only required columns
    data = pd.DataFrame({
        'Actual Load': base_load + noise,
        'Forecasted Load': base_load + np.random.normal(0, 2000, size=len(dates))
    }, index=dates)
    
    return data

@pytest.fixture
def sample_real_data():
    """Fixture to load real sample data"""
    if not os.path.exists(SAMPLE_DATA_PATH):
        pytest.skip(f"Sample data file not found: {SAMPLE_DATA_PATH}")
    return pd.read_csv(SAMPLE_DATA_PATH, index_col=0, parse_dates=True)

@pytest.fixture
def mock_entsoe_client():
    """Mock ENTSOE client that returns synthetic data"""
    with patch('src.data.clients.entsoe_client.EntsoeClient') as mock:
        client = MagicMock()
        
        def get_load_data(start_time, end_time):
            data = create_mock_load_data(start_time, end_time)
            # Ensure timezone awareness
            if data.index.tzinfo is None:
                data.index = data.index.tz_localize('Europe/Berlin')
            return data
            
        def get_load_forecast(start_time, end_time):
            data = create_mock_load_data(start_time, end_time)
            # Only return forecasted load for future dates
            data = data[['Forecasted Load']]
            if data.index.tzinfo is None:
                data.index = data.index.tz_localize('Europe/Berlin')
            return data
            
        client.get_load_data.side_effect = get_load_data
        client.get_load_forecast.side_effect = get_load_forecast
        mock.return_value = client
        yield mock

@pytest.fixture
def initialized_pipeline(mock_entsoe_client):
    """Fixture providing a fully initialized pipeline with historical data"""
    pipeline = DataPipeline()
    # Initialize with 7 days of historical data
    end_time = pd.Timestamp.now(tz='Europe/Berlin')
    start_time = end_time - pd.Timedelta(days=7)
    historical_data = create_mock_load_data(start_time, end_time)
    pipeline.initialize_with_historical_data(historical_data)
    return pipeline

def test_end_to_end_prediction_mock(mock_entsoe_client):
    """Test prediction pipeline with mock data"""
    # Create historical data with extra days for feature extraction
    end_time = pd.Timestamp.now(tz='Europe/Berlin')
    start_time = end_time - pd.Timedelta(days=10)  # Extra days for feature extraction
    historical_data = create_mock_load_data(start_time, end_time)
    
    # Initialize deployer
    deployer = ModelDeployer()
    success = deployer.initialize_pipeline(historical_data)
    assert success, "Pipeline initialization failed"
    deployer.load_model()
    
    # Make prediction for a future timestamp
    future_time = end_time + pd.Timedelta(hours=1)
    prediction = deployer.make_prediction(future_time)
    
    assert prediction is not None
    assert isinstance(prediction, pd.Series)
    assert len(prediction) == 96  # 24 hours of 15-minute intervals
    assert prediction.min() >= 30000  # Reasonable load values
    assert prediction.max() <= 70000

def test_end_to_end_prediction_real(sample_real_data):
    """Test prediction pipeline with real sample data"""
    deployer = ModelDeployer()
    success = deployer.initialize_pipeline(sample_real_data)
    assert success, "Pipeline initialization failed"
    deployer.load_model()
    
    # Use a timestamp that ensures we have enough historical data for features
    test_time = sample_real_data.index[len(sample_real_data)//2]
    prediction = deployer.make_prediction(test_time)
    
    assert prediction is not None
    assert isinstance(prediction, pd.Series)
    assert len(prediction) == 96
    assert prediction.min() >= 30000
    assert prediction.max() <= 70000

def test_forecast_generation(mock_entsoe_client):
    """Test forecast generation with mock data"""
    # Create historical data first
    end_time = pd.Timestamp.now(tz='Europe/Berlin')
    start_time = end_time - pd.Timedelta(days=7)
    historical_data = create_mock_load_data(start_time, end_time)
    
    service = PredictionService()
    # Initialize with historical data
    service.deployer.initialize_pipeline(historical_data)
    service.deployer.load_model()
    service.initialized = True
    
    forecast = service.get_forecast(hours=24)
    
    assert forecast is not None
    assert isinstance(forecast, pd.Series)
    assert len(forecast) == 96  # 24 hours * 4 (15-min intervals)
    assert not forecast.isnull().any()

def test_cache_functionality(mock_entsoe_client):
    """Test prediction caching with mock data"""
    # Create historical data first
    end_time = pd.Timestamp.now(tz='Europe/Berlin')
    start_time = end_time - pd.Timedelta(days=7)
    historical_data = create_mock_load_data(start_time, end_time)
    
    service = PredictionService()
    # Initialize with historical data
    service.deployer.initialize_pipeline(historical_data)
    service.deployer.load_model()
    service.initialized = True
    
    # Get initial forecast
    initial_forecast = service.get_forecast(hours=24)
    
    # Get another forecast immediately (should use cache)
    cached_forecast = service.get_forecast(hours=24)
    
    # Compare only the values, not the index
    np.testing.assert_array_almost_equal(
        initial_forecast.values,
        cached_forecast.values
    )