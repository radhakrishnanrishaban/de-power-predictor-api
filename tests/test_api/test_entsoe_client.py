import pytest
from datetime import datetime, timedelta
import pandas as pd
import pytz
from src.data.clients.entsoe_client import EntsoeClient
from src.config import Config
import requests
import numpy as np

@pytest.fixture
def entsoe_client():
    """Create an EntsoeClient instance for testing"""
    return EntsoeClient(
        api_key=Config.ENTSOE_API_KEY,
        country_code=Config.COUNTRY_CODE
    )

def test_fetch_load_data_future_timestamp(entsoe_client):
    """Fetching load data for 48 hours in the future"""
    future_start = datetime.now() + timedelta(hours=48)
    future_end = future_start + timedelta(days=1)
    
    data = entsoe_client.fetch_load_data(
        start_date=future_start.strftime('%Y%m%d'),
        end_date=future_end.strftime('%Y%m%d')
    )
    
    # Should return empty DataFrame with correct structure
    assert isinstance(data, pd.DataFrame)
    assert data.empty
    if not data.empty:  # Only check timezone if there's data
        assert data.index.tzinfo is not None

def test_fetch_load_data_specific_timestamp(entsoe_client):
    """Fetching load data for a specific timestamp should return known values"""
    # given
    specific_date = datetime.now() - timedelta(days=7)  # A week ago
    end_date = specific_date + timedelta(days=1)  # Next day
    
    # when
    fetched_df = entsoe_client.fetch_load_data(
        start_date=specific_date.strftime('%Y%m%d'),
        end_date=end_date.strftime('%Y%m%d')
    )
    
    # then
    assert isinstance(fetched_df, pd.DataFrame)
    if not fetched_df.empty:
        assert 'Actual Load' in fetched_df.columns
        assert fetched_df.index.is_monotonic_increasing
        assert fetched_df.index.is_unique
        assert not fetched_df['Actual Load'].isna().all()
        assert (fetched_df['Actual Load'] >= 0).all()  

def test_fetch_load_data_past_24h(entsoe_client, mocker):
    """Fetching load data for past 24 hours should return valid data"""
    # given
    start_time = datetime.now(entsoe_client.tz) - timedelta(hours=24)
    end_time = datetime.now(entsoe_client.tz)
    
    # Create mock data with proper hourly points
    index = pd.date_range(start=start_time, end=end_time, freq='h')
    mock_data = pd.DataFrame({
        'Actual Load': np.random.uniform(40000, 60000, size=len(index))
    }, index=index)
    
    # Mock the API call
    mocker.patch.object(
        entsoe_client.client,
        'query_load_and_forecast',
        return_value=mock_data
    )
    
    # Mock cache to return None
    mocker.patch(
        'src.data.utils.storage.get_cached_data',
        return_value=None
    )

    # when
    start_date = start_time.strftime('%Y%m%d')
    end_date = end_time.strftime('%Y%m%d')
    fetched_df = entsoe_client.fetch_load_data(
        start_date=start_date,
        end_date=end_date
    )

    # then
    # Data structure checks
    assert isinstance(fetched_df, pd.DataFrame)
    assert not fetched_df.empty
    assert 'Actual Load' in fetched_df.columns
    
    # Data quality checks
    assert len(fetched_df) >= 24  # At least 24 hourly data points
    assert fetched_df.index.is_monotonic_increasing
    assert fetched_df.index.is_unique
    
    # Type checks
    assert fetched_df['Actual Load'].dtype == 'float64'
    assert str(fetched_df.index.dtype).startswith(f'datetime64[ns, {Config.TIMEZONE}]')

def test_fetch_load_data_specific_timestamp(entsoe_client):
    """Fetching load data for a specific timestamp should return known values"""
    # given
    specific_date = "20240301"  # Use a known date
    # Set end date to the next day
    next_date = "20240302"
    # when
    fetched_df = entsoe_client.fetch_load_data(
        start_date=specific_date,
        end_date=next_date
    )
    
    # then
    # Data structure checks
    assert isinstance(fetched_df, pd.DataFrame)
    assert not fetched_df.empty
    assert 'Actual Load' in fetched_df.columns
    
    # Data quality checks
    assert fetched_df.index.is_monotonic_increasing
    assert fetched_df.index.is_unique
    assert not fetched_df['Actual Load'].isna().all()
    
    # Value range checks (adjust these based on your known data)
    assert (fetched_df['Actual Load'] >= 0).all()  # Load should be positive
    assert (fetched_df['Actual Load'] <= 100000).all()  # Reasonable upper limit

def test_get_load_forecast_next_24h(entsoe_client):
    """Getting load forecast for next 24 hours should return valid forecast data"""
    # given
    start_time = datetime.now(pytz.timezone(Config.TIMEZONE))
    end_time = start_time + timedelta(hours=24)
    
    # when
    forecast_df = entsoe_client.get_load_forecast(
        start_time=start_time,
        end_time=end_time
    )
    
    # then
    # Data structure checks
    assert isinstance(forecast_df, (pd.DataFrame, pd.Series))
    assert not forecast_df.empty
    
    # Time range checks
    if isinstance(forecast_df, pd.DataFrame):
        data_series = forecast_df.iloc[:, 0]
    else:
        data_series = forecast_df
        
    assert len(data_series) >= 24  # At least 24 hourly forecasts
    assert data_series.index.is_monotonic_increasing
    assert data_series.index.is_unique
    
    # Value checks
    assert not data_series.isna().all()
    assert (data_series >= 0).all()  # Load should be positive
    assert data_series.index.tzinfo is not None

def test_get_latest_load_data(entsoe_client):
    """Getting latest load data should return recent measurements"""
    # when
    latest_df = entsoe_client.get_latest_load()
    
    # then
    # Data structure checks
    assert isinstance(latest_df, pd.DataFrame)
    assert not latest_df.empty
    assert 'Actual Load' in latest_df.columns
    
    # Time checks
    latest_time = latest_df.index.max()
    current_time = datetime.now(latest_time.tzinfo)
    time_difference = current_time - latest_time
    assert time_difference < timedelta(hours=24)  # Data should be recent
    
    # Data quality checks
    assert latest_df.index.is_monotonic_increasing
    assert latest_df.index.is_unique
    assert latest_df['Actual Load'].dtype == 'float64'

def test_fetch_load_data_invalid_format(entsoe_client):
    """Test handling of invalid date formats"""
    with pytest.raises(ValueError, match="Dates must be in YYYYMMDD format"):
        entsoe_client.fetch_load_data(
            start_date="invalid",
            end_date="20240326"
        )

def test_fetch_load_data_connection_retry(entsoe_client, mocker):
    """Test connection retry logic"""
    # Mock cache to return None
    mocker.patch(
        'src.data.utils.storage.get_cached_data',
        return_value=None
    )
    
    # Create mock data with proper timezone
    mock_data = pd.DataFrame(
        {'Actual Load': [1, 2, 3]},
        index=pd.date_range(
            start=datetime.now(entsoe_client.tz),
            periods=3,
            freq='h',
            tz=entsoe_client.tz
        )
    )
    
    # Mock query_load_and_forecast to fail twice then succeed
    mock_query = mocker.patch.object(
        entsoe_client.client,
        'query_load_and_forecast',
        side_effect=[
            requests.ConnectionError("First failure"),
            requests.ConnectionError("Second failure"),
            mock_data
        ]
    )

    start_date = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
    end_date = datetime.now().strftime('%Y%m%d')
    
    data = entsoe_client.fetch_load_data(
        start_date=start_date,
        end_date=end_date,
        chunk_size=1  # Use small chunk size to ensure single chunk
    )
    
    assert not data.empty
    assert mock_query.call_count == 3
    assert data.index.tz == entsoe_client.tz

def test_get_load_forecast_synthetic_data(entsoe_client, mocker):
    """Test synthetic data generation when API fails"""
    mocker.patch.object(
        entsoe_client.client,
        'query_load_forecast',
        side_effect=Exception("API Error")
    )
    mocker.patch.object(
        entsoe_client.client,
        'query_load',
        side_effect=Exception("API Error")
    )
    
    start_time = datetime.now(entsoe_client.tz)
    end_time = start_time + timedelta(days=1)
    
    forecast = entsoe_client.get_load_forecast(
        start_time=start_time,
        end_time=end_time
    )
    
    assert isinstance(forecast, pd.DataFrame)
    assert not forecast.empty
    assert len(forecast) > 0
    assert forecast['Load Forecast'].min() >= 40000
    assert forecast['Load Forecast'].max() <= 60000

def test_save_data_handling(entsoe_client, mocker):
    """Test data saving functionality"""
    # Mock cache to return None
    mocker.patch(
        'src.data.utils.storage.get_cached_data',
        return_value=None
    )
    
    # Mock save_raw_data to raise exception
    mocker.patch(
        'src.data.utils.storage.save_raw_data',
        side_effect=Exception("Save failed")
    )
    
    # Create mock data with proper timezone
    mock_data = pd.DataFrame(
        {'Actual Load': [1, 2, 3]},
        index=pd.date_range(
            start=datetime.now(entsoe_client.tz),
            periods=3,
            freq='h',
            tz=entsoe_client.tz
        )
    )
    
    # Mock successful API call
    mocker.patch.object(
        entsoe_client.client,
        'query_load_and_forecast',
        return_value=mock_data
    )

    mock_logger = mocker.patch('src.data.clients.entsoe_client.logger')

    start_date = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
    end_date = datetime.now().strftime('%Y%m%d')

    data = entsoe_client.fetch_load_data(
        start_date=start_date,
        end_date=end_date
    )
    
    assert not data.empty
    mock_logger.error.assert_called_with("Failed to save data: Save failed")

def test_get_load_data_error_handling(entsoe_client, mocker):
    """Test error handling in get_load_data method"""
    # Mock both query methods to fail
    mocker.patch.object(
        entsoe_client.client,
        'query_load_and_forecast',
        side_effect=Exception("Primary method failed")
    )
    mocker.patch.object(
        entsoe_client.client,
        'query_load',
        side_effect=Exception("Fallback method failed")
    )
    
    start_time = datetime.now(entsoe_client.tz)
    end_time = start_time + timedelta(days=1)
    
    # Should return empty DataFrame on failure
    data = entsoe_client.get_load_data(
        start_time=start_time,
        end_time=end_time
    )
    
    assert isinstance(data, pd.DataFrame)
    assert data.empty

def test_get_load_data_series_handling(entsoe_client, mocker):
    """Test handling of Series data in get_load_data"""
    # Create test data as Series
    index = pd.date_range(
        start=datetime.now(entsoe_client.tz),
        periods=24,
        freq='h'
    )
    test_series = pd.Series(
        data=range(24),
        index=index,
        name='load'
    )
    
    # Mock query to return Series
    mocker.patch.object(
        entsoe_client.client,
        'query_load',
        return_value=test_series
    )
    
    data = entsoe_client.get_load_data(
        start_time=index[0],
        end_time=index[-1]
    )
    
    assert isinstance(data, pd.DataFrame)
    assert 'Actual Load' in data.columns

def test_get_latest_load_error(entsoe_client, mocker):
    """Test error handling in get_latest_load"""
    mocker.patch.object(
        entsoe_client.client,
        'query_load_and_forecast',
        side_effect=Exception("API Error")
    )
    
    with pytest.raises(Exception):
        entsoe_client.get_latest_load()

def test_get_load_forecast_timezone_handling(entsoe_client):
    """Test timezone handling in get_load_forecast"""
    # Test with naive datetime
    start_time = datetime.now()
    end_time = start_time + timedelta(days=1)
    
    forecast = entsoe_client.get_load_forecast(
        start_time=start_time,
        end_time=end_time
    )
    
    assert not forecast.empty
    assert forecast.index.tzinfo is not None

def test_get_load_forecast_alternative_method(entsoe_client, mocker):
    """Test alternative forecast method when primary fails"""
    # Mock primary method to fail
    mocker.patch.object(
        entsoe_client.client,
        'query_load_forecast',
        return_value=None
    )
    
    # Mock alternative method to succeed
    test_data = pd.Series(
        data=range(24),
        index=pd.date_range(
            start=datetime.now(entsoe_client.tz),
            periods=24,
            freq='h'
        )
    )
    mocker.patch.object(
        entsoe_client.client,
        'query_load',
        return_value=test_data
    )
    
    forecast = entsoe_client.get_load_forecast()
    
    assert not forecast.empty
    assert isinstance(forecast, (pd.DataFrame, pd.Series))