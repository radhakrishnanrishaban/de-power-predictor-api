import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from src.data.preprocess.preprocessor import LoadDataPreprocessor

@pytest.fixture
def preprocessor():
    """Create a preprocessor instance with default settings"""
    return LoadDataPreprocessor()

@pytest.fixture
def basic_load_data():
    """Create a simple, clean load data DataFrame"""
    tz = pytz.timezone('Europe/Berlin')
    start_time = datetime.now(tz).replace(hour=0, minute=0, second=0, microsecond=0)
    end_time = start_time + timedelta(days=1)
    
    dates = pd.date_range(
        start=start_time,
        end=end_time,  # This ensures we get exactly 24 hours
        freq='15min',
        tz='Europe/Berlin'
    )
    
    # Create simple load values with daily pattern
    load_values = 50000 + 10000 * np.sin(np.pi * np.arange(len(dates)) / 48)
    return pd.DataFrame({'Actual Load': load_values}, index=dates)

@pytest.fixture
def problematic_load_data():
    """Create load data with various issues to test preprocessing"""
    tz = pytz.timezone('Europe/Berlin')
    dates = pd.date_range(
        start=datetime.now(tz) - timedelta(days=1),
        periods=96,
        freq='15min',
        tz='Europe/Berlin'
    )
    
    df = pd.DataFrame({'Actual Load': 50000 * np.ones(96)}, index=dates)
    
    # Add specific test cases
    df.iloc[0] = -1000  # Negative value
    df.iloc[1] = 200000  # Unreasonably high value
    df.iloc[2] = np.nan  # Missing value
    df.iloc[3] = df.iloc[4] * 10  # Outlier
    
    # Add duplicate index
    duplicate_row = pd.DataFrame(
        {'Actual Load': [60000]},
        index=[dates[5]]
    )
    df = pd.concat([df, duplicate_row])
    
    return df

def test_initialization(preprocessor):
    """Test preprocessor initialization"""
    assert preprocessor.freq == '15min'
    assert preprocessor.timezone == 'Europe/Berlin'
    assert preprocessor.rolling_window == 96
    assert preprocessor.zscore_threshold == 3.0

def test_prepare_index_basic(preprocessor, basic_load_data):
    """Test index preparation with clean data"""
    result = preprocessor._prepare_index(basic_load_data)
    
    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.index.tz.zone == 'Europe/Berlin'
    assert result.index.freq == '15min'
    assert len(result) == len(basic_load_data)

def test_prepare_index_with_issues(preprocessor, problematic_load_data):
    """Test index preparation with problematic data"""
    result = preprocessor._prepare_index(problematic_load_data)
    
    assert not result.index.duplicated().any()
    assert result.index.freq == '15min'
    assert result.index.tz.zone == 'Europe/Berlin'

def test_handle_duplicates(preprocessor, problematic_load_data):
    """Test duplicate handling"""
    result = preprocessor._handle_duplicates(problematic_load_data)
    
    assert not result.index.duplicated().any()
    assert len(result) == len(problematic_load_data) - 1  # One duplicate removed

def test_validate_values(preprocessor, problematic_load_data):
    """Test value validation"""
    result = preprocessor._validate_values(problematic_load_data)
    
    assert (result['Actual Load'] >= 0).all()  # No negative values
    assert (result['Actual Load'] <= 100000).all()  # No unreasonably high values

def test_remove_outliers(preprocessor, problematic_load_data):
    """Test outlier removal"""
    result = preprocessor._remove_outliers(problematic_load_data)
    
    # Calculate z-scores using the same parameters as the implementation
    min_periods = max(3, preprocessor.rolling_window // 4)
    rolling_mean = result['Actual Load'].rolling(
        window=preprocessor.rolling_window,
        center=True,
        min_periods=min_periods
    ).mean().fillna(result['Actual Load'].mean())
    
    rolling_std = result['Actual Load'].rolling(
        window=preprocessor.rolling_window,
        center=True,
        min_periods=min_periods
    ).std().fillna(result['Actual Load'].std())
    
    # Handle zero/NaN standard deviation
    rolling_std = rolling_std.replace(0, result['Actual Load'].std())
    
    # Calculate z-scores
    z_scores = abs((result['Actual Load'] - rolling_mean) / rolling_std)
    z_scores = z_scores.fillna(0)  # Replace any remaining NaNs with 0
    
    assert (z_scores <= preprocessor.zscore_threshold).all()

def test_handle_missing_values(preprocessor, problematic_load_data):
    """Test missing value handling"""
    result = preprocessor._handle_missing_values(problematic_load_data)
    
    assert not result['Actual Load'].isna().any()
    assert result['Actual Load'].notna().all()

def test_full_preprocessing_pipeline(preprocessor, problematic_load_data):
    """Test complete preprocessing pipeline"""
    result = preprocessor.preprocess(problematic_load_data)
    
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert not result.index.duplicated().any()
    assert (result['Actual Load'] >= 0).all()
    assert not result['Actual Load'].isna().any()
    assert result.index.freq == '15min'

def test_historical_validation(preprocessor, basic_load_data):
    """Test historical data validation"""
    # Test with sufficient data (modify min_days parameter)
    assert preprocessor.validate_historical_data(basic_load_data, min_days=1)
    
    # Test with insufficient data
    short_data = basic_load_data.iloc[:10]
    assert not preprocessor.validate_historical_data(short_data, min_days=1)

def test_historical_initialization(preprocessor, basic_load_data):
    """Test initialization with historical data"""
    preprocessor.initialize_with_historical(basic_load_data)
    
    assert preprocessor.historical_start is not None
    assert preprocessor.historical_end is not None
    assert preprocessor.data_mean is not None
    assert preprocessor.data_std is not None

def test_live_preprocessing(preprocessor, basic_load_data):
    """Test live data preprocessing"""
    # Initialize with first half of data
    historical_data = basic_load_data.iloc[:48]
    preprocessor.initialize_with_historical(historical_data)
    
    # Process second half as live data
    live_data = basic_load_data.iloc[48:]
    result = preprocessor.preprocess_live(live_data)
    
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert not result['Actual Load'].isna().any()
    assert (result['Actual Load'] >= 0).all()

def test_error_handling(preprocessor):
    """Test error handling"""
    with pytest.raises(ValueError, match="Input must be a pandas DataFrame"):
        preprocessor.preprocess(None)
    
    with pytest.raises(ValueError, match="DataFrame must contain 'Actual Load' column"):
        preprocessor.preprocess(pd.DataFrame({'wrong_column': [1, 2, 3]}))

def test_edge_cases(preprocessor):
    """Test edge cases"""
    # Empty DataFrame
    empty_df = pd.DataFrame(columns=['Actual Load'])
    assert preprocessor.preprocess(empty_df).empty
    
    # Single row
    single_row = pd.DataFrame(
        {'Actual Load': [50000]},
        index=[datetime.now(pytz.timezone('Europe/Berlin'))]
    )
    result = preprocessor.preprocess(single_row)
    assert len(result) == 1

def test_error_handling_comprehensive(preprocessor):
    """Test comprehensive error handling scenarios"""
    # Test None data in preprocess method instead of _prepare_index
    with pytest.raises(ValueError, match="Input must be a pandas DataFrame"):
        preprocessor.preprocess(None)
    
    # Test empty DataFrame without required column
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="DataFrame must contain 'Actual Load' column"):
        preprocessor.preprocess(empty_df)
    
    # Test invalid index with required column
    invalid_df = pd.DataFrame({'Actual Load': [1, 2, 3]}, index=[1, 2, 3])
    result = preprocessor.preprocess(invalid_df)
    assert isinstance(result.index, pd.DatetimeIndex)  # Should convert to datetime index

def test_statistical_updates(preprocessor, basic_load_data):
    """Test statistical updates with various data scenarios"""
    # Test with small dataset
    small_data = basic_load_data.iloc[:5]
    preprocessor._update_statistics(small_data)
    assert preprocessor.rolling_mean is None  # Should not compute rolling stats
    
    # Test with normal dataset
    preprocessor._update_statistics(basic_load_data)
    assert preprocessor.rolling_mean is not None
    assert preprocessor.data_mean is not None

def test_live_preprocessing_edge_cases(preprocessor):
    """Test live preprocessing with edge cases"""
    # Test with empty DataFrame
    assert preprocessor.preprocess_live(pd.DataFrame()) is None
    
    # Test with small DataFrame with historical context
    small_df = pd.DataFrame(
        {'Actual Load': [50000]},
        index=[datetime.now(pytz.timezone('Europe/Berlin'))]
    )
    
    # Initialize with historical data first
    historical_data = pd.DataFrame(
        {'Actual Load': [50000] * 96},
        index=pd.date_range(
            start=datetime.now(pytz.timezone('Europe/Berlin')) - timedelta(days=1),
            periods=96,
            freq='15min'
        )
    )
    preprocessor.initialize_with_historical(historical_data)
    
    # Now test live preprocessing
    result = preprocessor.preprocess_live(small_df)
    assert result is not None
    
    # Test with invalid data
    invalid_df = pd.DataFrame({'Wrong Column': [1, 2, 3]})
    with pytest.raises(ValueError):
        preprocessor.preprocess_live(invalid_df)

def test_historical_validation_edge_cases(preprocessor):
    """Test historical validation with edge cases"""
    tz = pytz.timezone('Europe/Berlin')
    now = datetime.now(tz)
    
    # Test with proper one-day data
    one_day_data = pd.DataFrame(
        {'Actual Load': [50000] * 96},
        index=pd.date_range(
            start=now.replace(hour=0, minute=0, second=0, microsecond=0),
            periods=96,
            freq='15min',
            tz=tz
        )
    )
    assert preprocessor.validate_historical_data(one_day_data, min_days=1)
    
    # Test with very short data
    short_df = pd.DataFrame(
        {'Actual Load': [50000]},
        index=[now]
    )
    assert not preprocessor.validate_historical_data(short_df, min_days=1)
    
    # Test with None
    assert not preprocessor.validate_historical_data(None, min_days=1)
    
    # Test with empty DataFrame
    assert not preprocessor.validate_historical_data(pd.DataFrame(), min_days=1)