import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

from src.model.feature_extractor import LoadFeatureExtractor

@pytest.fixture
def sample_load_data():
    """Create sample load data for testing."""
    # Create datetime index
    tz = pytz.timezone('Europe/Berlin')
    end_time = datetime.now(tz)
    start_time = end_time - timedelta(days=10)
    date_range = pd.date_range(start=start_time, end=end_time, freq='15min', tz=tz)
    
    # Generate synthetic load data with daily and weekly patterns
    hourly_pattern = np.sin(np.pi * date_range.hour / 24) + 1
    weekly_pattern = np.sin(np.pi * date_range.dayofweek / 7) * 0.3
    
    # Base load with patterns
    load = 50000 + 10000 * hourly_pattern + 5000 * weekly_pattern
    
    # Add random noise
    noise = np.random.normal(0, 1000, size=len(date_range))
    load += noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'Actual Load': load
    }, index=date_range)
    
    return df

def test_feature_extractor_initialization():
    """Test feature extractor initialization."""
    extractor = LoadFeatureExtractor()
    assert extractor.required_columns == ['Actual Load']
    assert isinstance(extractor.feature_names, list)
    assert len(extractor.feature_names) == 0

def test_extract_features_basic(sample_load_data):
    """Test basic feature extraction functionality."""
    extractor = LoadFeatureExtractor()
    features = extractor.extract_features(sample_load_data)
    
    assert features is not None
    assert isinstance(features, pd.DataFrame)
    assert not features.empty
    
    # Check if all expected features are present
    expected_features = [
        'hour', 'weekday', 'month', 'is_weekend',
        'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos',
        'load_1h_ago', 'load_24h_ago', 'load_7d_ago',
        'rolling_mean_24h', 'rolling_std_24h'
    ]
    
    for feature in expected_features:
        assert feature in features.columns

def test_extract_features_values(sample_load_data):
    """Test the values of extracted features."""
    extractor = LoadFeatureExtractor()
    features = extractor.extract_features(sample_load_data)
    
    # Test hour values
    assert features['hour'].min() >= 0
    assert features['hour'].max() <= 23
    
    # Test weekday values
    assert features['weekday'].min() >= 0
    assert features['weekday'].max() <= 6
    
    # Test month values
    assert features['month'].min() >= 1
    assert features['month'].max() <= 12
    
    # Test cyclical features bounds
    assert features['hour_sin'].between(-1, 1).all()
    assert features['hour_cos'].between(-1, 1).all()
    assert features['weekday_sin'].between(-1, 1).all()
    assert features['weekday_cos'].between(-1, 1).all()

def test_extract_features_without_target(sample_load_data):
    """Test feature extraction without target variable."""
    extractor = LoadFeatureExtractor()
    features = extractor.extract_features(sample_load_data, include_target=False)
    
    assert 'target' not in features.columns

def test_extract_features_with_missing_data():
    """Test feature extraction with missing data."""
    # Create data with missing values
    tz = pytz.timezone('Europe/Berlin')
    date_range = pd.date_range(
        start=datetime.now(tz) - timedelta(days=1),
        end=datetime.now(tz),
        freq='15min'
    )
    df = pd.DataFrame({
        'Actual Load': np.nan
    }, index=date_range)
    
    extractor = LoadFeatureExtractor()
    features = extractor.extract_features(df)
    
    assert features is not None
    # Check if rolling statistics handle missing values
    assert not features['rolling_mean_24h'].isna().all()
    assert not features['rolling_std_24h'].isna().all()

def test_extract_features_empty_data():
    """Test feature extraction with empty DataFrame."""
    extractor = LoadFeatureExtractor()
    features = extractor.extract_features(pd.DataFrame())
    
    assert features is None

def test_extract_features_invalid_input():
    """Test feature extraction with invalid input."""
    extractor = LoadFeatureExtractor()
    
    # Test with None
    assert extractor.extract_features(None) is None
    
    # Test with invalid type
    assert extractor.extract_features([1, 2, 3]) is None

def test_get_feature_names(sample_load_data):
    """Test getting feature names after extraction."""
    extractor = LoadFeatureExtractor()
    features = extractor.extract_features(sample_load_data)
    
    feature_names = extractor.get_feature_names()
    assert isinstance(feature_names, list)
    assert len(feature_names) > 0
    
    # Check if all feature names are in the DataFrame
    for name in feature_names:
        assert name in features.columns

def test_feature_extraction_consistency(sample_load_data):
    """Test consistency of feature extraction across multiple calls."""
    extractor = LoadFeatureExtractor()
    
    features1 = extractor.extract_features(sample_load_data)
    features2 = extractor.extract_features(sample_load_data)
    
    # Check if features are identical across calls
    pd.testing.assert_frame_equal(features1, features2)

def test_feature_extraction_with_timezone_handling(sample_load_data):
    """Test feature extraction with different timezones."""
    extractor = LoadFeatureExtractor()
    
    # Convert to UTC
    utc_data = sample_load_data.tz_convert('UTC')
    utc_features = extractor.extract_features(utc_data)
    
    # Convert to America/New_York
    ny_data = sample_load_data.tz_convert('America/New_York')
    ny_features = extractor.extract_features(ny_data)
    
    # Reset index to compare hours directly
    utc_hours = utc_features['hour'].values
    ny_hours = ny_features['hour'].values
    
    # Hours should be different due to timezone conversion
    assert not np.array_equal(utc_hours, ny_hours)
    
    # But the actual load values and derived features should be the same
    pd.testing.assert_series_equal(
        utc_features['rolling_mean_24h'],
        ny_features['rolling_mean_24h'],
        check_index=False  # Don't check index equality since timezones differ
    )

if __name__ == '__main__':
    pytest.main([__file__])