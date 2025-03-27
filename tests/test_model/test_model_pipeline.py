import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from unittest.mock import Mock, patch

from src.deployment.pipeline import DataPipeline

@pytest.fixture
def sample_load_data():
    """Create sample load data for testing"""
    dates = pd.date_range(
        start='2024-01-01',
        end='2024-01-08',
        freq='15min',
        tz='Europe/Berlin'
    )
    
    # Create realistic load pattern
    hours = dates.hour
    weekdays = dates.weekday < 5  # True for weekdays
    
    base_load = 45000 + 15000 * np.sin(np.pi * (hours - 6) / 12)  # Daily pattern
    weekday_effect = weekdays.astype(int) * 5000  # Higher load on weekdays
    noise = np.random.normal(0, 1000, len(dates))
    
    load = base_load + weekday_effect + noise
    
    return pd.DataFrame({
        'Actual Load': load
    }, index=dates)

@pytest.fixture
def mock_pipeline():
    """Create a pipeline with mocked components"""
    with patch('src.deployment.pipeline.EntsoeClient') as mock_client, \
         patch('src.deployment.pipeline.LoadDataPreprocessor') as mock_preprocessor, \
         patch('src.deployment.pipeline.LoadFeatureExtractor') as mock_extractor, \
         patch('src.deployment.pipeline.LoadPredictor') as mock_predictor:
        
        pipeline = DataPipeline()
        
        # Configure mocks
        pipeline.entsoe_client = mock_client.return_value
        pipeline.preprocessor = mock_preprocessor.return_value
        pipeline.feature_extractor = mock_extractor.return_value
        pipeline.model = mock_predictor.return_value
        
        yield pipeline

def test_pipeline_initialization():
    """Test pipeline initialization"""
    pipeline = DataPipeline()
    assert pipeline.entsoe_client is not None
    assert pipeline.preprocessor is not None
    assert pipeline.feature_extractor is not None
    assert pipeline.model is not None
    assert pipeline.tz == pytz.timezone('Europe/Berlin')

def test_prepare_training_data(mock_pipeline, sample_load_data):
    """Test preparation of training data"""
    # Configure mocks
    mock_pipeline.entsoe_client.fetch_load_data.return_value = sample_load_data
    mock_pipeline.preprocessor.preprocess.return_value = sample_load_data
    
    # Create sample features - use the full data length
    # The pipeline will handle the trimming internally
    sample_features = pd.DataFrame(
        np.random.random((len(sample_load_data), 5)),
        index=sample_load_data.index,
        columns=['feat1', 'feat2', 'feat3', 'feat4', 'feat5']
    )
    mock_pipeline.feature_extractor.transform.return_value = sample_features
    
    # Test the method
    features, target = mock_pipeline.prepare_training_data('2024-01-01', '2024-01-08')
    
    # Verify lengths
    assert len(features) == len(target)
    assert len(features) == len(sample_load_data) - 96  # Should remove last 24 hours
    assert not features.isnull().any().any()
    assert not target.isnull().any()
    assert all(features.index == target.index)  # Check index alignment

def test_train_model(mock_pipeline, sample_load_data):
    """Test model training"""
    # Configure mocks
    mock_pipeline.entsoe_client.fetch_load_data.return_value = sample_load_data
    mock_pipeline.preprocessor.preprocess.return_value = sample_load_data
    
    features_index = sample_load_data.index[:-96]
    sample_features = pd.DataFrame(
        np.random.random((len(features_index), 5)),
        index=features_index,
        columns=['feat1', 'feat2', 'feat3', 'feat4', 'feat5']
    )
    mock_pipeline.feature_extractor.transform.return_value = sample_features
    
    # Test training
    mock_pipeline.train_model('2024-01-01', '2024-01-08', model_path='test_model.pkl')
    
    # Verify calls
    mock_pipeline.model.train.assert_called_once()
    mock_pipeline.model.save.assert_called_once_with('test_model.pkl')

def test_make_prediction(mock_pipeline, sample_load_data):
    """Test prediction generation"""
    # Configure mocks
    mock_pipeline.entsoe_client.get_load_data.return_value = sample_load_data
    mock_pipeline.preprocessor.preprocess.return_value = sample_load_data
    
    sample_features = pd.DataFrame(
        np.random.random((len(sample_load_data), 5)),
        index=sample_load_data.index,
        columns=['feat1', 'feat2', 'feat3', 'feat4', 'feat5']
    )
    mock_pipeline.feature_extractor.transform.return_value = sample_features
    
    # Configure prediction output
    prediction_index = pd.date_range(
        start=sample_load_data.index[-1],
        periods=96,  # 24 hours of 15-min intervals
        freq='15min',
        tz='Europe/Berlin'
    )
    mock_predictions = pd.Series(
        np.random.normal(50000, 5000, 96),
        index=prediction_index
    )
    mock_pipeline.model.predict.return_value = mock_predictions
    
    # Test prediction
    current_time = datetime.now(pytz.timezone('Europe/Berlin'))
    predictions = mock_pipeline.make_prediction(current_time)
    
    assert len(predictions) == 96  # Should predict 24 hours
    assert predictions.index.freq == '15min'
    assert not predictions.isnull().any()

def test_error_handling_no_data(mock_pipeline):
    """Test error handling when no data is available"""
    mock_pipeline.entsoe_client.fetch_load_data.return_value = pd.DataFrame()
    
    with pytest.raises(ValueError, match="No data fetched from ENTSOE"):
        mock_pipeline.prepare_training_data('2024-01-01', '2024-01-08')

def test_error_handling_prediction(mock_pipeline):
    """Test error handling during prediction"""
    mock_pipeline.entsoe_client.get_load_data.return_value = pd.DataFrame()
    
    with pytest.raises(ValueError, match="No historical data available"):
        mock_pipeline.make_prediction()