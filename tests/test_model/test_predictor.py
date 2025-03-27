import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
import pytz
from datetime import datetime, timedelta

from src.model.predictor import LoadPredictor

@pytest.fixture
def predictor():
    return LoadPredictor()

@pytest.fixture
def sample_features():
    """Create sample features for testing."""
    # Create datetime index
    tz = pytz.timezone('Europe/Berlin')
    end_time = datetime.now(tz)
    start_time = end_time - timedelta(days=30)  # 30 days of data
    date_range = pd.date_range(start=start_time, end=end_time, freq='h', tz=tz)  # Changed from '1H' to 'h'
    
    # Generate synthetic features
    n_samples = len(date_range)
    
    data = {
        'hour': date_range.hour,
        'weekday': date_range.dayofweek,
        'month': date_range.month,
        'is_weekend': (date_range.dayofweek >= 5).astype(int),
        'hour_sin': np.sin(2 * np.pi * date_range.hour / 24),
        'hour_cos': np.cos(2 * np.pi * date_range.hour / 24),
        'weekday_sin': np.sin(2 * np.pi * date_range.dayofweek / 7),
        'weekday_cos': np.cos(2 * np.pi * date_range.dayofweek / 7),
        'load_1h_ago': np.random.normal(50000, 5000, n_samples),
        'load_24h_ago': np.random.normal(50000, 5000, n_samples),
        'load_7d_ago': np.random.normal(50000, 5000, n_samples),
        'rolling_mean_24h': np.random.normal(50000, 2000, n_samples),
        'rolling_std_24h': np.abs(np.random.normal(2000, 500, n_samples))
    }
    
    return pd.DataFrame(data, index=date_range)

def test_predictor_initialization(predictor):
    """Test predictor initialization"""
    assert predictor.model is not None
    assert predictor.params is not None
    assert predictor.feature_importance_ is None

def test_predictor_train(predictor, sample_features):
    """Test model training"""
    X = sample_features.copy()
    y = pd.Series(np.random.normal(50000, 5000, size=len(X)), index=X.index)
    
    predictor.train(X, y)
    assert predictor.feature_importance_ is not None
    assert len(predictor.feature_importance_) == len(X.columns)

def test_predictor_predict(predictor, sample_features):
    """Test model predictions"""
    X = sample_features.copy()
    y = pd.Series(np.random.normal(50000, 5000, size=len(X)), index=X.index)
    
    predictor.train(X, y)
    predictions = predictor.predict(X)
    
    assert isinstance(predictions, pd.Series)
    assert len(predictions) == len(X)
    assert predictions.index.equals(X.index)

def test_predictor_save_load(predictor, sample_features):
    """Test model saving and loading"""
    X = sample_features.copy()
    y = pd.Series(np.random.normal(50000, 5000, size=len(X)), index=X.index)
    
    predictor.train(X, y)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.joblib")
        predictor.save(model_path)
        assert os.path.exists(model_path)
        
        new_predictor = LoadPredictor()
        new_predictor.load(model_path)
        
        new_predictions = new_predictor.predict(X)
        original_predictions = predictor.predict(X)
        
        pd.testing.assert_series_equal(new_predictions, original_predictions)

def test_predictor_get_params(predictor):
    """Test getting model parameters"""
    params = predictor.get_params()
    assert isinstance(params, dict)
    assert 'n_estimators' in params
    assert 'learning_rate' in params

if __name__ == '__main__':
    pytest.main([__file__])