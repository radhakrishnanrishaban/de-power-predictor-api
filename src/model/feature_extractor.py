import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import List, Optional
import traceback

logger = logging.getLogger(__name__)

class LoadFeatureExtractor:
    """Feature extractor for load prediction with enhanced pattern detection."""
    
    def __init__(self):
        self.required_columns = ['Actual Load']
        self.feature_names = []
    
    def extract_features(self, data: pd.DataFrame, historical_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Extract features with proper handling of gradients."""
        try:
            features = pd.DataFrame(index=data.index)
            
            # Time-based features
            features['hour'] = data.index.hour
            features['weekday'] = data.index.dayofweek
            features['month'] = data.index.month
            features['is_weekend'] = (data.index.weekday >= 5).astype(int)
            
            # Cyclical encoding
            features['hour_sin'] = np.sin(2 * np.pi * data.index.hour / 24)
            features['hour_cos'] = np.cos(2 * np.pi * data.index.hour / 24)
            features['weekday_sin'] = np.sin(2 * np.pi * data.index.dayofweek / 7)
            features['weekday_cos'] = np.cos(2 * np.pi * data.index.dayofweek / 7)
            
            if 'Actual Load' in data.columns:
                # Lagged features
                for lag in [1, 2, 3, 4]:
                    features[f'load_lag_{lag}'] = data['Actual Load'].shift(lag)
                
                # Daily and weekly patterns
                features['load_yesterday'] = data['Actual Load'].shift(96)  # 24h ago
                features['load_lastweek'] = data['Actual Load'].shift(672)  # 1 week ago
                
                # Rolling statistics
                for window in [24, 48, 168]:  # 24h, 48h, 1w
                    roll = data['Actual Load'].rolling(window=window)
                    features[f'rolling_mean_{window}h'] = roll.mean()
                    features[f'rolling_std_{window}h'] = roll.std()
                
                # Peak/trough indicators
                features['is_peak_hour'] = ((data.index.hour >= 8) & (data.index.hour <= 10) |
                                          (data.index.hour >= 18) & (data.index.hour <= 20)).astype(int)
                features['is_trough_hour'] = ((data.index.hour >= 2) & 
                                            (data.index.hour <= 4)).astype(int)
                
                # Load dynamics (with proper handling of NaN values)
                load_series = data['Actual Load'].ffill()
                features['load_gradient'] = load_series.pct_change(fill_method=None)
                features['load_acceleration'] = features['load_gradient'].pct_change(fill_method=None)
            
            # Fill missing values using newer pandas methods
            features = features.ffill().bfill()
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _get_rolling_max(self, data: pd.DataFrame, window: int) -> pd.Series:
        """Get rolling maximum with forward-looking prevention"""
        return data['Actual Load'].rolling(window=window, min_periods=1).max().shift(1)
    
    def _get_rolling_min(self, data: pd.DataFrame, window: int) -> pd.Series:
        """Get rolling minimum with forward-looking prevention"""
        return data['Actual Load'].rolling(window=window, min_periods=1).min().shift(1)
    
    def _get_rate_of_change(self, data: pd.DataFrame, periods: int) -> pd.Series:
        """Calculate rate of change"""
        return (data['Actual Load'] - data['Actual Load'].shift(periods)) / periods
    
    def _get_acceleration(self, data: pd.DataFrame, periods: int) -> pd.Series:
        """Calculate acceleration (change in rate of change)"""
        roc = self._get_rate_of_change(data, periods)
        return roc - roc.shift(periods)
    
    def _get_peak_deviation(self, historical_data: pd.DataFrame, target_index: pd.DatetimeIndex) -> pd.Series:
        """Calculate deviation from expected peaks"""
        # Group by hour and day of week
        expected_peaks = historical_data.groupby(
            [historical_data.index.hour, historical_data.index.dayofweek]
        )['Actual Load'].mean()
        
        deviations = []
        for idx in target_index:
            expected = expected_peaks.get((idx.hour, idx.dayofweek), expected_peaks.mean())
            actual = historical_data.loc[idx, 'Actual Load'] if idx in historical_data.index else expected
            deviations.append((actual - expected) / expected)
        
        return pd.Series(deviations, index=target_index)
    
    def _get_rolling_mean(self, data: pd.DataFrame, start_time: pd.Timestamp, periods: int) -> pd.Series:
        """Calculate rolling mean"""
        return data['Actual Load'].rolling(window=periods, min_periods=1).mean()
    
    def _get_rolling_std(self, data: pd.DataFrame, start_time: pd.Timestamp, periods: int) -> pd.Series:
        """Calculate rolling standard deviation"""
        return data['Actual Load'].rolling(window=periods, min_periods=1).std()
    
    def _get_load_change(self, data: pd.DataFrame, start_time: pd.Timestamp, periods: int) -> pd.Series:
        """Calculate load change"""
        return data['Actual Load'].diff(periods=periods)
    
    def _is_holiday(self, index: pd.DatetimeIndex) -> pd.Series:
        """Detect if a date is a holiday"""
        # This is a placeholder implementation. You might want to implement a more robust holiday detection logic based on your data.
        return pd.Series(index=index, data=[0] * len(index))
    
    def _get_pattern_deviation(self, historical_data: pd.DataFrame, target_index: pd.DatetimeIndex) -> pd.Series:
        """Calculate deviation from expected pattern"""
        # Group by hour and day of week
        expected_pattern = historical_data.groupby(
            [historical_data.index.hour, historical_data.index.dayofweek]
        )['Actual Load'].mean()
        
        deviations = []
        for idx in target_index:
            expected = expected_pattern.get((idx.hour, idx.dayofweek), expected_pattern.mean())
            actual = historical_data.loc[idx, 'Actual Load'] if idx in historical_data.index else expected
            deviations.append((actual - expected) / expected)
        
        return pd.Series(deviations, index=target_index)

    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_names

    def transform(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Transform data by extracting features."""
        return self.extract_features(data, include_target=False)