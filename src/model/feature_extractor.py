import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class LoadFeatureExtractor:
    """Feature extractor for load prediction."""
    
    def __init__(self):
        self.required_columns = ['Actual Load']
        self.feature_names: List[str] = []
    
    def extract_features(
        self,
        data: pd.DataFrame,
        include_target: bool = True
    ) -> Optional[pd.DataFrame]:
        """Extract features from load data."""
        try:
            if not isinstance(data, pd.DataFrame) or data.empty:
                logger.error("Invalid input data")
                return None
            
            df = data.copy()
            print("\nInput data columns:", df.columns.tolist())
            
            # Handle missing values first
            df['Actual Load'] = df['Actual Load'].ffill().bfill()
            if df['Actual Load'].isna().all():
                df['Actual Load'] = 0
            
            # Create features in a controlled manner
            features = pd.DataFrame(index=df.index)
            
            # Copy Actual Load directly (model expects this)
            features['Actual Load'] = df['Actual Load']
            
            # Time-based features (these match the model's expected features)
            features['hour'] = df.index.hour
            features['weekday'] = df.index.dayofweek
            features['month'] = df.index.month
            features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
            features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
            features['weekday_sin'] = np.sin(2 * np.pi * features['weekday'] / 7)
            features['weekday_cos'] = np.cos(2 * np.pi * features['weekday'] / 7)
            
            # Load history features with correct names
            features['96_periods_ago_load'] = df['Actual Load'].shift(96)  # 24 hours ago
            features['672_periods_ago_load'] = df['Actual Load'].shift(672)  # 7 days ago
            
            # Rolling statistics
            features['rolling_mean_24h'] = df['Actual Load'].rolling(window=96, min_periods=1, center=True).mean()
            features['rolling_std_24h'] = df['Actual Load'].rolling(window=96, min_periods=1, center=True).std()
            features['rolling_mean_168h'] = df['Actual Load'].rolling(window=672, min_periods=1, center=True).mean()
            features['rolling_std_168h'] = df['Actual Load'].rolling(window=672, min_periods=1, center=True).std()
            
            # Holiday feature (default to 0)
            features['is_day_before_holiday'] = 0
            
            # Fill NaN values
            features = features.ffill().bfill()
            
            print("\nCreated features:", sorted(features.columns.tolist()))
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return None
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_names

    def transform(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Transform data by extracting features."""
        return self.extract_features(data, include_target=False)