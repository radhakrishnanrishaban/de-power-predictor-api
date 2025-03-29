import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import List, Optional
import traceback

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
        """Extract features from load data focusing solely on feature creation."""
        try:
            if not isinstance(data, pd.DataFrame) or data.empty:
                logger.error("Invalid input data")
                return None
            
            df = data.copy()
            logger.info(f"Input data columns: {df.columns.tolist()}")
            
            # Ensure index is datetime with timezone
            if not isinstance(df.index, pd.DatetimeIndex):
                logger.error("DataFrame index must be DatetimeIndex")
                return None
            
            # Create features
            features = pd.DataFrame(index=df.index)
            
            # Time-based features (these are independent of the load data)
            features['hour'] = df.index.hour
            features['weekday'] = df.index.dayofweek
            features['month'] = df.index.month
            features['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
            features['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
            features['weekday_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
            features['weekday_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
            
            # Load history features (assuming preprocessor has handled missing values)
            features['96_periods_ago_load'] = df['Actual Load'].shift(96)  # 24 hours ago
            features['672_periods_ago_load'] = df['Actual Load'].shift(672)  # 7 days ago
            
            # Rolling statistics
            features['rolling_mean_24h'] = df['Actual Load'].rolling(window=96, min_periods=1, center=True).mean()
            features['rolling_std_24h'] = df['Actual Load'].rolling(window=96, min_periods=1, center=True).std()
            features['rolling_mean_168h'] = df['Actual Load'].rolling(window=672, min_periods=1, center=True).mean()
            features['rolling_std_168h'] = df['Actual Load'].rolling(window=672, min_periods=1, center=True).std()
            
            # Holiday feature (default to 0, should be updated by the pipeline)
            features['is_day_before_holiday'] = 0
            
            # Copy the target variable if requested
            if include_target:
                features['Actual Load'] = df['Actual Load']
            
            logger.info(f"Created features: {sorted(features.columns.tolist())}")
            logger.info(f"Created {len(features.columns)} features")
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_names

    def transform(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Transform data by extracting features."""
        return self.extract_features(data, include_target=False)