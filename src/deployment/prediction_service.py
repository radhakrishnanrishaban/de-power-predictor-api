import logging
import pandas as pd
from datetime import datetime, timedelta
import pytz
import numpy as np
from typing import Optional
import traceback

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.deployment.model_deployer import ModelDeployer
from src.data.utils.metrics import MetricsCollector
from src.model.feature_extractor import LoadFeatureExtractor
from src.data.clients.entsoe_client import EntsoeClient

class PredictionService:
    """Service for generating load predictions."""
    
    def __init__(self):
        """Initialize the prediction service."""
        self.logger = logging.getLogger(__name__)
        self.deployer = ModelDeployer()
        self.initialized = False
        self.predictions_cache = pd.DataFrame()
        self.metrics_collector = MetricsCollector()
        self.feature_extractor = LoadFeatureExtractor()
        self.entsoe_client = EntsoeClient()
        
    def initialize(self):
        """Initialize the service."""
        if not self.initialized:
            success = self.deployer.initialize_pipeline()
            if success:
                success = self.deployer.load_model()
                self.initialized = True
                self.logger.info("Prediction service initialized")
                return success
            else:
                self.logger.error("Failed to initialize prediction service")
                return False
        return True
    
    def get_prediction(self, timestamp):
        """Get prediction for a specific timestamp."""
        if not self.initialized:
            self.initialize()
            
        try:
            # Ensure timestamp has timezone info
            berlin_tz = pytz.timezone('Europe/Berlin')
            if timestamp.tzinfo is None:
                timestamp = berlin_tz.localize(timestamp)
            
            # Check cache first
            if not self.predictions_cache.empty and timestamp in self.predictions_cache.index:
                cached_value = self.predictions_cache.loc[timestamp].iloc[0]
                if isinstance(cached_value, (np.integer, np.floating)):
                    return cached_value.item()
                return cached_value
            
            # Get new prediction
            try:
                prediction = self.deployer.make_prediction(timestamp)
                if prediction is not None:
                    # Add to cache
                    self.predictions_cache = pd.concat([self.predictions_cache, 
                                                       pd.DataFrame(prediction).T])
                    pred_value = prediction.iloc[0]
                    if isinstance(pred_value, (np.integer, np.floating)):
                        pred_value = pred_value.item()
                    return pred_value
                else:
                    self.logger.error(f"No prediction available for {timestamp} - insufficient data")
                    return None
            except Exception as e:
                self.logger.error(f"Error making prediction for {timestamp}: {str(e)}")
                import traceback
                self.logger.debug(f"Traceback: {traceback.format_exc()}")
                return None
            
        except Exception as e:
            self.logger.error(f"Error in get_prediction: {str(e)}")
            return None
    
    def get_forecast(self, hours: int = 24, start_time: Optional[datetime] = None) -> pd.DataFrame:
        """Get load forecast with proper feature handling."""
        try:
            if start_time is None:
                start_time = datetime.now(pytz.timezone('Europe/Berlin'))
            
            # Get historical data for feature creation
            historical_end = start_time
            historical_start = historical_end - timedelta(days=7)  # Get 1 week of history
            
            historical_data = self.entsoe_client.get_load_data(historical_start, historical_end)
            
            if historical_data.empty:
                self.logger.error("No historical data available")
                return pd.DataFrame()
            
            # Create prediction timestamps
            pred_index = pd.date_range(
                start=start_time,
                periods=hours * 4,  # 15-minute intervals
                freq='15min',
                tz='Europe/Berlin'
            )
            
            # Prepare data for feature extraction
            pred_data = pd.DataFrame(index=pred_index)
            if 'Actual Load' in historical_data.columns:
                pred_data['Actual Load'] = historical_data['Actual Load']
            
            # Extract features
            features = self.feature_extractor.extract_features(
                data=pred_data,
                historical_data=historical_data
            )
            
            # Make predictions
            predictions = self.deployer.make_prediction(features)
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error getting forecast: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()
    
