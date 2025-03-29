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

class PredictionService:
    """Service for generating load predictions."""
    
    def __init__(self):
        """Initialize the prediction service."""
        self.logger = logging.getLogger(__name__)
        self.deployer = ModelDeployer()
        self.initialized = False
        self.predictions_cache = pd.DataFrame()
        self.metrics_collector = MetricsCollector()
        
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
    
    def get_forecast(self, hours=24, now=None):
        """Get load forecast with improved historical overlap and timestamp handling."""
        try:
            if now is None:
                now = datetime.now(pytz.timezone('Europe/Berlin'))
            
            # Get historical data
            historical_data = self.deployer.pipeline.get_historical_data()
            if historical_data is None:
                self.logger.error("No historical data available for forecasting")
                return None
            
            # Find the last valid timestamp in historical data
            last_valid_time = historical_data['Actual Load'].dropna().index.max()
            self.logger.info(f"Last valid timestamp in historical data: {last_valid_time}")
            
            # Extend historical overlap to 24 hours
            historical_overlap = pd.date_range(
                start=max(now - pd.Timedelta(hours=24), last_valid_time - pd.Timedelta(hours=24)),
                end=last_valid_time,
                freq='15min'
            )
            
            # Generate future timestamps
            forecast_timestamps = pd.date_range(
                start=now, 
                periods=hours*4, 
                freq='15min'
            )
            
            # Combine timestamps and ensure timezone
            timestamps = historical_overlap.union(forecast_timestamps)
            if timestamps.tz is None:
                timestamps = timestamps.tz_localize('Europe/Berlin')
            
            # Get predictions
            predictions = self.deployer.make_prediction(timestamps)
            if predictions is None:
                self.logger.error("Failed to generate predictions")
                return None
            
            self.logger.info(f"Generated predictions from {predictions.index.min()} to {predictions.index.max()}")
            self.logger.info(f"Final forecast has {len(predictions)} points")
            
            return predictions
        
        except Exception as e:
            self.logger.error(f"Error in get_forecast: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None
     
