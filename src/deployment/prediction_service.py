import logging
import pandas as pd
from datetime import datetime, timedelta
import pytz
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.deployment.model_deployer import ModelDeployer

class PredictionService:
    """Service for generating load predictions."""
    
    def __init__(self):
        """Initialize the prediction service."""
        self.logger = logging.getLogger(__name__)
        self.deployer = ModelDeployer()
        self.initialized = False
        self.predictions_cache = pd.DataFrame()
        
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
    
    def get_forecast(self, hours=24):
        if not self.initialized:
            self.initialize()
        
        try:
            # Start with UTC time for consistency - use pandas Timestamp instead of datetime
            utc_now = pd.Timestamp.now(tz='UTC').replace(second=0, microsecond=0)
            
            # Round to nearest 15 minutes
            minutes = utc_now.minute
            utc_now = utc_now.replace(minute=15 * (minutes // 15))
            
            # Convert to Berlin time for model predictions
            berlin_tz = pytz.timezone('Europe/Berlin')
            now = utc_now.tz_convert(berlin_tz)
            
            # Create exactly 96 timestamps (24 hours at 15-min intervals)
            timestamps = pd.date_range(start=now, periods=hours*4, freq='15min')
            timestamps_series = pd.Series(index=timestamps, dtype=float)
            
            if not self.predictions_cache.empty:
                # Ensure cache index is in Berlin timezone
                if self.predictions_cache.index.tzinfo is None:
                    self.predictions_cache.index = self.predictions_cache.index.tz_localize(berlin_tz)
                else:
                    self.predictions_cache.index = self.predictions_cache.index.tz_convert(berlin_tz)
                    
                cached_preds = self.predictions_cache.loc[
                    self.predictions_cache.index.isin(timestamps)
                ]
                if not cached_preds.empty:
                    timestamps_series.loc[cached_preds.index] = cached_preds.iloc[:, 0]
            
            missing_timestamps = timestamps_series.index[timestamps_series.isna()].tolist()
            if not missing_timestamps:
                self.logger.info(f"All {len(timestamps)} forecasts found in cache")
                return timestamps_series
            
            historical_data = self.deployer.pipeline.historical_data
            if historical_data is None:
                self.logger.error("No historical data available for forecasting")
                return None
            
            if isinstance(historical_data, pd.Series):
                historical_data = historical_data.to_frame(name='Actual Load')
            last_actual_load = historical_data['Actual Load'].iloc[-1] if 'Actual Load' in historical_data.columns else None
            
            self.logger.info(f"Generating predictions for {len(missing_timestamps)} missing timestamps")
            try:
                predictions = self.deployer.make_prediction(missing_timestamps)
                if predictions is None:
                    raise ValueError("Prediction returned None")
                
                if isinstance(predictions, pd.DataFrame):
                    pred_series = pd.Series(predictions.iloc[:, 0].values, index=predictions.index)
                elif isinstance(predictions, pd.Series):
                    pred_series = predictions
                else:
                    pred_series = pd.Series(predictions, index=missing_timestamps[:len(predictions)])
                
                # Log how many predictions we got vs how many we requested
                self.logger.info(f"Received {len(pred_series)} predictions for {len(missing_timestamps)} requested timestamps")
                
                # Check if we're missing any predictions
                if len(pred_series) < len(missing_timestamps):
                    self.logger.warning(f"Missing {len(missing_timestamps) - len(pred_series)} predictions")
                    
                    # For any missing predictions, use a simple time-of-day and day-of-week based model
                    missing_after_prediction = [ts for ts in missing_timestamps if ts not in pred_series.index]
                    
                    if missing_after_prediction and last_actual_load is not None:
                        self.logger.info(f"Generating simple predictions for {len(missing_after_prediction)} remaining timestamps")
                        for ts in missing_after_prediction:
                            hour = ts.hour
                            weekday = ts.weekday()
                            tod_factor = 1.0 + 0.2 * np.sin(np.pi * (hour - 6) / 12)
                            day_factor = 0.85 if weekday >= 5 else 1.0
                            simple_pred = last_actual_load * tod_factor * day_factor
                            pred_series[ts] = simple_pred
                
                # Apply reasonableness checks to all predictions
                for ts in pred_series.index:
                    pred = pred_series[ts]
                    if not 20000 <= pred <= 80000 and last_actual_load is not None:
                        self.logger.warning(f"Prediction {pred} for {ts} outside reasonable range")
                        hour = ts.hour
                        weekday = ts.weekday()
                        tod_factor = 1.0 + 0.2 * np.sin(np.pi * (hour - 6) / 12)
                        day_factor = 0.85 if weekday >= 5 else 1.0
                        pred_series[ts] = last_actual_load * tod_factor * day_factor
                        self.logger.info(f"Adjusted prediction to {pred_series[ts]} for {ts}")
                
                # Update the timestamps_series with our predictions
                timestamps_series.loc[pred_series.index] = pred_series
                
                # Check if we have all predictions now
                still_missing = timestamps_series.isna().sum()
                if still_missing > 0:
                    self.logger.warning(f"Still missing {still_missing} predictions after all attempts")
                    
                    # Fill any remaining missing values with interpolation
                    timestamps_series = timestamps_series.interpolate(method='time').ffill().bfill()
                
                # Update the cache
                self.predictions_cache = pd.concat([
                    self.predictions_cache,
                    timestamps_series.to_frame()
                ]).drop_duplicates()
                
                self.logger.info(f"Final forecast has {len(timestamps_series)} points")
                return timestamps_series
            
            except Exception as e:
                self.logger.error(f"Batch prediction failed: {str(e)}")
                import traceback
                self.logger.debug(f"Traceback: {traceback.format_exc()}")
                
                if last_actual_load is not None:
                    self.logger.warning("Falling back to historical pattern")
                    historical_pattern = historical_data['Actual Load'].tail(96)  # Last 24 hours
                    historical_pattern = historical_pattern.values
                    for i, ts in enumerate(missing_timestamps):
                        idx = i % len(historical_pattern)
                        timestamps_series[ts] = historical_pattern[idx]
                    self.predictions_cache = pd.concat([
                        self.predictions_cache,
                        timestamps_series.to_frame()
                    ]).drop_duplicates()
                    return timestamps_series
                else:
                    self.logger.error("No fallback available due to missing actual load")
                    return None
        
        except Exception as e:
            self.logger.error(f"Error in get_forecast: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def get_weather_forecast(self, hours=24):
        """Get weather forecast for the next N hours."""
        if not self.initialized:
            self.initialize()
        
        # Use pandas Timestamp instead of datetime
        now = pd.Timestamp.now(tz='Europe/Berlin').replace(second=0, microsecond=0)
        now = now.replace(minute=15 * (now.minute // 15))
        
        weather_data = self.deployer.pipeline.get_weather_data(
            start_time=now,
            end_time=now + pd.Timedelta(hours=hours)
        )
        
        return weather_data