import os
import pickle
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

from src.deployment.pipeline import DataPipeline
from src.model.feature_extractor import LoadFeatureExtractor

class ModelDeployer:
    """Handles model deployment and real-time predictions with enhanced feature handling."""
    
    def __init__(self, model_path=None, config_path=None):
        """Initialize the model deployer."""
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path or "data/models/latest_model.pkl"
        self.config_path = config_path or "config/deployment_config.json"
        self.pipeline = DataPipeline()  # Initialize pipeline immediately
        self.model = None
        self.feature_extractor = None
        self.last_prediction_time = None
        
    def load_model(self):
        """Load the trained model from disk."""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                self.logger.info(f"Model loaded from {self.model_path}")
                # Log expected features for debugging
                expected_features = self._get_model_features()
                if expected_features:
                    self.logger.info(f"Model expects {len(expected_features)} features: {expected_features}")
                return True
            else:
                self.logger.warning(f"Model file not found at {self.model_path}")
                return False
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return False
    
    def initialize_pipeline(self, historical_data=None):
        """Initialize the live pipeline with historical data."""
        try:
            self.pipeline = DataPipeline()
            self.feature_extractor = LoadFeatureExtractor()  # Initialize feature extractor
            
            if historical_data is not None:
                success = self.pipeline.initialize_with_historical_data(historical_data)
            else:
                # Default to 7 days if no data provided
                end_time = datetime.now(self.pipeline.tz)
                start_time = end_time - timedelta(days=7)
                historical_data = self.pipeline.entsoe_client.get_load_data(start_time, end_time)
                success = self.pipeline.initialize_with_historical_data(historical_data)
            
            if success:
                self.logger.info("Pipeline initialized successfully")
                return True
            else:
                self.logger.error("Failed to initialize pipeline")
                return False
        except Exception as e:
            self.logger.error(f"Error initializing pipeline: {str(e)}")
            return False
    
    def make_prediction(self, timestamp) -> Optional[pd.Series]:
        """Make predictions for a given timestamp."""
        try:
            if not isinstance(timestamp, pd.Timestamp):
                timestamp = pd.Timestamp(timestamp)
            
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize('Europe/Berlin')
            else:
                timestamp = timestamp.tz_convert('Europe/Berlin')
            
            # Get data up to the prediction timestamp
            historical_data = self.pipeline.get_historical_data()
            if historical_data is None or historical_data.empty:
                self.logger.error("No historical data available")
                return None
            
            # Extract features using the feature extractor
            features = self.feature_extractor.extract_features(historical_data)
            if features is None:
                self.logger.error("Failed to extract features")
                return None
            
            # Get expected features from model
            expected_features = self._get_model_features()
            if expected_features is None:
                self.logger.error("Could not determine expected features from model")
                return None
            
            # Get the latest available features
            latest_features = features.iloc[-1:].copy()
            
            if latest_features.empty:
                self.logger.error("No features available for requested timestamps")
                return None
            
            # Make predictions for next 24 hours (96 intervals)
            predictions = []
            current_features = latest_features.copy()
            last_actual_load = latest_features['Actual Load'].iloc[0]
            
            for i in range(96):  # 24 hours * 4 (15-min intervals)
                current_time = timestamp + pd.Timedelta(minutes=15*i)
                
                # Update time-based features
                current_features['hour'] = current_time.hour
                current_features['weekday'] = current_time.dayofweek
                current_features['month'] = current_time.month
                current_features['hour_sin'] = np.sin(2 * np.pi * current_time.hour / 24)
                current_features['hour_cos'] = np.cos(2 * np.pi * current_time.hour / 24)
                current_features['weekday_sin'] = np.sin(2 * np.pi * current_time.dayofweek / 7)
                current_features['weekday_cos'] = np.cos(2 * np.pi * current_time.dayofweek / 7)
                
                # Update load-based features
                if i > 0:
                    current_features['Actual Load'] = predictions[-1]  # Use last prediction
                    
                    if i >= 96:  # After 24 hours
                        current_features['96_periods_ago_load'] = predictions[i-96]
                    if i >= 672:  # After 7 days
                        current_features['672_periods_ago_load'] = predictions[i-672]
                    
                    # Update rolling statistics
                    recent_loads = predictions[-96:] if i >= 96 else predictions + [last_actual_load] * (96-i)
                    current_features['rolling_mean_24h'] = np.mean(recent_loads)
                    current_features['rolling_std_24h'] = np.std(recent_loads)
                    
                    recent_loads_168h = predictions[-672:] if i >= 672 else predictions + [last_actual_load] * (672-i)
                    current_features['rolling_mean_168h'] = np.mean(recent_loads_168h)
                    current_features['rolling_std_168h'] = np.std(recent_loads_168h)
                
                # Ensure all expected features are present
                for feature in expected_features:
                    if feature not in current_features:
                        current_features[feature] = 0
                
                # Make prediction using only the expected features
                pred = self.model.predict(current_features[expected_features])[0]
                predictions.append(pred)
            
            # Create prediction series
            pred_index = pd.date_range(
                start=timestamp,
                periods=96,
                freq='15min',
                tz='Europe/Berlin'
            )
            return pd.Series(predictions, index=pred_index)
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _get_model_features(self):
        """Get feature names from the model."""
        if self.model is None:
            return None
        
        try:
            # For LightGBM models
            if hasattr(self.model, 'feature_name_'):
                features = self.model.feature_name_
                self.logger.info(f"Got features from model.feature_name_: {features}")
                return features
            elif hasattr(self.model, 'booster_') and hasattr(self.model.booster_, 'feature_name_'):
                features = self.model.booster_.feature_name_
                self.logger.info(f"Got features from model.booster_.feature_name_: {features}")
                return features
            elif hasattr(self.model, 'feature_names_in_'):
                features = list(self.model.feature_names_in_)
                self.logger.info(f"Got features from model.feature_names_in_: {features}")
                return features
            
            # Default feature list if we can't get it from the model
            default_features = [
                'hour', 'weekday', 'month', 'is_weekend',
                'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos',
                'load_1h_ago', 'load_24h_ago', 'load_7d_ago',
                'rolling_mean_24h', 'rolling_std_24h',
                'Forecasted Load'
            ]
            self.logger.info(f"Using default feature list: {default_features}")
            return default_features
            
        except Exception as e:
            self.logger.warning(f"Error getting model features: {str(e)}")
            return None

    def _align_features(self, features_df, expected_features):
        """Align features to match what the model expects."""
        try:
            # Create a copy to avoid modifying the original
            aligned_df = features_df.copy()
            
            # Add missing features with appropriate default values
            for feature in expected_features:
                if feature not in aligned_df.columns:
                    if feature == 'Forecasted Load':
                        # Use mean of actual load as default forecast
                        aligned_df[feature] = aligned_df['load_1h_ago']
                    else:
                        aligned_df[feature] = 0
            
            # Select only the expected features in the right order
            return aligned_df[expected_features]
        except Exception as e:
            self.logger.error(f"Error aligning features: {str(e)}")
            return None
    
    def save_model(self, model):
        """Save a new model to disk."""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            with open(self.model_path, 'wb') as f:
                pickle.dump(model, f)
            self.model = model
            self.logger.info(f"Model saved to {self.model_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            return False

    def _debug_feature_mismatch(self, features_df, expected_features):
        """Debug helper to identify feature mismatches."""
        current_features = set(features_df.columns)
        expected_features_set = set(expected_features)
        
        self.logger.info("=== Feature Debug Information ===")
        self.logger.info(f"Number of current features: {len(current_features)}")
        self.logger.info(f"Number of expected features: {len(expected_features_set)}")
        
        # Find extra and missing features
        extra_features = current_features - expected_features_set
        missing_features = expected_features_set - current_features
        
        if extra_features:
            self.logger.info(f"Extra features: {sorted(extra_features)}")
        if missing_features:
            self.logger.info(f"Missing features: {sorted(missing_features)}")
        
        self.logger.info("Current features:")
        for feat in sorted(current_features):
            self.logger.info(f"  - {feat}")
        
        self.logger.info("Expected features:")
        for feat in sorted(expected_features_set):
            self.logger.info(f"  - {feat}")