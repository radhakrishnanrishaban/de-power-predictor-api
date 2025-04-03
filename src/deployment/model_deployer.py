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
                # Add debug information
                self.logger.info(f"Model file size: {os.path.getsize(self.model_path)} bytes")
                self.logger.info(f"Model file last modified: {datetime.fromtimestamp(os.path.getmtime(self.model_path))}")
                
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
    
    def make_prediction(self, features: pd.DataFrame) -> pd.DataFrame:
        """Make predictions using the loaded model."""
        try:
            # Ensure features match model expectations
            expected_features = self.model.feature_name_
            missing_features = set(expected_features) - set(features.columns)
            if missing_features:
                self.logger.error(f"Missing features: {missing_features}")
                return pd.DataFrame()

            # Make predictions
            predictions = pd.DataFrame(index=features.index)
            predictions['predicted_load'] = self.model.predict(features[expected_features])
            
            # Add confidence intervals (simple ±5% for now)
            predictions['lower_bound'] = predictions['predicted_load'] * 0.95
            predictions['upper_bound'] = predictions['predicted_load'] * 1.05
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Prediction error: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()
    
    def _calculate_baseline_stats(self, historical_data):
        """Calculate baseline statistics for feature estimation."""
        try:
            stats = {
                'hourly_patterns': {},
                'daily_patterns': {},
                'weekly_std': historical_data['Actual Load'].std(),
                'overall_mean': historical_data['Actual Load'].mean()
            }
            
            # Calculate hourly patterns
            for hour in range(24):
                hourly_data = historical_data[historical_data.index.hour == hour]
                stats['hourly_patterns'][hour] = {
                    'mean': hourly_data['Actual Load'].mean(),
                    'std': hourly_data['Actual Load'].std()
                }
            
            # Calculate daily patterns
            for day in range(7):
                daily_data = historical_data[historical_data.index.dayofweek == day]
                stats['daily_patterns'][day] = {
                    'mean': daily_data['Actual Load'].mean(),
                    'std': daily_data['Actual Load'].std()
                }
            
            return stats
        except Exception as e:
            self.logger.error(f"Error calculating baseline stats: {str(e)}")
            return None
                
    def _add_confidence_intervals(self, predictions, baseline_stats):
        predictions_df = predictions.to_frame('predicted_load')
        
        # Calculate confidence intervals based on hourly patterns
        predictions_df['lower_bound'] = predictions_df['predicted_load']
        predictions_df['upper_bound'] = predictions_df['predicted_load']
        
        for idx in predictions_df.index:
            hour = idx.hour
            hour_std = baseline_stats['hourly_patterns'][hour]['std']
            predictions_df.loc[idx, 'lower_bound'] = predictions_df.loc[idx, 'predicted_load'] - 2 * hour_std
            predictions_df.loc[idx, 'upper_bound'] = predictions_df.loc[idx, 'predicted_load'] + 2 * hour_std
        
        return predictions_df
    
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
            # Ensure features_df is a DataFrame
            if isinstance(features_df, pd.Series):
                features_df = features_df.to_frame().T
            elif not isinstance(features_df, pd.DataFrame):
                raise ValueError(f"Expected a DataFrame or Series, got {type(features_df)}")
            
            # Create a copy to avoid modifying the original
            aligned_df = features_df.copy()
            
            # Add missing features with appropriate default values
            for feature in expected_features:
                if feature not in aligned_df.columns:
                    if feature == 'Forecasted Load':
                        # Use mean of actual load as default forecast
                        aligned_df[feature] = aligned_df.get('load_1h_ago', aligned_df['Actual Load'])
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