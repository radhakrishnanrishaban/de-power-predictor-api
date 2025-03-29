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
    
    def make_prediction(self, timestamps) -> Optional[pd.DataFrame]:
        """Make predictions for a list of timestamps with improved feature handling."""
        try:
            # Ensure timestamps is a list
            if not isinstance(timestamps, (list, pd.Index, pd.Series)):
                timestamps = [timestamps]
            
            # Convert all timestamps to pd.Timestamp with correct timezone
            berlin_tz = pytz.timezone('Europe/Berlin')
            timestamps = [
                pd.Timestamp(ts).tz_localize(berlin_tz) if ts.tzinfo is None else pd.Timestamp(ts).tz_convert(berlin_tz)
                for ts in timestamps
            ]
            
            # Get historical data
            historical_data = self.pipeline.get_historical_data()
            if historical_data is None or historical_data.empty:
                self.logger.error("No historical data available")
                return None
            
            # Create prediction DataFrame with timestamps
            pred_df = pd.DataFrame(index=pd.Index(timestamps))
            pred_df['Actual Load'] = np.nan  # Initialize with NaN
            
            # Combine historical and prediction data
            combined_df = pd.concat([historical_data, pred_df]).sort_index()
            combined_df = combined_df[~combined_df.index.duplicated(keep='first')]  # Remove any duplicates
            
            # Extract features using LoadFeatureExtractor
            features = self.feature_extractor.extract_features(combined_df)
            if features is None or features.empty:
                self.logger.error("Failed to extract features")
                return None
            
            # Get features only for prediction timestamps
            pred_features = features.loc[timestamps]
            
            # Get expected features from the model
            expected_features = self._get_model_features()
            if expected_features is None:
                self.logger.error("Could not determine expected features from model")
                return None
            
            # Ensure features are aligned with model expectations
            pred_features = self._align_features(pred_features, expected_features)
            if pred_features is None:
                self.logger.error("Failed to align features")
                return None
            
            # Log feature shapes for debugging
            self.logger.info(f"Number of timestamps: {len(timestamps)}")
            self.logger.info(f"Shape of prediction features: {pred_features.shape}")
            
            # Make predictions
            predictions = self.model.predict(pred_features)
            self.logger.info(f"Number of predictions: {len(predictions)}")
            
            # Create predictions series with matching index
            predictions_series = pd.Series(predictions, index=pred_features.index, name='predicted_load')
            
            # Add confidence intervals
            baseline_stats = self._calculate_baseline_stats(historical_data)
            return self._add_confidence_intervals(predictions_series, baseline_stats)
        
        except Exception as e:
            self.logger.error(f"Error making prediction: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
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
                
    def _create_feature_row(self, timestamp, latest_features, historical_data, baseline_stats, initial_pred=None):
        """Create a feature row for prediction with improved future handling."""
        try:
            feature_row = latest_features.copy()
            
            # Basic time features (these are always available)
            feature_row['hour'] = timestamp.hour
            feature_row['weekday'] = timestamp.dayofweek
            feature_row['month'] = timestamp.month
            feature_row['hour_sin'] = np.sin(2 * np.pi * timestamp.hour / 24)
            feature_row['hour_cos'] = np.cos(2 * np.pi * timestamp.hour / 24)
            feature_row['weekday_sin'] = np.sin(2 * np.pi * timestamp.dayofweek / 7)
            feature_row['weekday_cos'] = np.cos(2 * np.pi * timestamp.dayofweek / 7)
            
            # Get holidays and check if day before holiday
            holidays = self.pipeline.get_holidays()
            day_before = timestamp - pd.Timedelta(days=1)
            is_day_before_holiday = 1 if day_before.date() in holidays else 0
            feature_row['is_day_before_holiday'] = is_day_before_holiday

            # Get the last valid timestamp from historical data
            last_valid_time = historical_data.index.max()
            
            # For future timestamps, use a rolling window of predictions
            if timestamp > last_valid_time:
                # Use pattern-based estimation for Actual Load
                hour_stats = baseline_stats['hourly_patterns'][timestamp.hour]
                day_stats = baseline_stats['daily_patterns'][timestamp.dayofweek]
                
                # Calculate estimated load using patterns
                estimated_load = (
                    hour_stats['mean'] * 0.4 +     # Hourly pattern
                    day_stats['mean'] * 0.3 +      # Daily pattern
                    historical_data['Actual Load'].tail(96).mean() * 0.2 +  # Recent trend
                    baseline_stats['overall_mean'] * 0.1  # Overall baseline
                )
                
                # Adjust for time of day and holidays
                if 8 <= timestamp.hour <= 18:
                    estimated_load *= 1.1
                elif 0 <= timestamp.hour <= 5:
                    estimated_load *= 0.9
                if is_day_before_holiday:
                    estimated_load *= 0.95
                    
                feature_row['Actual Load'] = estimated_load
                
                # Handle lagged features
                one_day_ago = timestamp - pd.Timedelta(days=1)
                week_ago = timestamp - pd.Timedelta(days=7)
                
                feature_row['96_periods_ago_load'] = self._get_historical_or_estimated_load(
                    one_day_ago, historical_data, baseline_stats
                )
                feature_row['672_periods_ago_load'] = self._get_historical_or_estimated_load(
                    week_ago, historical_data, baseline_stats
                )
                
                # Calculate rolling statistics using available data and estimates
                feature_row['rolling_mean_24h'] = self._estimate_rolling_mean(
                    timestamp, historical_data, baseline_stats, hours=24
                )
                feature_row['rolling_mean_168h'] = self._estimate_rolling_mean(
                    timestamp, historical_data, baseline_stats, hours=168
                )
                feature_row['rolling_std_24h'] = hour_stats['std']
                feature_row['rolling_std_168h'] = baseline_stats['weekly_std']
            else:
                # For historical timestamps, use actual values where available
                feature_row['Actual Load'] = historical_data.loc[timestamp, 'Actual Load']
                feature_row['96_periods_ago_load'] = historical_data.loc[
                    timestamp - pd.Timedelta(days=1), 'Actual Load'
                ]
                feature_row['672_periods_ago_load'] = historical_data.loc[
                    timestamp - pd.Timedelta(days=7), 'Actual Load'
                ]
                # Calculate rolling statistics from historical data
                past_24h = historical_data.loc[:timestamp].last('24H')['Actual Load']
                past_168h = historical_data.loc[:timestamp].last('168H')['Actual Load']
                
                feature_row['rolling_mean_24h'] = past_24h.mean()
                feature_row['rolling_mean_168h'] = past_168h.mean()
                feature_row['rolling_std_24h'] = past_24h.std()
                feature_row['rolling_std_168h'] = past_168h.std()
            
            # Check for any remaining NaN values and fill with pattern-based estimates
            for col in feature_row.columns:
                if pd.isna(feature_row[col].iloc[0]):
                    if col in ['rolling_mean_24h', 'rolling_mean_168h']:
                        feature_row[col] = baseline_stats['overall_mean']
                    elif col in ['rolling_std_24h', 'rolling_std_168h']:
                        feature_row[col] = baseline_stats['weekly_std']
                    elif col in ['96_periods_ago_load', '672_periods_ago_load']:
                        feature_row[col] = baseline_stats['hourly_patterns'][timestamp.hour]['mean']
            
            return feature_row
            
        except Exception as e:
            self.logger.error(f"Error creating feature row for {timestamp}: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None
    
    def _get_historical_or_estimated_load(self, timestamp, historical_data, baseline_stats):
        """Get historical load if available, otherwise estimate it."""
        try:
            # Check if timestamp exists in historical data and has a valid value
            if timestamp in historical_data.index and not pd.isna(historical_data.loc[timestamp, 'Actual Load']):
                return historical_data.loc[timestamp, 'Actual Load']
            
            # If data is missing, use a more sophisticated estimation approach
            hour_stats = baseline_stats['hourly_patterns'][timestamp.hour]
            day_stats = baseline_stats['daily_patterns'][timestamp.dayofweek]
            
            # Try to get the most recent valid load value
            recent_loads = historical_data['Actual Load'].dropna()
            recent_mean = recent_loads.tail(96).mean() if not recent_loads.empty else baseline_stats['overall_mean']
            
            # Combine different signals for estimation
            estimated_load = (
                hour_stats['mean'] * 0.4 +     # Hourly pattern
                day_stats['mean'] * 0.3 +      # Daily pattern
                recent_mean * 0.2 +            # Recent trend
                baseline_stats['overall_mean'] * 0.1  # Overall baseline
            )
            
            # Adjust for time of day
            if 8 <= timestamp.hour <= 18:  # Peak hours
                estimated_load *= 1.1
            elif 0 <= timestamp.hour <= 5:  # Night hours
                estimated_load *= 0.9
            
            # Check if the estimation seems reasonable
            if estimated_load < 0:
                self.logger.warning(f"Negative load estimation for {timestamp}, using fallback")
                estimated_load = baseline_stats['overall_mean']
            
            if timestamp not in historical_data.index:
                self.logger.debug(
                    f"Estimated load for {timestamp}: {estimated_load:.2f} MW "
                    f"(hour_mean: {hour_stats['mean']:.2f}, day_mean: {day_stats['mean']:.2f}, "
                    f"recent_mean: {recent_mean:.2f}, overall_mean: {baseline_stats['overall_mean']:.2f})"
                )
            
            return estimated_load
        
        except Exception as e:
            self.logger.error(f"Error in _get_historical_or_estimated_load for {timestamp}: {str(e)}")
            # Fallback to overall mean if everything else fails
            return baseline_stats['overall_mean']
    
    def _estimate_rolling_mean(self, timestamp, historical_data, baseline_stats, hours):
        """Estimate rolling mean using available data and patterns with improved handling."""
        try:
            # For 168h mean, use a combination of available data and patterns if we don't have full history
            if hours == 168:  # 7 days
                # Get available historical data up to the timestamp
                available_data = historical_data[historical_data.index <= timestamp]['Actual Load']
                if len(available_data) >= 24 * 4:  # At least 24 hours of data
                    # Use a weighted combination of available data mean and pattern-based estimation
                    recent_mean = available_data.mean()
                    day_stats = baseline_stats['daily_patterns'][timestamp.dayofweek]
                    hour_stats = baseline_stats['hourly_patterns'][timestamp.hour]
                    
                    weighted_mean = (
                        recent_mean * 0.4 +                    # Recent data
                        day_stats['mean'] * 0.3 +             # Day of week pattern
                        hour_stats['mean'] * 0.2 +            # Hour of day pattern
                        baseline_stats['overall_mean'] * 0.1   # Overall pattern
                    )
                    return weighted_mean
                else:
                    # If we have very limited data, use pattern-based estimation
                    return (
                        baseline_stats['daily_patterns'][timestamp.dayofweek]['mean'] * 0.5 +
                        baseline_stats['hourly_patterns'][timestamp.hour]['mean'] * 0.3 +
                        baseline_stats['overall_mean'] * 0.2
                    )
            
            # For shorter windows (e.g., 24h), use the original logic with improvements
            relevant_times = pd.date_range(
                end=timestamp,
                periods=hours * 4,  # 15-minute intervals
                freq='15min'
            )
            
            values = []
            weights = []
            
            for ts in relevant_times:
                if ts in historical_data.index and not pd.isna(historical_data.loc[ts, 'Actual Load']):
                    # Actual historical value
                    values.append(historical_data.loc[ts, 'Actual Load'])
                    weights.append(1.0)  # Full weight for actual values
                else:
                    # Estimate missing values
                    hour_stats = baseline_stats['hourly_patterns'][ts.hour]
                    day_stats = baseline_stats['daily_patterns'][ts.dayofweek]
                    
                    estimated = (
                        hour_stats['mean'] * 0.4 +
                        day_stats['mean'] * 0.4 +
                        baseline_stats['overall_mean'] * 0.2
                    )
                    values.append(estimated)
                    weights.append(0.7)  # Lower weight for estimated values
            
            if not values:
                return baseline_stats['overall_mean']
            
            # Apply exponential decay to weights
            decay_factor = np.exp(-np.arange(len(values)) / (len(values) / 2))
            final_weights = np.array(weights) * decay_factor
            
            # Calculate weighted average
            weighted_mean = np.average(values, weights=final_weights)
            
            return weighted_mean
            
        except Exception as e:
            self.logger.error(f"Error in _estimate_rolling_mean for {timestamp}: {str(e)}")
            self.logger.error(traceback.format_exc())
            return baseline_stats['overall_mean']
    
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