import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple
import pytz

from src.data.clients.entsoe_client import EntsoeClient
from src.data.preprocess.preprocessor import LoadDataPreprocessor
from src.model.feature_extractor import LoadFeatureExtractor
from src.model.predictor import LoadPredictor

logger = logging.getLogger(__name__)

class DataPipeline:
    """End-to-end pipeline for load prediction"""
    
    def __init__(self):
        self.entsoe_client = EntsoeClient()
        self.preprocessor = LoadDataPreprocessor()
        self.feature_extractor = LoadFeatureExtractor()
        self.model = LoadPredictor()
        self.tz = pytz.timezone('Europe/Berlin')
        self.historical_data = None
        
    def initialize_with_historical_data(self, days_or_data=7) -> bool:
        """Initialize the pipeline with historical data.
        
        Args:
            days_or_data: Either number of days to fetch or actual DataFrame to use
        """
        try:
            if isinstance(days_or_data, pd.DataFrame):
                # Use provided data directly
                raw_data = days_or_data
            else:
                # Fetch historical data for specified days
                end_time = datetime.now(self.tz)
                start_time = end_time - timedelta(days=days_or_data)
                raw_data = self.entsoe_client.get_load_data(start_time, end_time)
            
            if raw_data.empty:
                logger.error("No historical data available")
                return False
            
            # Preprocess data
            processed_data = self.preprocessor.preprocess(raw_data)
            if processed_data is None or processed_data.empty:
                logger.error("Failed to preprocess historical data")
                return False
            
            # Store historical data
            self.historical_data = processed_data
            
            # Initialize preprocessor with historical data
            self.preprocessor.initialize_with_historical(processed_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing with historical data: {str(e)}")
            return False
    
    def process_new_data(self, current_time: Optional[datetime] = None) -> pd.DataFrame:
        """Process new data for prediction."""
        try:
            if current_time is None:
                current_time = datetime.now(self.tz)
            
            # For future timestamps, use historical data and forecast
            if current_time > datetime.now(self.tz):
                if self.historical_data is None:
                    raise ValueError("No historical data available for future prediction")
                return self.historical_data
            
            # Get latest data
            start_time = current_time - timedelta(days=1)
            raw_data = self.entsoe_client.get_load_data(start_time, current_time)
            
            if raw_data.empty:
                raise ValueError("No new data available")
            
            # Preprocess data
            processed_data = self.preprocessor.preprocess_live(raw_data)
            if processed_data is None:
                raise ValueError("Failed to preprocess new data")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing new data: {str(e)}")
            raise
    
    def get_historical_data(self) -> pd.DataFrame:
        """Get the historical data."""
        return self.historical_data if self.historical_data is not None else pd.DataFrame()
    
    def prepare_training_data(
        self,
        start_date: str,
        end_date: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for training"""
        try:
            # Fetch raw data
            raw_data = self.entsoe_client.fetch_load_data(start_date, end_date)
            if raw_data.empty:
                raise ValueError("No data fetched from ENTSOE")
            
            # Preprocess data
            processed_data = self.preprocessor.preprocess(raw_data)
            
            # Extract features
            features = self.feature_extractor.transform(processed_data)
            
            # Prepare target (24h ahead load)
            target = processed_data['Actual Load'].shift(-96)  # 96 quarters = 24 hours
            
            # Remove rows with NaN target
            features = features[:-96]  # Remove last 24 hours
            target = target[:-96]
            
            return features, target
            
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            raise
    
    def train_model(
        self,
        start_date: str,
        end_date: str,
        model_path: Optional[str] = None
    ) -> None:
        """Train the model with data from start_date to end_date"""
        try:
            # Prepare training data
            features, target = self.prepare_training_data(start_date, end_date)
            
            # Train model
            self.model.train(features, target)
            
            # Save model if path provided
            if model_path:
                self.model.save(model_path)
                
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def make_prediction(
        self,
        current_time: Optional[datetime] = None
    ) -> pd.Series:
        """Make prediction for the next 24 hours"""
        try:
            if current_time is None:
                current_time = datetime.now(self.tz)
            
            # Get historical data for feature creation
            start_time = current_time - timedelta(days=7)
            raw_data = self.entsoe_client.get_load_data(start_time, current_time)
            
            if raw_data.empty:
                raise ValueError("No historical data available")
            
            # Preprocess data
            processed_data = self.preprocessor.preprocess(raw_data)
            
            # Extract features - using extract_features instead of transform
            features = self.feature_extractor.extract_features(processed_data)
            if features is None:
                raise ValueError("Failed to extract features")
            
            # Make prediction
            predictions = self.model.predict(features)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise