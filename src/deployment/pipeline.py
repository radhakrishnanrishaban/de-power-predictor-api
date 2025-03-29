import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple
import pytz
import holidays

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
        self.historical_data = None  # To store historical data for deployment
    
    def initialize_with_historical_data(self, historical_data: pd.DataFrame) -> bool:
        """Initialize the pipeline with historical data for deployment."""
        try:
            self.historical_data = historical_data
            return True
        except Exception as e:
            logger.error(f"Error initializing with historical data: {str(e)}")
            return False
    
    def get_historical_data(self) -> Optional[pd.DataFrame]:
        """Return the historical data stored in the pipeline."""
        return self.historical_data
    
    def get_holidays(self, start_year: Optional[int] = None, end_year: Optional[int] = None) -> list:
        """Fetch German holidays for the given year range."""
        try:
            # Default to the current year and the next year if not specified
            if start_year is None:
                start_year = datetime.now().year - 1
            if end_year is None:
                end_year = datetime.now().year + 1

            # Initialize German holidays
            de_holidays = holidays.Germany(years=range(start_year, end_year + 1))

            # Convert holiday dates to a list of dates
            holiday_dates = [date for date, _ in de_holidays.items()]
            return holiday_dates
        except Exception as e:
            logger.error(f"Error fetching holidays: {str(e)}")
            return []
    
    
    def prepare_training_data(
        self,
        start_date: str,
        end_date: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for training"""
        try:

            # Validate date format
            start = pd.to_datetime(start_date, errors='raise')
            end = pd.to_datetime(end_date, errors='raise')
            if start >= end:
                raise ValueError("start_date must be before end_date")
        
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
            
            # Extract features
            features = self.feature_extractor.transform(processed_data)
            
            # Make prediction
            predictions = self.model.predict(features)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise