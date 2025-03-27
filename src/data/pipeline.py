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
            
            # Extract features
            features = self.feature_extractor.transform(processed_data)
            
            # Make prediction
            predictions = self.model.predict(features)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise