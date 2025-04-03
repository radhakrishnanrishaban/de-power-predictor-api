import pandas as pd
import pytz
from datetime import datetime
from pathlib import Path
import logging
from src.data.clients.entsoe_client import EntsoeClient

logger = logging.getLogger(__name__)

class TrainingDataUpdater:
    def __init__(self, data_dir="data/raw/train_data"):
        self.data_dir = Path(data_dir)
        self.client = EntsoeClient()
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def get_latest_data_file(self):
        """Find the most recent data file in the directory."""
        data_files = list(self.data_dir.glob("load_data_*.csv"))
        if not data_files:
            raise FileNotFoundError("No training data files found")
        return max(data_files, key=lambda x: x.stat().st_mtime)

    def update_training_data(self):
        """Update training data with latest available data."""
        try:
            # Load existing data
            latest_file = self.get_latest_data_file()
            logger.info(f"Loading existing data from {latest_file}")
            
            existing_data = pd.read_csv(latest_file)
            existing_data['timestamp'] = pd.to_datetime(existing_data['timestamp'])
            
            # Get new data
            last_date = existing_data['timestamp'].max()
            current_date = datetime.now(pytz.timezone('Europe/Berlin'))
            
            logger.info(f"Fetching new data from {last_date} to {current_date}")
            new_data = self.client.get_load_data(
                start_time=last_date,
                end_time=current_date
            )
            
            # Combine data
            updated_data = pd.concat([existing_data, new_data])
            
            # Keep only last 3 years
            three_years_ago = current_date - pd.DateOffset(years=3)
            updated_data = updated_data[updated_data['timestamp'] >= three_years_ago]
            
            # Save with new date range
            start_date = updated_data['timestamp'].min().strftime('%Y%m%d')
            end_date = updated_data['timestamp'].max().strftime('%Y%m%d')
            new_filename = f"load_data_{start_date}_{end_date}.csv"
            new_filepath = self.data_dir / new_filename
            
            updated_data.to_csv(new_filepath, index=False)
            logger.info(f"Updated data saved to {new_filepath}")
            
            return new_filepath
            
        except Exception as e:
            logger.error(f"Error updating training data: {str(e)}")
            raise 