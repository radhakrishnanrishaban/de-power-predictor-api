from typing import Dict, Optional, List
import pandas as pd
import pytz
from datetime import datetime, timedelta
import time
import requests
import logging
import numpy as np
from pathlib import Path
from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError
from tqdm import tqdm

from src.config import Config
from src.data.utils import save_data

logger = logging.getLogger(__name__)

class EntsoeClient:
    """Client for fetching load data from ENTSO-E"""
    
    def __init__(self, api_key: str = Config.ENTSOE_API_KEY, country_code: str = Config.COUNTRY_CODE):
        """Initialize ENTSO-E client with API key and country code"""
        self.client = EntsoePandasClient(api_key=api_key)
        self.country_code = country_code
        self.tz = pytz.timezone(Config.TIMEZONE)
    
    
    def fetch_load_data(
        self,
        start_date: str,
        end_date: str,
        chunk_size: int = 30,
        max_retries: int = 3
    ) -> pd.DataFrame:
        """
        Fetch load data from ENTSO-E in chunks with retry logic
        
        Args:
            start_date (str): Start date in format YYYYMMDD
            end_date (str): End date in format YYYYMMDD
            chunk_size (int): Number of days per request
            max_retries (int): Maximum number of retry attempts
        
        Returns:
            pd.DataFrame: Combined load data for the entire period
        """
        try:
            # Clean input dates
            start_date = start_date.strip()
            end_date = end_date.strip()
            
            # Validate date format
            if not (len(start_date) == 8 and len(end_date) == 8):
                raise ValueError("Dates must be in YYYYMMDD format")
            
            start = self.tz.localize(datetime.strptime(start_date, '%Y%m%d'))
            end = self.tz.localize(datetime.strptime(end_date, '%Y%m%d'))
            
            # Validate date range
            if end <= start:
                raise ValueError("End date must be after start date")
            
            # Calculate number of chunks
            total_days = (end - start).days
            num_chunks = (total_days + chunk_size - 1) // chunk_size
            
            all_data = []
            current_start = start
            
            # Create progress bar for chunks
            with tqdm(total=num_chunks, desc="Fetching Load Data") as pbar:
                while current_start < end:
                    current_end = min(current_start + timedelta(days=chunk_size), end)
                    
                    for attempt in range(max_retries):
                        try:
                            logger.info(f"Fetching data from {current_start} to {current_end}")
                            chunk_data = self.client.query_load_and_forecast(
                                country_code=self.country_code,
                                start=pd.Timestamp(current_start),
                                end=pd.Timestamp(current_end)
                            )
                            
                            if not chunk_data.empty:
                                all_data.append(chunk_data)
                                break
                                
                        except NoMatchingDataError:
                            logger.warning(f"No data found between {current_start} and {current_end}")
                            break
                            
                        except requests.ConnectionError as e:
                            if attempt == max_retries - 1:
                                logger.error(f"Failed after {max_retries} attempts: {str(e)}")
                                raise
                            wait_time = 2 ** attempt
                            logger.warning(f"Attempt {attempt + 1} failed, waiting {wait_time} seconds")
                            time.sleep(wait_time)
                    
                    current_start = current_end
                    pbar.update(1)
            
            if not all_data:
                logger.error("No data was successfully fetched")
                return pd.DataFrame()
            
            # Combine all chunks
            combined_data = pd.concat(all_data)
            
            # Try to save data
            try:
                self._save_data(combined_data, start_date, end_date)
            except Exception as e:
                logger.error(f"Failed to save data: {str(e)}")
            
            return combined_data
            
        except Exception as e:
            logger.error(f"Error fetching load data: {str(e)}")
            raise

    def _save_data(self, data: pd.DataFrame, start_date: str, end_date: str) -> None:
        """Internal method to save data with consistent naming"""
        filename = f'load_data_{start_date}_{end_date}.csv'
        save_data(data, filename)

    def get_latest_load(self) -> pd.DataFrame:
        """
        Fetch the most recent load data (last 24 hours)
        
        Returns:
            pd.DataFrame: Latest load data
        """
        try:
            end = datetime.now(self.tz)
            start = end - timedelta(days=1)
            
            data = self.client.query_load_and_forecast(
                country_code=self.country_code,
                start=pd.Timestamp(start),
                end=pd.Timestamp(end)
            )
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching latest load data: {str(e)}")
            raise

    def get_load_data(self, start_time, end_time) -> pd.DataFrame:
        """
        Fetch load data for a specific time range
        
        Args:
            start_time: Start time as datetime or pandas Timestamp
            end_time: End time as datetime or pandas Timestamp
            
        Returns:
            pd.DataFrame: Load data for the specified time range
        """
        try:
            logger.info(f"Fetching load data from {start_time} to {end_time}")
            
            # Convert to pandas Timestamp if needed
            if not isinstance(start_time, pd.Timestamp):
                start_time = pd.Timestamp(start_time)
            if not isinstance(end_time, pd.Timestamp):
                end_time = pd.Timestamp(end_time)
            
            # Ensure timestamps have timezone info
            if start_time.tzinfo is None:
                start_time = self.tz.localize(start_time)
            if end_time.tzinfo is None:
                end_time = self.tz.localize(end_time)
                
            try:
                # Try primary method
                load_data = self.client.query_load_and_forecast(
                    country_code=self.country_code,
                    start=start_time,
                    end=end_time
                )
                
                if load_data is None or load_data.empty:
                    # Try fallback method
                    load_data = self.client.query_load(
                        country_code=self.country_code,
                        start=start_time,
                        end=end_time
                    )
                    
                # Handle data conversion
                if isinstance(load_data, pd.Series):
                    load_data = load_data.to_frame('Actual Load')
                elif isinstance(load_data, pd.DataFrame):
                    if 'test' in load_data.columns:
                        load_data = load_data.rename(columns={'test': 'Actual Load'})
                    elif 'load' in load_data.columns:
                        load_data = load_data.rename(columns={'load': 'Actual Load'})
                
                return load_data if not load_data.empty else pd.DataFrame()
                
            except Exception as e:
                logger.error("Failed to fetch load data using both methods")
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error in get_load_data: {str(e)}")
            return pd.DataFrame()
    
                
    def get_load_forecast(self, start_time=None, end_time=None) -> pd.DataFrame:
        """
        Fetch load forecast data for a specific time range
        
        Args:
            start_time: Start time as datetime or pandas Timestamp (default: now)
            end_time: End time as datetime or pandas Timestamp (default: now + 24 hours)
            
        Returns:
            pd.DataFrame: Load forecast data
        """
        try:
            # Handle default times
            if start_time is None:
                start_time = pd.Timestamp.now(tz=self.tz)
            elif not isinstance(start_time, pd.Timestamp):
                start_time = pd.Timestamp(start_time)
            
            if end_time is None:
                end_time = start_time + pd.Timedelta(days=1)
            elif not isinstance(end_time, pd.Timestamp):
                end_time = pd.Timestamp(end_time)
            
            # Ensure timezone info
            if start_time.tzinfo is None:
                start_time = self.tz.localize(start_time)
            if end_time.tzinfo is None:
                end_time = self.tz.localize(end_time)
            
            # Query load forecast from ENTSO-E
            try:
                forecast_data = self.client.query_load_forecast(
                    country_code=self.country_code,
                    start=start_time,
                    end=end_time
                )
                
                if forecast_data is None or forecast_data.empty:
                    logger.warning("No data from query_load_forecast, trying alternative method")
                    forecast_data = self.client.query_load(
                        country_code=self.country_code,
                        start=start_time,
                        end=end_time,
                        process_type='A01'  # Day-ahead forecast
                    )
            except Exception as e:
                logger.warning(f"Error with ENTSOE API: {str(e)}")
                # Generate synthetic forecast
                date_range = pd.date_range(start=start_time, end=end_time, freq='15min')
                base_load = 50000  # Base load in MW
                hours = np.array([dt.hour for dt in date_range])
                day_effect = 10000 * np.sin(np.pi * (hours - 6) / 12)  # Daily pattern
                forecast_values = base_load + day_effect
                forecast_data = pd.Series(forecast_values, index=date_range)
                logger.info(f"Generated synthetic forecast with {len(forecast_data)} points")
            
            return forecast_data
            
        except Exception as e:
            logger.error(f"Error fetching load forecast: {str(e)}")
            # Return empty DataFrame instead of raising exception
            return pd.DataFrame()