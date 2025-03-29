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
import traceback

from src.config import Config
from src.data.utils import storage

logger = logging.getLogger(__name__)

class EntsoeClient:
    """Client for fetching load data from ENTSO-E"""
    
    def __init__(self, api_key: str = Config.ENTSOE_API_KEY, country_code: str = Config.COUNTRY_CODE):
        """Initialize ENTSO-E client with API key and country code"""
        self.client = EntsoePandasClient(api_key=api_key)
        self.country_code = country_code
        self.tz = pytz.timezone(Config.TIMEZONE)
        self.logger = logging.getLogger(__name__)
    
    
    def fetch_load_data(
        self,
        start_date: str,
        end_date: str,
        chunk_size: int = 30,
        max_retries: int = 3
    ) -> pd.DataFrame:
        """Fetch load data from ENTSO-E in chunks with retry logic"""
        # Validate dates before try-except block
        start_date = start_date.strip()
        end_date = end_date.strip()
        
        if not (start_date.isdigit() and len(start_date) == 8 and 
                end_date.isdigit() and len(end_date) == 8):
            raise ValueError("Dates must be in YYYYMMDD format")

        try:
            # Cache check after validation
            cache_key = f"load_data_{start_date}_{end_date}"
            cached_data = storage.get_cached_data(cache_key)
            if cached_data is not None:
                logger.info("Using cached data")
                return cached_data
            
            start = pd.Timestamp(datetime.strptime(start_date, '%Y%m%d'), tz=self.tz)
            end = pd.Timestamp(datetime.strptime(end_date, '%Y%m%d'), tz=self.tz)
            
            if end <= start:
                raise ValueError("End date must be after start date")
            
            # Check if requesting future data
            now = pd.Timestamp.now(tz=self.tz)
            if start > now:
                logger.info("Requesting future data, returning empty DataFrame")
                return pd.DataFrame()
            
            total_days = (end - start).days
            num_chunks = (total_days + chunk_size - 1) // chunk_size
            
            all_data = []
            current_start = start
            
            with tqdm(total=num_chunks, desc="Fetching Load Data") as pbar:
                while current_start < end:
                    current_end = min(current_start + timedelta(days=chunk_size), end)
                    chunk_success = False
                    
                    for attempt in range(max_retries):
                        try:
                            logger.debug(f"Attempt {attempt + 1} for chunk {current_start} to {current_end}")
                            chunk_data = self.client.query_load_and_forecast(
                                country_code=self.country_code,
                                start=current_start,
                                end=current_end
                            )
                            
                            if isinstance(chunk_data, pd.DataFrame) and not chunk_data.empty:
                                # Handle timezone-naive data
                                if chunk_data.index.tz is None:
                                    chunk_data.index = chunk_data.index.tz_localize('UTC').tz_convert(self.tz)
                                elif chunk_data.index.tz != self.tz:
                                    chunk_data.index = chunk_data.index.tz_convert(self.tz)
                                
                                # Ensure hourly frequency
                                chunk_data = chunk_data.asfreq('h')
                                all_data.append(chunk_data)
                                chunk_success = True
                                break
                                
                        except (requests.ConnectionError, NoMatchingDataError) as e:
                            if isinstance(e, NoMatchingDataError):
                                logger.info(f"No data available for period {current_start} to {current_end}")
                                break
                            if attempt == max_retries - 1:
                                logger.error(f"Failed after {max_retries} attempts: {str(e)}")
                                raise
                            wait_time = 2 ** attempt
                            logger.warning(f"Attempt {attempt + 1} failed, waiting {wait_time} seconds")
                            time.sleep(wait_time)
                    
                    if not chunk_success:
                        logger.warning(f"No data found between {current_start} and {current_end}")
                    
                    current_start = current_end
                    pbar.update(1)
            
            if not all_data:
                return pd.DataFrame()
            
            combined_data = pd.concat(all_data)
            
            # Ensure no duplicates and proper sorting
            combined_data = combined_data[~combined_data.index.duplicated(keep='first')]
            combined_data = combined_data.sort_index()
            
            try:
                metadata = {
                    "start_date": start_date,
                    "end_date": end_date,
                    "source": "ENTSOE",
                    "query_type": "load_data"
                }
                storage.save_raw_data(combined_data, f"load_data_{start_date}_{end_date}", metadata)
                storage.cache_data(combined_data, cache_key, expire_hours=24)
            except Exception as e:
                logger.error(f"Failed to save data: {str(e)}")
            
            return combined_data
            
        except Exception as e:
            logger.error(f"Error fetching load data: {str(e)}")
            raise

    def get_latest_load(self) -> pd.DataFrame:
        """Fetch the most recent load data (last 24 hours)"""
        try:
            # Temporarily disable cache for testing
            end = datetime.now(self.tz)
            start = end - timedelta(days=1)
            
            data = self.client.query_load_and_forecast(
                country_code=self.country_code,
                start=pd.Timestamp(start),
                end=pd.Timestamp(end)
            )
            
            if data is None or (isinstance(data, pd.DataFrame) and data.empty):
                raise Exception("No data available")
            
            if isinstance(data, pd.Series):
                data = data.to_frame('Actual Load')
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching latest load data: {str(e)}")
            raise

    def get_load_data(self, start_time, end_time):
        """Get load data with improved logging and data validation."""
        try:
            self.logger.info(f"Fetching load data from {start_time} to {end_time}")
            
            # Convert timestamps to YYYYMMDD format for fetch_load_data
            start_date = start_time.strftime('%Y%m%d')
            end_date = end_time.strftime('%Y%m%d')
            
            # Fetch data from API
            data = self.fetch_load_data(start_date, end_date)
            
            if data is None or data.empty:
                self.logger.warning("No data retrieved")
                return pd.DataFrame()
            
            # Log data quality information
            self.logger.info(f"Retrieved {len(data)} rows of data")
            nan_actual = data['Actual Load'].isna().sum() if 'Actual Load' in data.columns else 0
            nan_forecast = data['Forecasted Load'].isna().sum() if 'Forecasted Load' in data.columns else 0
            
            self.logger.info(f"Data quality check:")
            self.logger.info(f"- NaN values in Actual Load: {nan_actual}")
            self.logger.info(f"- NaN values in Forecasted Load: {nan_forecast}")
            
            # Ensure we have the required columns
            if 'Actual Load' not in data.columns:
                data['Actual Load'] = np.nan
            
            # Attempt to impute missing values
            if nan_actual > 0:
                self.logger.info("Imputing missing Actual Load values")
                if 'Forecasted Load' in data.columns:
                    data['Actual Load'] = data['Actual Load'].fillna(data['Forecasted Load'])
                    self.logger.info("Used Forecasted Load to fill missing values")
                
                # Interpolate remaining NaN values
                data['Actual Load'] = data['Actual Load'].interpolate(method='linear').ffill().bfill()
                remaining_nan = data['Actual Load'].isna().sum()
                self.logger.info(f"Remaining NaN values after imputation: {remaining_nan}")
            
            # Filter to requested time range
            data = data[start_time:end_time]
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching load data: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()
    
                
    def get_load_forecast(self, start_time=None, end_time=None) -> pd.DataFrame:
        """Fetch load forecast data for a specific time range"""
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
            
            try:
                forecast_data = self.client.query_load_forecast(
                    country_code=self.country_code,
                    start=start_time,
                    end=end_time
                )
                
                if forecast_data is None or (isinstance(forecast_data, pd.DataFrame) and forecast_data.empty):
                    logger.warning("No data from query_load_forecast, trying alternative method")
                    forecast_data = self.client.query_load(
                        country_code=self.country_code,
                        start=start_time,
                        end=end_time,
                        process_type='A01'
                    )
            except Exception as e:
                logger.warning(f"Error with ENTSOE API: {str(e)}")
                # Generate synthetic forecast
                date_range = pd.date_range(start=start_time, end=end_time, freq='h')
                base_load = 50000
                hours = np.array([dt.hour for dt in date_range])
                day_effect = 10000 * np.sin(np.pi * (hours - 6) / 12)
                forecast_values = base_load + day_effect
                forecast_data = pd.DataFrame({'Load Forecast': forecast_values}, index=date_range)
            
            if isinstance(forecast_data, pd.Series):
                forecast_data = forecast_data.to_frame('Load Forecast')
            
            return forecast_data
            
        except Exception as e:
            logger.error(f"Error fetching load forecast: {str(e)}")
            return pd.DataFrame()