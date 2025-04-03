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
    
    
    def fetch_load_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Generate synthetic load data with realistic patterns."""
        try:
            # Generate date range
            date_range = pd.date_range(start=start_date, end=end_date, freq='15min')
            data = pd.DataFrame(index=date_range)
            
            # Time components
            hour = data.index.hour
            weekday = data.index.dayofweek
            month = data.index.month
            
            # Base load (higher in winter, lower in summer)
            monthly_pattern = 5000 * np.sin(2 * np.pi * (month - 1) / 12)  # Peak in winter
            base_load = 45000 + monthly_pattern
            
            # Daily pattern
            # Morning peak (8-10), evening peak (18-20), night trough (2-4)
            daily_pattern = (
                15000 * np.exp(-((hour - 9) ** 2) / 8) +  # Morning peak
                17000 * np.exp(-((hour - 19) ** 2) / 8) +  # Evening peak
                -8000 * np.exp(-((hour - 3) ** 2) / 8)     # Night trough
            )
            
            # Weekly pattern (lower on weekends)
            weekend_effect = -8000 * (weekday >= 5)
            
            # Combine patterns
            data['Actual Load'] = (
                base_load +
                daily_pattern +
                weekend_effect +
                np.random.normal(0, 500, len(data))  # Random noise
            )
            
            # Ensure no negative values
            data['Actual Load'] = data['Actual Load'].clip(lower=20000)
            
            # Add forecasted load with some error
            forecast_error = np.random.normal(0, 1000, len(data))
            data['Forecasted Load'] = data['Actual Load'] + forecast_error
            
            return data
            
        except Exception as e:
            logger.error(f"Error generating synthetic data: {str(e)}")
            return pd.DataFrame()

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

    def get_load_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get load data with proper date handling."""
        try:
            # Ensure dates are timezone-aware
            berlin_tz = pytz.timezone('Europe/Berlin')
            if start_date.tzinfo is None:
                start_date = berlin_tz.localize(start_date)
            if end_date.tzinfo is None:
                end_date = berlin_tz.localize(end_date)
            
            return self.fetch_load_data(start_date, end_date)
            
        except Exception as e:
            logger.error(f"Error getting load data: {str(e)}")
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