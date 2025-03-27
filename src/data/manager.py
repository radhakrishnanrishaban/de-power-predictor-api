import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Union, List
import pandas as pd
import pytz
from pathlib import Path
import json

from src.data.utils import storage
from src.data.clients.entsoe_client import EntsoeClient
from src.data.preprocess.preprocessor import LoadDataPreprocessor
from src.config import Config

logger = logging.getLogger(__name__)

class DataManager:
    """Coordinates data operations between different components."""
    
    def __init__(self):
        self.storage = storage
        self.entsoe_client = EntsoeClient()
        self.preprocessor = LoadDataPreprocessor()
        self.tz = pytz.timezone(Config.TIMEZONE)
        
        # Initialize cache cleanup schedule
        self._last_cleanup = datetime.now()
        self.cleanup_interval = timedelta(hours=6)
    
    def get_load_data(
        self,
        start_time: Union[str, datetime],
        end_time: Union[str, datetime],
        use_cache: bool = True,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Get load data for a specific time range with caching.
        
        Args:
            start_time: Start time
            end_time: End time
            use_cache: Whether to use cached data
            force_refresh: Force fetch new data even if cached
        """
        try:
            # Convert times to datetime if needed
            start_dt = pd.Timestamp(start_time)
            end_dt = pd.Timestamp(end_time)
            
            if use_cache and not force_refresh:
                cache_key = f"load_data_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}"
                cached_data = self.storage.get_cached_data(cache_key)
                if cached_data is not None:
                    logger.info("Using cached load data")
                    return cached_data
            
            # Fetch new data
            raw_data = self.entsoe_client.get_load_data(start_dt, end_dt)
            if raw_data.empty:
                logger.warning("No load data received from ENTSOE")
                return pd.DataFrame()
            
            # Process data
            processed_data = self.preprocessor.preprocess(raw_data)
            
            # Save both raw and processed data
            metadata = {
                "start_time": start_dt.isoformat(),
                "end_time": end_dt.isoformat(),
                "source": "ENTSOE",
                "processing_timestamp": datetime.now().isoformat()
            }
            
            self.storage.save_raw_data(raw_data, "load_data", metadata)
            self.storage.save_processed_data(
                processed_data,
                "load_data",
                processing_info={"preprocessor_version": "1.0"}
            )
            
            # Cache the processed data
            if use_cache:
                cache_key = f"load_data_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}"
                self.storage.cache_data(processed_data, cache_key, expire_hours=24)
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error in get_load_data: {str(e)}")
            return pd.DataFrame()
    
    def get_latest_load(self, hours: int = 24) -> pd.DataFrame:
        """Get the latest load data."""
        try:
            end_time = datetime.now(self.tz)
            start_time = end_time - timedelta(hours=hours)
            return self.get_load_data(start_time, end_time, use_cache=True)
        except Exception as e:
            logger.error(f"Error in get_latest_load: {str(e)}")
            return pd.DataFrame()
    
    def get_load_forecast(
        self,
        hours: int = 24,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Get load forecast data."""
        try:
            cache_key = f"forecast_{hours}h"
            
            if use_cache:
                cached_forecast = self.storage.get_cached_data(cache_key)
                if cached_forecast is not None:
                    return cached_forecast
            
            forecast_data = self.entsoe_client.get_load_forecast(
                end_time=datetime.now(self.tz) + timedelta(hours=hours)
            )
            
            if not forecast_data.empty:
                self.storage.cache_data(forecast_data, cache_key, expire_hours=3)
                
                # Save forecast data
                metadata = {
                    "forecast_hours": hours,
                    "timestamp": datetime.now().isoformat()
                }
                self.storage.save_processed_data(
                    forecast_data,
                    "forecast_data",
                    processing_info=metadata
                )
            
            return forecast_data
            
        except Exception as e:
            logger.error(f"Error in get_load_forecast: {str(e)}")
            return pd.DataFrame()
    
    def cleanup_cache(self, force: bool = False) -> None:
        """Clean up expired cache entries."""
        try:
            if not force and datetime.now() - self._last_cleanup < self.cleanup_interval:
                return
            
            logger.info("Starting cache cleanup")
            cache_files = list(self.storage.cache_path.glob("*.parquet"))
            
            for data_path in cache_files:
                info_path = data_path.with_suffix(".json")
                if not info_path.exists():
                    logger.warning(f"Missing info file for {data_path}")
                    data_path.unlink()
                    continue
                
                try:
                    with open(info_path, 'r') as f:
                        cache_info = json.load(f)
                    
                    cache_time = datetime.fromisoformat(cache_info["timestamp"])
                    expire_hours = cache_info["expire_hours"]
                    
                    if (datetime.now() - cache_time).total_seconds() > expire_hours * 3600:
                        data_path.unlink()
                        info_path.unlink()
                        logger.debug(f"Removed expired cache: {data_path.name}")
                        
                except Exception as e:
                    logger.error(f"Error processing cache file {data_path}: {str(e)}")
                    
            self._last_cleanup = datetime.now()
            logger.info("Cache cleanup completed")
            
        except Exception as e:
            logger.error(f"Error in cleanup_cache: {str(e)}")
    
    def get_cache_stats(self) -> Dict:
        """Get statistics about cached data."""
        try:
            cache_files = list(self.storage.cache_path.glob("*.parquet"))
            total_size = sum(f.stat().st_size for f in cache_files)
            
            return {
                "total_files": len(cache_files),
                "total_size_mb": total_size / (1024 * 1024),
                "last_cleanup": self._last_cleanup.isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
            return {}

# Create singleton instance
data_manager = DataManager()