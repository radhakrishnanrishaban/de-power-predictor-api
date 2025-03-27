import pandas as pd
from pathlib import Path
from datetime import datetime
import json
from typing import Optional, Union, Dict
from src.config import Config

class DataStorage:
    """Handles data storage and retrieval operations."""
    
    def __init__(self):
        self.raw_data_path = Config.RAW_DATA_PATH
        self.processed_data_path = Config.RAW_DATA_PATH / "processed"
        self.cache_path = Config.RAW_DATA_PATH / "cache"
        
        # Create directories if they don't exist
        for path in [self.raw_data_path, self.processed_data_path, self.cache_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    def save_raw_data(
        self,
        data: pd.DataFrame,
        dataset_name: str,
        metadata: Optional[Dict] = None
    ) -> Path:
        """Save raw data with metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{dataset_name}_{timestamp}"
        
        # Save data
        data_path = self.raw_data_path / f"{filename}.parquet"
        data.to_parquet(data_path)
        
        # Save metadata if provided
        if metadata:
            metadata.update({
                "timestamp": timestamp,
                "rows": len(data),
                "columns": list(data.columns)
            })
            meta_path = self.raw_data_path / f"{filename}_metadata.json"
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        return data_path
    
    def save_processed_data(
        self,
        data: pd.DataFrame,
        dataset_name: str,
        processing_info: Optional[Dict] = None
    ) -> Path:
        """Save processed data with processing information."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{dataset_name}_processed_{timestamp}"
        
        # Save data
        data_path = self.processed_data_path / f"{filename}.parquet"
        data.to_parquet(data_path)
        
        # Save processing info
        if processing_info:
            processing_info.update({
                "timestamp": timestamp,
                "rows": len(data),
                "columns": list(data.columns)
            })
            info_path = self.processed_data_path / f"{filename}_info.json"
            with open(info_path, 'w') as f:
                json.dump(processing_info, f, indent=2)
        
        return data_path
    
    def load_latest_data(
        self,
        dataset_name: str,
        processed: bool = True
    ) -> Optional[pd.DataFrame]:
        """Load the latest version of a dataset."""
        base_path = self.processed_data_path if processed else self.raw_data_path
        pattern = f"{dataset_name}_processed_*" if processed else f"{dataset_name}_*"
        
        # Find all matching files
        files = list(base_path.glob(f"{pattern}.parquet"))
        if not files:
            return None
        
        # Get the latest file
        latest_file = max(files, key=lambda x: x.stat().st_mtime)
        return pd.read_parquet(latest_file)
    
    def cache_data(
        self,
        data: pd.DataFrame,
        cache_key: str,
        expire_hours: int = 24
    ) -> None:
        """Cache data with expiration."""
        cache_info = {
            "timestamp": datetime.now().isoformat(),
            "expire_hours": expire_hours
        }
        
        # Save data and cache info
        data_path = self.cache_path / f"{cache_key}.parquet"
        info_path = self.cache_path / f"{cache_key}_info.json"
        
        data.to_parquet(data_path)
        with open(info_path, 'w') as f:
            json.dump(cache_info, f)
    
    def get_cached_data(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Retrieve cached data if not expired."""
        data_path = self.cache_path / f"{cache_key}.parquet"
        info_path = self.cache_path / f"{cache_key}_info.json"
        
        if not (data_path.exists() and info_path.exists()):
            return None
        
        # Check expiration
        with open(info_path, 'r') as f:
            cache_info = json.load(f)
        
        cache_time = datetime.fromisoformat(cache_info["timestamp"])
        expire_hours = cache_info["expire_hours"]
        
        if (datetime.now() - cache_time).total_seconds() > expire_hours * 3600:
            # Cache expired, remove files
            data_path.unlink()
            info_path.unlink()
            return None
        
        return pd.read_parquet(data_path)

# Create a singleton instance
storage = DataStorage()