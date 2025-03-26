import pandas as pd
from pathlib import Path
from src.config import Config

def save_data(data: pd.DataFrame, filename: str) -> None:
    """
    Save data to the raw data directory
    
    Args:
        data: DataFrame to save
        filename: Name of the file to save
    """
    # Ensure the raw data directory exists
    save_path = Config.RAW_DATA_PATH
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Create full file path
    file_path = save_path / filename
    
    # Save the data
    if filename.endswith('.csv'):
        data.to_csv(file_path)
    elif filename.endswith('.parquet'):
        data.to_parquet(file_path)
    else:
        # Default to csv if no extension matches
        file_path = file_path.with_suffix('.csv')
        data.to_csv(file_path)