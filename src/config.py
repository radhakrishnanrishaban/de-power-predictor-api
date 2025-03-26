from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
ENTSOE_API_KEY = os.getenv("ENTSOE_API_KEY")

# Data Settings
COUNTRY_CODE = "DE"
TIMEZONE = "Europe/Berlin"

# Paths
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / 'data'
RAW_DATA_PATH = DATA_DIR / 'raw'
CACHE_DIR = DATA_DIR / 'cache'

class Config:
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "DE Power Predictor API"
    
    # Data Settings
    COUNTRY_CODE: str = COUNTRY_CODE
    TIMEZONE: str = TIMEZONE
    ENTSOE_API_KEY: str = ENTSOE_API_KEY
    
    # Paths
    ROOT_DIR: Path = ROOT_DIR
    DATA_DIR: Path = DATA_DIR
    CACHE_DIR: Path = CACHE_DIR
    RAW_DATA_PATH: Path = RAW_DATA_PATH