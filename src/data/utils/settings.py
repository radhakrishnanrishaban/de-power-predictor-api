import os
from pathlib import Path
from typing import Dict, Any
import json
import yaml

class Settings:
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent.parent
        self.config_dir = self.base_dir / "config"
        self.model_dir = self.base_dir / "data" / "models"
        
        # Load configuration
        self.config = self._load_config()
        
        # API settings
        self.api_host = self.config.get('api', {}).get('host', 'localhost')
        self.api_port = self.config.get('api', {}).get('port', 8000)
        
        # Model settings
        self.model_version = self.config.get('model', {}).get('version', 'latest')
        self.cache_duration = self.config.get('cache', {}).get('duration', 900)  # 15 minutes
        
        # Monitoring settings
        self.enable_monitoring = self.config.get('monitoring', {}).get('enabled', True)
        self.metrics_port = self.config.get('monitoring', {}).get('port', 9090)
    
    def _load_config(self) -> Dict[str, Any]:
        config_file = self.config_dir / "config.yaml"
        if not config_file.exists():
            return {}
        
        with open(config_file) as f:
            return yaml.safe_load(f)