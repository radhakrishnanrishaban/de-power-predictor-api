import time
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

@dataclass
class PredictionMetrics:
    timestamp: datetime
    prediction_value: float
    prediction_time_ms: float
    feature_count: int
    cache_hit: bool

class MetricsCollector:
    def __init__(self):
        self.predictions: List[PredictionMetrics] = []
        self.errors: Dict[str, int] = {}
        self.cache_stats = {'hits': 0, 'misses': 0}
    
    def record_prediction(self, metrics: PredictionMetrics):
        self.predictions.append(metrics)
        if metrics.cache_hit:
            self.cache_stats['hits'] += 1
        else:
            self.cache_stats['misses'] += 1
    
    def record_error(self, error_type: str):
        self.errors[error_type] = self.errors.get(error_type, 0) + 1
    
    def get_prediction_stats(self) -> Dict:
        if not self.predictions:
            return {}
        
        recent_preds = pd.DataFrame([
            {
                'timestamp': p.timestamp,
                'value': p.prediction_value,
                'time_ms': p.prediction_time_ms
            }
            for p in self.predictions[-96:]  # Last 24 hours
        ])
        
        return {
            'mean_prediction': recent_preds['value'].mean(),
            'std_prediction': recent_preds['value'].std(),
            'avg_prediction_time_ms': recent_preds['time_ms'].mean(),
            'cache_hit_rate': self.cache_stats['hits'] / sum(self.cache_stats.values()),
            'error_count': sum(self.errors.values())
        }