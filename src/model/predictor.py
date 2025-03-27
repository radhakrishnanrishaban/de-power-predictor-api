import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import Dict, Any, Optional
import joblib
import logging
from pathlib import Path

from .base import BaseModel

logger = logging.getLogger(__name__)

class LoadPredictor(BaseModel):
    """Load prediction model using LightGBM"""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {
            'n_estimators': 1000,
            'learning_rate': 0.03,
            'num_leaves': 31,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'lambda_l1': 0.05,
            'lambda_l2': 0.05,
            'min_data_in_leaf': 30,
            'max_depth': 12,
            'force_row_wise': True,
            'verbose': -1,
            'metric': ['mae', 'rmse'],
            'n_jobs': -1
        }
        self.model = lgb.LGBMRegressor(**self.params)
        self.feature_importance_ = None
        
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model with early stopping"""
        try:
            # Split data for validation
            val_size = min(96*7, len(X)//5)  # 7 days or 20% of data
            X_train = X[:-val_size]
            X_val = X[-val_size:]
            y_train = y[:-val_size]
            y_val = y[-val_size:]
            
            # LightGBM specific parameters for early stopping
            fit_params = {
                'eval_metric': ['mae', 'rmse'],
                'eval_set': [(X_val, y_val)],
                'callbacks': [
                    lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(period=0)  # Disable logging
                ]
            }
            
            self.model.fit(
                X_train, y_train,
                **fit_params
            )
            
            self.feature_importance_ = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info(f"Model trained successfully with {len(X)} samples")
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Make predictions"""
        try:
            predictions = self.model.predict(X)
            return pd.Series(predictions, index=X.index)
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def save(self, path: str) -> None:
        """Save model to disk"""
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.model, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load(self, path: str) -> None:
        """Load model from disk"""
        try:
            self.model = joblib.load(path)
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        return self.params