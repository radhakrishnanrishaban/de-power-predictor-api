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
    """Load prediction model using LightGBM with enhanced validation"""
    
    def __init__(self):
        # Base parameters shared across all models
        self.base_params = {
            'n_estimators': 2000,
            'learning_rate': 0.01,
            'num_leaves': 63,
            'max_depth': 8,
            'feature_fraction': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'min_child_samples': 20,
            'force_row_wise': True
        }
        
        # Initialize main model for median predictions
        self.model = lgb.LGBMRegressor(
            **self.base_params,
            objective='regression',
            metric='mae'
        )
        
        # Initialize models for prediction intervals
        self.model_upper = lgb.LGBMRegressor(
            **self.base_params,
            objective='quantile',
            alpha=0.95  # 95th percentile
        )
        
        self.model_lower = lgb.LGBMRegressor(
            **self.base_params,
            objective='quantile',
            alpha=0.05  # 5th percentile
        )
        
        self.feature_importance_ = None
    
    def custom_peak_loss(self, y_true, y_pred):
        """Custom loss function that penalizes peak/trough errors more heavily"""
        # Calculate basic errors
        errors = np.abs(y_true - y_pred)
        
        # Identify peaks and troughs
        rolling_max = pd.Series(y_true).rolling(window=24, center=True).max()
        rolling_min = pd.Series(y_true).rolling(window=24, center=True).min()
        
        # Calculate weights (higher for peaks/troughs)
        weights = np.ones_like(errors)
        peak_mask = (y_true >= rolling_max * 0.95)
        trough_mask = (y_true <= rolling_min * 1.05)
        weights[peak_mask | trough_mask] = 3.0  # Triple the weight for peaks/troughs
        
        return np.mean(errors * weights)
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train all models with early stopping"""
        # Split data for validation
        train_size = int(len(X) * 0.8)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        eval_set = [(X_val, y_val)]
        
        # Train main model
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_metric=['mae', 'rmse'],
            callbacks=[lgb.early_stopping(50)]
        )
        
        # Train upper bound model
        self.model_upper.fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_metric='quantile',
            callbacks=[lgb.early_stopping(50)]
        )
        
        # Train lower bound model
        self.model_lower.fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_metric='quantile',
            callbacks=[lgb.early_stopping(50)]
        )
        
        # Store feature importance
        self.feature_importance_ = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("Models trained successfully")
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Make predictions with confidence intervals"""
        try:
            predictions = pd.DataFrame(index=X.index)
            
            # Get predictions from all models
            predictions['predicted_load'] = self.model.predict(X)
            predictions['upper_bound'] = self.model_upper.predict(X)
            predictions['lower_bound'] = self.model_lower.predict(X)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def save(self, path: str) -> None:
        """Save all models to disk"""
        try:
            model_dir = Path(path).parent
            model_dir.mkdir(parents=True, exist_ok=True)
            
            models = {
                'main': self.model,
                'upper': self.model_upper,
                'lower': self.model_lower,
                'feature_importance': self.feature_importance_
            }
            
            joblib.dump(models, path)
            logger.info(f"Models saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            raise
    
    def load(self, path: str) -> None:
        """Load all models from disk"""
        try:
            models = joblib.load(path)
            
            self.model = models['main']
            self.model_upper = models['upper']
            self.model_lower = models['lower']
            self.feature_importance_ = models['feature_importance']
            
            logger.info(f"Models loaded from {path}")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        return {
            'n_estimators': self.model.n_estimators,
            'learning_rate': self.model.learning_rate,
            'num_leaves': self.model.num_leaves,
            'max_depth': self.model.max_depth,
            'feature_fraction': self.model.feature_fraction,
            'reg_alpha': self.model.reg_alpha,
            'reg_lambda': self.model.reg_lambda,
            'min_child_samples': self.model.min_child_samples,
            'force_row_wise': self.model.force_row_wise,
            'early_stopping_rounds': self.model.early_stopping_rounds
        }