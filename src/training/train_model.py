import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
import pytz
import traceback
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, max_error, median_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMRegressor
import pickle
import json
import lightgbm
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.model.predictor import LoadPredictor
from src.model.feature_extractor import LoadFeatureExtractor
from src.deployment.pipeline import DataPipeline
from src.data.preprocess.preprocessor import LoadDataPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive metrics."""
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        'r2': r2_score(y_true, y_pred),
        'max_error': max_error(y_true, y_pred),
        'median_ae': median_absolute_error(y_true, y_pred)
    }

def load_and_preprocess_data():
    """Load and preprocess the training data."""
    try:
        # Load data with explicit datetime parsing
        data_path = project_root / "data/raw/train_data/load_data_20220309_20250308.csv"
        logger.info(f"Loading data from {data_path}")
        
        # Read the data
        df = pd.read_csv(data_path)
        
        # Convert the first column to datetime with timezone
        berlin_tz = pytz.timezone('Europe/Berlin')
        datetime_col = df.columns[0]
        df[datetime_col] = pd.to_datetime(df[datetime_col], utc=True)
        df[datetime_col] = df[datetime_col].dt.tz_convert(berlin_tz)
        df.set_index(datetime_col, inplace=True)
        
        # Initialize preprocessor
        preprocessor = LoadDataPreprocessor(
            freq='15min',
            timezone='Europe/Berlin',
            rolling_window=96,
            zscore_threshold=3.0
        )
        
        # Preprocess data
        df_processed = preprocessor.preprocess(df)
        
        # Validate the processed data
        if preprocessor.validate_historical_data(df_processed):
            logger.info("Data validation passed")
            return df_processed
        else:
            raise ValueError("Data validation failed")
            
    except Exception as e:
        logger.error(f"Error loading and preprocessing data: {str(e)}")
        raise

def save_model_and_metrics(model, metrics, feature_importance=None):
    """Save model, metrics, and feature importance."""
    try:
        # Create directories if they don't exist
        model_dir = project_root / "data/models"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp for versioning
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = model_dir / "latest_model.pkl"
        model_version_path = model_dir / f"model_{timestamp}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        with open(model_version_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save metrics
        metrics_path = model_dir / "model_metrics.txt"
        metrics_version_path = model_dir / f"metrics_{timestamp}.txt"
        
        # Add timestamp to metrics
        metrics['timestamp'] = timestamp
        
        with open(metrics_path, 'w') as f:
            f.write("Model Training Metrics\n")
            f.write("=====================\n\n")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    f.write(f"{metric}: {value:.2f}\n")
                else:
                    f.write(f"{metric}: {value}\n")
            
            if feature_importance is not None:
                f.write("\nFeature Importance:\n")
                for feature, importance in feature_importance.items():
                    f.write(f"{feature}: {importance:.4f}\n")
        
        # Save a copy with timestamp
        with open(metrics_version_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logger.info(f"Model and metrics saved to {model_dir}")
        logger.info(f"Model version: {timestamp}")
        
    except Exception as e:
        logger.error(f"Error saving model and metrics: {str(e)}")
        raise

def plot_training_progress(model, save_dir):
    """Plot the training progress using the model's best iteration scores."""
    results = model.evals_result_

    plt.figure(figsize=(12, 6))
    for metric in results['valid_0'].keys():
        plt.plot(results['valid_0'][metric], label=f'Validation {metric}')
    plt.xlabel('Iterations')
    plt.ylabel('Metric Value')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / 'training_progress.png')
    plt.close()

def plot_predictions(y_true, y_pred, timestamps, save_dir):
    """Plot actual vs predicted values and residuals."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Actual vs Predicted
    ax1.plot(timestamps, y_true, label='Actual', alpha=0.7)
    ax1.plot(timestamps, y_pred, label='Predicted', alpha=0.7)
    ax1.set_title('Actual vs Predicted Load')
    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel('Load (MW)')
    ax1.legend()
    ax1.grid(True)
    
    # Residuals
    residuals = y_true - y_pred
    ax2.scatter(timestamps, residuals, alpha=0.5)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_title('Residuals Over Time')
    ax2.set_xlabel('Timestamp')
    ax2.set_ylabel('Residual (MW)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'predictions.png')
    plt.close()

def normalize_feature_importance(importance_dict):
    """Normalize feature importance values to percentages."""
    total = sum(importance_dict.values())
    return {k: (v/total)*100 for k, v in importance_dict.items()}

def train_with_prediction_simulation(X, y):
    """Train model with prediction simulation to prevent error accumulation"""
    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    # Initialize model with modified parameters
    model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        num_leaves=31,
        max_depth=8,
        feature_fraction=0.8,
        lambda_l1=1.0,
        lambda_l2=1.0,
        min_data_in_leaf=50
    )
    
    # First, train the model on training data
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=['mae', 'rmse'],
        callbacks=[lightgbm.early_stopping(stopping_rounds=50)]
    )
    
    # Now simulate prediction scenario
    predictions = []
    X_val_copy = X_val.copy()
    
    # Make predictions in 24-hour blocks
    for i in range(0, len(X_val), 96):  # 96 = 24 hours of 15-min intervals
        # Get the current block
        block_end = min(i + 96, len(X_val))
        X_block = X_val_copy.iloc[i:block_end].copy()
        
        # Make predictions for this block
        block_predictions = model.predict(X_block)
        predictions.extend(block_predictions)
        
        # Update features for the next block using predictions
        if block_end < len(X_val):
            # Update lagged features
            if 'load_lag_1' in X_val_copy.columns:
                X_val_copy.iloc[block_end:block_end+96, X_val_copy.columns.get_loc('load_lag_1')] = block_predictions[-1]
            if 'load_lag_2' in X_val_copy.columns:
                X_val_copy.iloc[block_end:block_end+96, X_val_copy.columns.get_loc('load_lag_2')] = block_predictions[-2] if len(block_predictions) > 1 else block_predictions[-1]
            if 'load_lag_3' in X_val_copy.columns:
                X_val_copy.iloc[block_end:block_end+96, X_val_copy.columns.get_loc('load_lag_3')] = block_predictions[-3] if len(block_predictions) > 2 else block_predictions[-1]
            
            # Update rolling statistics if they exist
            if 'rolling_mean_24' in X_val_copy.columns:
                X_val_copy.iloc[block_end:block_end+96, X_val_copy.columns.get_loc('rolling_mean_24')] = np.mean(block_predictions[-96:])
            if 'rolling_std_24' in X_val_copy.columns:
                X_val_copy.iloc[block_end:block_end+96, X_val_copy.columns.get_loc('rolling_std_24')] = np.std(block_predictions[-96:])
    
    # Calculate simulation metrics
    simulation_metrics = calculate_metrics(y_val, predictions)
    logger.info("Simulation Metrics:")
    for metric, value in simulation_metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    return model

def plot_predictions_with_intervals(y_true, y_pred, lower, upper, timestamps, peaks, troughs, save_dir):
    """Enhanced plot with prediction intervals and peak/trough highlighting"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Plot 1: Actual vs Predicted with intervals
    ax1.fill_between(timestamps, lower, upper, alpha=0.2, color='gray', label='95% Prediction Interval')
    ax1.plot(timestamps, y_true, label='Actual', color='blue')
    ax1.plot(timestamps, y_pred, label='Predicted', color='red', linestyle='--')
    
    # Highlight peaks and troughs
    ax1.scatter(timestamps[peaks], y_true[peaks], color='orange', marker='^', label='Peaks', zorder=5)
    ax1.scatter(timestamps[troughs], y_true[troughs], color='green', marker='v', label='Troughs', zorder=5)
    
    ax1.set_title('Load Prediction with Peaks/Troughs')
    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel('Load (MW)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Residuals with peak/trough highlighting
    residuals = y_true - y_pred
    ax2.scatter(timestamps, residuals, alpha=0.5, color='blue', label='Normal')
    ax2.scatter(timestamps[peaks], residuals[peaks], color='orange', marker='^', label='Peaks')
    ax2.scatter(timestamps[troughs], residuals[troughs], color='green', marker='v', label='Troughs')
    ax2.axhline(y=0, color='r', linestyle='--')
    
    ax2.set_title('Residuals Over Time (Highlighting Peaks/Troughs)')
    ax2.set_xlabel('Timestamp')
    ax2.set_ylabel('Residual (MW)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'predictions_with_intervals.png')
    plt.close()

def train_model():
    """Train the model with enhanced peak/trough detection and prediction"""
    try:
        # Load and preprocess data
        data = load_and_preprocess_data()
        logger.info(f"Loaded data from {data.index.min()} to {data.index.max()}")
        
        # Create plots directory
        plots_dir = project_root / "data/plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Adjust model parameters for better peak capture
        base_params = {
            'n_estimators': 2000,  # Increase from 1000
            'learning_rate': 0.005,  # Decrease for finer convergence
            'num_leaves': 63,  # Increase complexity
            'max_depth': 8,
            'min_data_in_leaf': 20,  # Decrease to capture more patterns
            'feature_fraction': 0.9,  # Increase feature usage
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
        }

        # Separate models for mean prediction and quantiles
        models = {
            'mean': LGBMRegressor(
                **base_params,
                objective='regression',
                metric='mae'
            ),
            'upper': LGBMRegressor(
                **base_params,
                objective='quantile',
                alpha=0.95,
                metric='quantile'
            ),
            'lower': LGBMRegressor(
                **base_params,
                objective='quantile',
                alpha=0.05,
                metric='quantile'
            )
        }

        # Add peak/trough indicators
        def prepare_features(data):
            features = pd.DataFrame(index=data.index)
            
            # Time-based features
            features['hour'] = data.index.hour
            features['weekday'] = data.index.dayofweek
            features['month'] = data.index.month
            features['is_weekend'] = (data.index.weekday >= 5).astype(int)
            
            # Cyclical encoding
            features['hour_sin'] = np.sin(2 * np.pi * data.index.hour / 24)
            features['hour_cos'] = np.cos(2 * np.pi * data.index.hour / 24)
            features['weekday_sin'] = np.sin(2 * np.pi * data.index.dayofweek / 7)
            features['weekday_cos'] = np.cos(2 * np.pi * data.index.dayofweek / 7)
            
            # Historical load features (if available)
            if 'Actual Load' in data.columns:
                # Lagged features
                for lag in [1, 2, 3, 4]:  # Previous hours
                    features[f'load_lag_{lag}'] = data['Actual Load'].shift(lag * 4)
                
                # Daily patterns
                features['load_yesterday'] = data['Actual Load'].shift(96)  # 24h ago
                features['load_lastweek'] = data['Actual Load'].shift(672)  # 1 week ago
                
                # Rolling statistics
                for window in [24, 48, 168]:  # 6h, 12h, 1w
                    roll = data['Actual Load'].rolling(window=window * 4)
                    features[f'rolling_mean_{window}h'] = roll.mean()
                    features[f'rolling_std_{window}h'] = roll.std()
            
            # Add peak/trough indicators
            features['is_peak_hour'] = ((data.index.hour >= 8) & (data.index.hour <= 12) |
                                       (data.index.hour >= 17) & (data.index.hour <= 21)).astype(int)
            features['is_trough_hour'] = (data.index.hour >= 0) & (data.index.hour <= 5).astype(int)
            
            # Add rate of change features
            if 'Actual Load' in data.columns:
                features['load_gradient'] = data['Actual Load'].diff() / data['Actual Load'].shift(1)
                features['load_acceleration'] = features['load_gradient'].diff()
            
            # Handle missing values
            features = features.bfill().ffill()  # Use newer pandas methods
            
            return features

        # Prepare features
        features_df = prepare_features(data)
        target = data['Actual Load']

        # Train-test split
        train_data = data[:-672]  # Last month for testing
        test_data = data[-672:]
        
        X_train = prepare_features(train_data)
        y_train = train_data['Actual Load']
        X_test = prepare_features(test_data)
        y_test = test_data['Actual Load']

        # Train models with proper validation
        for name, model in models.items():
            # Create validation set
            val_size = min(len(X_train) // 5, 672)  # Max 1 month validation
            X_val = X_train[-val_size:]
            y_val = y_train[-val_size:]
            X_fit = X_train[:-val_size]
            y_fit = y_train[:-val_size]
            
            # Fit model
            model.fit(
                X_fit, y_fit,
                eval_set=[(X_val, y_val)],
                eval_metric=['mae', 'rmse'] if name == 'mean' else ['quantile'],
                callbacks=[lightgbm.early_stopping(stopping_rounds=50)],
                categorical_feature=['weekday', 'month']  # Specify categorical features
            )

        # Make predictions
        predictions = {
            name: model.predict(X_test) 
            for name, model in models.items()
        }

        # Calculate metrics
        metrics = calculate_metrics(y_test, predictions['mean'])
        
        # Detect peaks and troughs in test data
        window = 96  # 24 hours
        rolling_max = y_test.rolling(window=window, center=True).max()
        rolling_min = y_test.rolling(window=window, center=True).min()
        
        # Define peaks as points within 5% of local maximum
        test_peaks = (y_test >= rolling_max * 0.95)
        
        # Define troughs as points within 5% of local minimum
        test_troughs = (y_test <= rolling_min * 1.05)
        
        # Calculate peak/trough specific metrics
        metrics['peak_mae'] = mean_absolute_error(
            y_test[test_peaks], 
            predictions['mean'][test_peaks]
        )
        metrics['trough_mae'] = mean_absolute_error(
            y_test[test_troughs], 
            predictions['mean'][test_troughs]
        )
        
        # Add prediction interval coverage
        in_interval = (
            (y_test >= predictions['lower']) & 
            (y_test <= predictions['upper'])
        )
        metrics['prediction_interval_coverage'] = np.mean(in_interval) * 100
        
        # Add peak/trough coverage
        peak_coverage = np.mean(in_interval[test_peaks]) * 100
        trough_coverage = np.mean(in_interval[test_troughs]) * 100
        metrics['peak_coverage'] = peak_coverage
        metrics['trough_coverage'] = trough_coverage

        # Enhanced visualization
        plot_predictions_with_intervals(
            y_test, 
            predictions['mean'],
            predictions['lower'],
            predictions['upper'],
            test_data.index,
            test_peaks,
            test_troughs,
            plots_dir
        )

        # Save model and metrics
        save_model_and_metrics(models['mean'], metrics)
        
        # Log results
        logger.info("\nTest Set Metrics:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"{metric}: {value:.2f}")
        
        logger.info("\nPeak/Trough Performance:")
        logger.info(f"Peak MAE: {metrics['peak_mae']:.2f} MW")
        logger.info(f"Trough MAE: {metrics['trough_mae']:.2f} MW")
        logger.info(f"Peak Coverage: {peak_coverage:.2f}%")
        logger.info(f"Trough Coverage: {trough_coverage:.2f}%")
        
        logger.info(f"\nPlots saved in: {plots_dir}")
        
    except Exception as e:
        logger.error(f"Error in train_model: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    train_model() 