import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
import pytz
import logging
from pathlib import Path
import sys
import numpy as np

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from src.data.clients.entsoe_client import EntsoeClient
from src.deployment.prediction_service import PredictionService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_daily_predictions():
    """Test and visualize today's predictions vs actual load."""
    try:
        # Initialize services
        entsoe_client = EntsoeClient()
        prediction_service = PredictionService()
        
        if not prediction_service.initialize():
            logger.error("Failed to initialize prediction service")
            return
            
        berlin_tz = pytz.timezone('Europe/Berlin')
        current_time = datetime.now(berlin_tz)
        start_of_day = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Get actual load data for the entire day
        actual_data = entsoe_client.get_load_data(
            start_of_day,
            current_time
        )
        
        # Get predictions for the entire day
        predictions = prediction_service.get_forecast(
            hours=24,
            start_time=start_of_day
        )
        
        if not actual_data.empty and not predictions.empty:
            plot_daily_comparison(actual_data, predictions)
            analyze_daily_performance(actual_data, predictions)
            
    except Exception as e:
        logger.error(f"Error in daily monitoring: {str(e)}")
        raise

def plot_daily_comparison(actual_data, predictions):
    """Plot daily comparison of actual vs predicted load."""
    plt.style.use('dark_background')
    plt.clf()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Plot 1: Load values and prediction intervals
    ax1.plot(actual_data.index, actual_data['Actual Load'], 
             label='Actual Load', color='blue', linewidth=2)
    ax1.plot(predictions.index, predictions['predicted_load'], 
             label='Predicted Load', color='red', linestyle='--', linewidth=2)
    ax1.fill_between(predictions.index, 
                     predictions['lower_bound'], 
                     predictions['upper_bound'],
                     alpha=0.2, color='red', label='95% Prediction Interval')
    
    # Calculate metrics for displayed data
    errors = actual_data['Actual Load'] - predictions['predicted_load']
    mae = np.mean(np.abs(errors))
    mape = np.mean(np.abs(errors / actual_data['Actual Load'])) * 100
    
    metrics_text = (
        f'Daily Metrics:\n'
        f'MAE: {mae:.2f} MW\n'
        f'MAPE: {mape:.2f}%'
    )
    ax1.text(0.02, 0.98, metrics_text,
             transform=ax1.transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
    
    ax1.set_title("Today's Load Prediction Performance")
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Load (MW)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Format x-axis to show hours
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 2: Error distribution
    sns.histplot(errors, bins=30, ax=ax2, color='blue', alpha=0.6, stat='density')
    sns.kdeplot(errors, ax=ax2, color='red', linewidth=2)
    
    ax2.axvline(x=0, color='green', linestyle='--', label='Zero Error')
    ax2.set_title('Daily Prediction Error Distribution')
    ax2.set_xlabel('Error (MW)')
    ax2.set_ylabel('Density')
    ax2.legend()
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'daily_prediction_performance.png')
    plt.show()

def analyze_daily_performance(actual_data, predictions):
    """Analyze the daily prediction performance."""
    errors = actual_data['Actual Load'] - predictions['predicted_load']
    
    metrics = {
        'mae': np.mean(np.abs(errors)),
        'rmse': np.sqrt(np.mean(np.square(errors))),
        'mape': np.mean(np.abs(errors / actual_data['Actual Load'])) * 100,
        'max_error': np.max(np.abs(errors)),
        'min_error': np.min(np.abs(errors)),
        'std_error': np.std(errors),
        'interval_coverage': np.mean((actual_data['Actual Load'] >= predictions['lower_bound']) & 
                                   (actual_data['Actual Load'] <= predictions['upper_bound'])) * 100
    }
    
    # Save metrics
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'daily_metrics.txt', 'w') as f:
        f.write("Daily Prediction Performance Metrics\n")
        f.write("==================================\n\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.2f}\n")
    
    logger.info(f"\nDaily Metrics:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.2f}")

if __name__ == "__main__":
    test_daily_predictions() 