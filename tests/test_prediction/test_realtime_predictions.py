import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import pytz
import logging
from pathlib import Path
import sys
import numpy as np
import time
import traceback

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from src.data.clients.entsoe_client import EntsoeClient
from src.deployment.prediction_service import PredictionService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_realtime_predictions(monitoring_duration_hours=0.5):  # 30 minutes
    """Test real-time predictions with enhanced metrics."""
    try:
        # Initialize services
        entsoe_client = EntsoeClient()
        prediction_service = PredictionService()
        
        if not prediction_service.initialize():
            logger.error("Failed to initialize prediction service")
            return
            
        berlin_tz = pytz.timezone('Europe/Berlin')
        start_time = datetime.now(berlin_tz)
        end_time = start_time + timedelta(hours=monitoring_duration_hours)
        
        # Storage for results
        results = {
            'timestamp': [],
            'actual_load': [],
            'predicted_load': [],
            'lower_bound': [],
            'upper_bound': [],
            'prediction_error': [],
            'load_yesterday': []
        }
        
        logger.info(f"Starting real-time monitoring from {start_time} to {end_time}")
        logger.info("Press Ctrl+C to stop monitoring early")
        
        try:
            while datetime.now(berlin_tz) < end_time:
                current_time = datetime.now(berlin_tz)
                
                # Get actual load data
                actual_data = entsoe_client.get_load_data(
                    current_time - timedelta(hours=2),
                    current_time + timedelta(minutes=15)
                )
                
                # Get yesterday's load data
                yesterday_time = current_time - timedelta(days=1)
                yesterday_data = entsoe_client.get_load_data(
                    yesterday_time - timedelta(minutes=15),
                    yesterday_time + timedelta(minutes=15)
                )
                
                # Get prediction for current time
                predictions = prediction_service.get_forecast(
                    hours=2,
                    start_time=current_time
                )
                
                if not actual_data.empty and not predictions.empty and not yesterday_data.empty:
                    # Get the latest actual and predicted values
                    latest_actual = actual_data['Actual Load'].iloc[-1]
                    latest_pred = predictions['predicted_load'].iloc[0]
                    lower_bound = predictions['lower_bound'].iloc[0]
                    upper_bound = predictions['upper_bound'].iloc[0]
                    
                    # Calculate error
                    error = latest_actual - latest_pred
                    pct_error = (error / latest_actual) * 100
                    
                    # Store results
                    results['timestamp'].append(current_time)
                    results['actual_load'].append(latest_actual)
                    results['predicted_load'].append(latest_pred)
                    results['lower_bound'].append(lower_bound)
                    results['upper_bound'].append(upper_bound)
                    results['prediction_error'].append(error)
                    
                    # Add yesterday's load
                    yesterday_load = yesterday_data['Actual Load'].iloc[-1]
                    results['load_yesterday'].append(yesterday_load)
                    
                    # Enhanced logging
                    logger.info(f"\nPrediction Details:")
                    logger.info(f"Timestamp: {current_time}")
                    logger.info(f"Actual Load: {latest_actual:.2f} MW")
                    logger.info(f"Predicted Load: {latest_pred:.2f} MW")
                    logger.info(f"Error: {error:.2f} MW ({pct_error:.2f}%)")
                    logger.info(f"Prediction Interval: [{lower_bound:.2f}, {upper_bound:.2f}] MW")
                    logger.info(f"Interval Width: {upper_bound - lower_bound:.2f} MW")
                    logger.info(f"Actual within interval: {lower_bound <= latest_actual <= upper_bound}")
                    
                    # Create and update live plot
                    if len(results['timestamp']) > 1:
                        plot_realtime_results(results)
                
                # Reduce sleep time to 1 minute for more frequent updates
                time.sleep(60)  # Changed from 300 to 60 seconds
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        
        # Show final plot and keep window open
        plt.show()
        
    except Exception as e:
        logger.error(f"Error in real-time monitoring: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def plot_realtime_results(results):
    """Plot real-time results with continuous updates."""
    plt.style.use('dark_background')
    plt.clf()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Plot 1: Load values and prediction intervals
    ax1.plot(df['timestamp'], df['actual_load'], label='Actual Load', color='blue')
    ax1.plot(df['timestamp'], df['predicted_load'], label='Predicted Load', color='red', linestyle='--')
    ax1.fill_between(df['timestamp'], df['lower_bound'], df['upper_bound'],
                     alpha=0.2, color='red', label='95% Prediction Interval')
    
    # Calculate and display metrics
    mae = np.mean(np.abs(df['prediction_error']))
    mape = np.mean(np.abs(df['prediction_error'] / df['actual_load'])) * 100
    
    metrics_text = (
        f'Current Metrics:\n'
        f'MAE: {mae:.2f} MW\n'
        f'MAPE: {mape:.2f}%'
    )
    ax1.text(0.02, 0.98, metrics_text,
             transform=ax1.transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
    
    ax1.set_title('Real-time Load Prediction Monitoring')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Load (MW)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Error distribution
    sns.histplot(df['prediction_error'], bins=30, ax=ax2, color='blue', alpha=0.6, stat='density')
    sns.kdeplot(df['prediction_error'], ax=ax2, color='red', linewidth=2)
    
    ax2.axvline(x=0, color='green', linestyle='--', label='Zero Error')
    ax2.set_title('Prediction Error Distribution')
    ax2.set_xlabel('Error (MW)')
    ax2.set_ylabel('Density')
    ax2.legend()
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'realtime_monitoring.png')
    
    # Show plot and pause briefly to update display
    plt.draw()
    plt.pause(0.1)

def analyze_results(results):
    """Analyze the final results and save detailed metrics."""
    df = pd.DataFrame(results)
    
    # Calculate comprehensive metrics
    metrics = {
        'mae': np.mean(np.abs(df['prediction_error'])),
        'rmse': np.sqrt(np.mean(np.square(df['prediction_error']))),
        'mape': np.mean(np.abs(df['prediction_error'] / df['actual_load'])) * 100,
        'max_error': np.max(np.abs(df['prediction_error'])),
        'min_error': np.min(np.abs(df['prediction_error'])),
        'std_error': np.std(df['prediction_error']),
        'interval_coverage': np.mean((df['actual_load'] >= df['lower_bound']) & 
                                   (df['actual_load'] <= df['upper_bound'])) * 100
    }
    
    # Save metrics
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'realtime_metrics.txt', 'w') as f:
        f.write("Real-time Prediction Performance Metrics\n")
        f.write("=====================================\n\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.2f}\n")
    
    # Save detailed results
    df.to_csv(output_dir / 'realtime_results.csv')
    logger.info(f"\nFinal Metrics:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.2f}")

if __name__ == "__main__":
    # Test with 5 minutes duration for quicker feedback
    test_realtime_predictions(monitoring_duration_hours=0.083)  # 5 minutes 