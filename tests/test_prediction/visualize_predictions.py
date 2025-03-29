import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import pytz
import logging
from pathlib import Path
import sys
import numpy as np
import traceback
# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from src.data.clients.entsoe_client import EntsoeClient
from src.deployment.prediction_service import PredictionService
from src.deployment.model_deployer import ModelDeployer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_comparison_plot():
    """Create visualization comparing real and predicted data with improved handling of missing values."""
    try:
        # Set up plotting style
        plt.style.use('dark_background')
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 16))
        
        # Initialize clients and services
        entsoe_client = EntsoeClient()
        prediction_service = PredictionService()
        
        if not prediction_service.initialize():
            logger.error("Failed to initialize prediction service")
            return
        
        # Use a historical date range for validation
        end_time = datetime.now(pytz.timezone('Europe/Berlin')) - timedelta(hours=2)
        start_time = end_time - timedelta(days=2)
        
        logger.info(f"Fetching data from {start_time} to {end_time}")
        real_data = entsoe_client.get_load_data(start_time, end_time)
        
        if real_data.empty:
            logger.error("No real data available")
            return
            
        # Get the last valid timestamp from real data
        last_valid_time = real_data['Actual Load'].dropna().index.max()
        if last_valid_time is None:
            logger.error("No valid timestamps in real data")
            return
            
        logger.info(f"Last valid real data timestamp: {last_valid_time}")
        
        # Get predictions starting from 24 hours before last valid time
        prediction_start = last_valid_time - pd.Timedelta(hours=24)
        predictions_df = prediction_service.get_forecast(hours=48, now=prediction_start)  # Increased to 48 hours
        
        if predictions_df is None or predictions_df.empty:
            logger.error("Failed to get predictions")
            return
        
        # Plot 1: Real vs Predicted Load
        ax1.plot(real_data.index, real_data['Actual Load'], 
                label='Real Load', color='blue', alpha=0.7)
        
        if 'Forecasted Load' in real_data.columns:
            ax1.plot(real_data.index, real_data['Forecasted Load'],
                    label='ENTSOE Forecast', color='green', linestyle='--', alpha=0.7)
        
        ax1.plot(predictions_df.index, predictions_df['predicted_load'],
                label='Model Prediction', color='red', linestyle=':', alpha=0.7)
        
        if 'lower_bound' in predictions_df.columns:
            ax1.fill_between(predictions_df.index,
                           predictions_df['lower_bound'],
                           predictions_df['upper_bound'],
                           color='red', alpha=0.2,
                           label='95% Confidence Interval')
        
        # Calculate metrics for historical data (excluding last 2 hours)
        historical_mask = real_data.index <= (last_valid_time - pd.Timedelta(hours=2))
        if historical_mask.any():
            historical_real = real_data[historical_mask]
            
            # Get predictions for the historical period
            historical_pred = predictions_df[
                (predictions_df.index >= historical_real.index.min()) &
                (predictions_df.index <= historical_real.index.max())
            ]
            
            logger.info(f"Historical real data range: {historical_real.index.min()} to {historical_real.index.max()}")
            logger.info(f"Historical predictions range: {historical_pred.index.min()} to {historical_pred.index.max()}")
            
            # Get overlapping timestamps
            common_index = historical_real.index.intersection(historical_pred.index)
            if len(common_index) > 0:
                logger.info(f"Found {len(common_index)} overlapping timestamps")
                real_values = historical_real.loc[common_index, 'Actual Load']
                pred_values = historical_pred.loc[common_index, 'predicted_load']
                
                # Remove any remaining NaN values
                valid_mask = ~(real_values.isna() | pred_values.isna())
                if valid_mask.any():
                    real_values = real_values[valid_mask]
                    pred_values = pred_values[valid_mask]
                    
                    # Calculate metrics
                    mae = np.mean(np.abs(real_values - pred_values))
                    rmse = np.sqrt(np.mean((real_values - pred_values)**2))
                    mape = np.mean(np.abs((real_values - pred_values) / real_values)) * 100
                    
                    stats_text = (
                        f"Historical Prediction Metrics (excluding last 2 hours):\n"
                        f"MAE: {mae:.0f} MW\n"
                        f"RMSE: {rmse:.0f} MW\n"
                        f"MAPE: {mape:.2f}%\n\n"
                        f"Data Ranges:\n"
                        f"Historical: {real_values.min():.0f} - {real_values.max():.0f} MW\n"
                        f"Forecast: {predictions_df['predicted_load'].min():.0f} - {predictions_df['predicted_load'].max():.0f} MW"
                    )
                    ax1.text(0.02, 0.98, stats_text,
                            transform=ax1.transAxes,
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
                    
                    # Store errors for distribution plot
                    errors = pred_values - real_values
                else:
                    logger.warning("No valid data points after removing NaN values")
                    errors = None
            else:
                logger.warning("No common timestamps between real and predicted data")
                errors = None
        else:
            logger.warning("No historical data available")
            errors = None
        
        ax1.set_title('Load Comparison: Real vs Predicted')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Load (MW)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Daily Load Pattern
        real_hourly = real_data.groupby(real_data.index.hour)['Actual Load'].mean()
        pred_hourly = predictions_df.groupby(predictions_df.index.hour)['predicted_load'].mean()
        
        # Ensure the index is properly ordered from 0-23
        hours = np.arange(24)
        real_hourly = real_hourly.reindex(hours, fill_value=np.nan)
        pred_hourly = pred_hourly.reindex(hours, fill_value=np.nan)
        
        ax2.plot(hours, real_hourly.values,
                label='Average Real Load', color='blue')
        ax2.plot(hours, pred_hourly.values,
                label='Average Predicted Load', color='red', linestyle='--')
        
        ax2.set_title('Daily Load Pattern Comparison')
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Average Load (MW)')
        ax2.set_xticks(hours)  # Set ticks for each hour
        ax2.set_xlim(0, 23)    # Set x-axis limits
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Error Distribution
        if errors is not None and len(errors) > 0:
            sns.histplot(errors, bins=30, ax=ax3, color='blue', alpha=0.6)
            ax3.axvline(x=0, color='red', linestyle='--')
            ax3.set_title('Historical Prediction Error Distribution')
            ax3.set_xlabel('Error (MW)')
            ax3.set_ylabel('Count')
        else:
            ax3.text(0.5, 0.5, 'No historical data available for error distribution',
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax3.transAxes)
        
        # Add vertical lines to show data boundaries
        for ax in [ax1, ax2]:
            ax.axvline(x=last_valid_time, color='yellow', linestyle='--', alpha=0.5,
                      label='Last Available Real Data')
            ax.axvline(x=last_valid_time - pd.Timedelta(hours=2), color='red',
                      linestyle='--', alpha=0.5, label='Metrics Calculation Boundary')
        
        plt.tight_layout()
        
        # Create output directory if it doesn't exist
        output_dir = Path('output')
        output_dir.mkdir(exist_ok=True)
        
        # Save the plot
        plt.savefig(output_dir / 'load_comparison.png')
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")

def plot_predictions(predictions_df, historical_data):
    """Create enhanced visualization with confidence intervals and metrics."""
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Plot 1: Time series with confidence intervals
    ax1.plot(historical_data.index, historical_data['Actual Load'], 
             label='Real Load', color='blue')
    
    if 'lower_bound' in predictions_df.columns:
        ax1.fill_between(predictions_df.index, 
                        predictions_df['lower_bound'],
                        predictions_df['upper_bound'],
                        alpha=0.2, color='red',
                        label='95% Confidence Interval')
    
    ax1.plot(predictions_df.index, predictions_df['predicted_load'],
             label='Model Prediction', color='red', linestyle='--')
    
    # Calculate and display metrics
    overlap_mask = historical_data.index.isin(predictions_df.index)
    if overlap_mask.any():
        real_overlap = historical_data.loc[overlap_mask, 'Actual Load']
        pred_overlap = predictions_df.loc[real_overlap.index, 'predicted_load']
        
        mae = np.mean(np.abs(real_overlap - pred_overlap))
        rmse = np.sqrt(np.mean((real_overlap - pred_overlap)**2))
        mape = np.mean(np.abs((real_overlap - pred_overlap) / real_overlap)) * 100
        
        metrics_text = (
            f'MAE: {mae:.0f} MW\n'
            f'RMSE: {rmse:.0f} MW\n'
            f'MAPE: {mape:.1f}%'
        )
        ax1.text(0.02, 0.98, metrics_text,
                transform=ax1.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
    
    ax1.set_title('Load Comparison: Real vs Predicted')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Load (MW)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Daily patterns
    historical_data['hour'] = historical_data.index.hour
    predictions_df['hour'] = predictions_df.index.hour
    
    sns.lineplot(data=historical_data, x='hour', y='Actual Load',
                label='Average Real Load', ax=ax2)
    sns.lineplot(data=predictions_df, x='hour', y='predicted_load',
                label='Average Predicted Load', ax=ax2)
    
    ax2.set_title('Daily Load Pattern Comparison')
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Average Load (MW)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'load_comparison.png')
    plt.close()

if __name__ == "__main__":
    create_comparison_plot() 