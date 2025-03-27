import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import pytz
import logging
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from src.deployment.model_deployer import ModelDeployer
from src.data.clients.entsoe_client import EntsoeClient

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_predictions():
    """Test the prediction pipeline and visualize results."""
    try:
        logger.info("Starting prediction pipeline test...")
        
        # Set up the plot style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Initialize components
        logger.info("Initializing components...")
        entsoe_client = EntsoeClient()
        deployer = ModelDeployer()
        
        # Get historical data for comparison
        logger.info("Fetching historical data...")
        end_time = datetime.now(pytz.timezone('Europe/Berlin'))
        start_time = end_time - timedelta(days=10)
        historical_data = entsoe_client.get_load_data(start_time, end_time)
        
        if historical_data.empty:
            logger.error("No historical data retrieved")
            return
            
        logger.info(f"Retrieved {len(historical_data)} historical data points")
        logger.info(f"Historical data columns: {historical_data.columns.tolist()}")
        
        # Initialize pipeline with historical data
        logger.info("Initializing pipeline...")
        success = deployer.initialize_pipeline(historical_data)
        if not success:
            logger.error("Failed to initialize pipeline")
            return
        
        # Load the model
        logger.info("Loading model...")
        if not deployer.load_model():
            logger.error("Failed to load model")
            return
        
        # Make prediction starting from the last historical timestamp
        logger.info("Making predictions...")
        last_timestamp = historical_data.index[-1]
        prediction = deployer.make_prediction(last_timestamp)
        
        if prediction is None:
            logger.error("Failed to generate prediction")
            return
            
        logger.info(f"Generated {len(prediction)} predictions")
        
        # Create output directory if it doesn't exist
        output_dir = Path(project_root) / "output"
        output_dir.mkdir(exist_ok=True)
        
        # Create the plot
        logger.info("Creating visualization...")
        plt.figure(figsize=(15, 8))
        
        # Plot historical data
        plt.plot(historical_data.index, historical_data['Actual Load'], 
                 label='Historical Load', linewidth=2, alpha=0.7)
        
        if 'Forecasted Load' in historical_data.columns:
            plt.plot(historical_data.index, historical_data['Forecasted Load'],
                    label='ENTSOE Forecast', linewidth=2, alpha=0.7, linestyle='--')
        
        # Plot model prediction
        plt.plot(prediction.index, prediction.values,
                 label='Model Prediction', linewidth=2, alpha=0.7, linestyle=':')
        
        # Customize the plot
        plt.title('Load Prediction Comparison', fontsize=14, pad=20)
        plt.xlabel('Timestamp', fontsize=12)
        plt.ylabel('Load (MW)', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Add some statistics
        stats_text = (
            f"Historical Mean: {historical_data['Actual Load'].mean():.0f} MW\n"
            f"Prediction Mean: {prediction.mean():.0f} MW\n"
            f"Historical Std: {historical_data['Actual Load'].std():.0f} MW\n"
            f"Prediction Std: {prediction.std():.0f} MW"
        )
        plt.text(0.02, 0.98, stats_text,
                 transform=plt.gca().transAxes,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Adjust layout and save
        plt.tight_layout()
        output_path = output_dir / 'prediction_comparison.png'
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Plot saved to {output_path}")
        
        # Print some metrics
        print("\nPrediction Summary:")
        print(f"Number of historical points: {len(historical_data)}")
        print(f"Number of predicted points: {len(prediction)}")
        print(f"\nValue Ranges:")
        print(f"Historical: [{historical_data['Actual Load'].min():.0f}, "
              f"{historical_data['Actual Load'].max():.0f}] MW")
        print(f"Predicted: [{prediction.min():.0f}, {prediction.max():.0f}] MW")
        
        logger.info("Pipeline test completed successfully")
        
    except Exception as e:
        logger.error(f"Error in pipeline test: {str(e)}", exc_info=True)

if __name__ == "__main__":
    plot_predictions()