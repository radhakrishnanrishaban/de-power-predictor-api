import time
from datetime import datetime, timedelta
import pytz
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class RetrainingScheduler:
    def __init__(self, data_updater, model_trainer, retraining_hour=2):
        self.data_updater = data_updater
        self.model_trainer = model_trainer
        self.retraining_hour = retraining_hour
        
    def calculate_next_run(self):
        """Calculate next run time (default 2 AM Berlin time)."""
        now = datetime.now(pytz.timezone('Europe/Berlin'))
        next_run = now.replace(hour=self.retraining_hour, minute=0, second=0)
        if next_run <= now:
            next_run += timedelta(days=1)
        return next_run
        
    def run(self):
        """Run the scheduler."""
        while True:
            try:
                next_run = self.calculate_next_run()
                time_to_wait = (next_run - datetime.now(pytz.timezone('Europe/Berlin'))).total_seconds()
                
                logger.info(f"Next retraining scheduled for: {next_run}")
                time.sleep(time_to_wait)
                
                # Update data and retrain
                new_data_file = self.data_updater.update_training_data()
                logger.info(f"Data updated: {new_data_file}")
                
                self.model_trainer.train_model()
                logger.info("Model retrained successfully")
                
            except Exception as e:
                logger.error(f"Error in retraining schedule: {str(e)}")
                time.sleep(3600)  # Wait an hour before retrying 