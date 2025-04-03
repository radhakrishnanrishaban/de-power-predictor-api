import logging
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LoadDataPreprocessor:
    """Handles cleaning and preprocessing of actual load data."""
    
    def __init__(self, 
                 freq: str = '15min',
                 timezone: str = 'Europe/Berlin',
                 rolling_window: int = 96,  # 24 hours in 15-min intervals
                 zscore_threshold: float = 3.0):
        """Initialize the preprocessor.
        
        Args:
            freq: Expected frequency of the data (default: '15min')
            timezone: Target timezone (default: 'Europe/Berlin')
            rolling_window: Window size for rolling statistics (default: 96)
            zscore_threshold: Threshold for outlier detection (default: 3.0)
        """
        self.freq = freq
        self.timezone = timezone
        self.rolling_window = rolling_window
        self.zscore_threshold = zscore_threshold
        
        # Initialize attributes for historical data
        self.historical_start = None
        self.historical_end = None
        self.data_min = None
        self.data_max = None
        self.data_mean = None
        self.data_std = None
        self.rolling_mean = None
        self.rolling_std = None
        
        logger.info(f"Initialized LoadDataPreprocessor with freq={freq}, timezone={timezone}")
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main preprocessing pipeline for load data."""
        try:
            logger.info("Starting preprocessing pipeline")
            
            # Validate input first
            if not isinstance(df, pd.DataFrame):
                raise ValueError("Input must be a pandas DataFrame")
            
            if 'Actual Load' not in df.columns:
                raise ValueError("DataFrame must contain 'Actual Load' column")
            
            logger.info(f"Input data shape: {df.shape}")
            
            # Create a copy to avoid modifying the original
            df = df.copy()
            
            # Process pipeline
            df = self._prepare_index(df)
            df = self._handle_duplicates(df)
            df = self._validate_values(df)
            df = self._handle_missing_values(df)  # Handle missing values before outlier detection
            df = self._remove_outliers(df)
            
            logger.info("Preprocessing completed successfully")
            logger.info(f"Output data shape: {df.shape}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error in preprocessing pipeline: {str(e)}")
            raise
    
    def _prepare_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare the DataFrame index."""
        try:
            # Create a copy to avoid modifying the original
            df = df.copy()
            
            # First convert all timestamps to UTC
            if isinstance(df.index, pd.DatetimeIndex):
                if df.index.tz is not None:
                    df.index = df.index.tz_convert('UTC')
                else:
                    df.index = df.index.tz_localize('UTC')
            else:
                # Convert to datetime if not already
                df.index = pd.to_datetime(df.index, utc=True)
            
            # Then convert to desired timezone
            df.index = df.index.tz_convert(self.timezone)
            
            # Handle duplicates before setting frequency
            if df.index.duplicated().any():
                n_duplicates = df.index.duplicated().sum()
                logger.warning(f"Found {n_duplicates} duplicate timestamps")
                df = df.groupby(level=0).median()
                logger.info(f"Resolved {n_duplicates} duplicates using median values")
            
            # Now set frequency after duplicates are handled
            if df.index.freq is None:
                df = df.asfreq(self.freq, method='ffill')
                logger.info(f"Set frequency to {self.freq}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error preparing index: {str(e)}")
            raise
    
    def _handle_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle duplicate timestamps."""
        logger.info("Handling duplicate timestamps")
        
        if df.index.duplicated().any():
            n_duplicates = df.index.duplicated().sum()
            logger.warning(f"Found {n_duplicates} duplicate timestamps")
            
            # Keep the median value for duplicate timestamps
            df = df.groupby(level=0).median()
            logger.info(f"Resolved {n_duplicates} duplicates using median values")
        
        return df
    
    def _validate_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate load values."""
        logger.info("Validating load values")
        
        df = df.copy()
        
        # Handle NaN values first
        df['Actual Load'] = df['Actual Load'].fillna(df['Actual Load'].median())
        
        # Check for negative values
        negative_mask = df['Actual Load'] < 0
        if negative_mask.any():
            n_negative = negative_mask.sum()
            logger.warning(f"Found {n_negative} negative values")
            df.loc[negative_mask, 'Actual Load'] = 0.0
        
        # Check for unreasonably high values (e.g., > 100 GW)
        high_mask = df['Actual Load'] > 100000
        if high_mask.any():
            n_high = high_mask.sum()
            logger.warning(f"Found {n_high} unreasonably high values")
            df.loc[high_mask, 'Actual Load'] = df['Actual Load'].median()
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced outlier detection using multiple methods."""
        logger.info("Removing outliers with enhanced detection")
        df = df.copy()

        # 1. Calculate multiple rolling statistics
        windows = [24, 48, 96, 168]  # 6h, 12h, 24h, 1week in 15-min intervals
        outlier_masks = []
        
        for window in windows:
            # Calculate rolling stats
            rolling_mean = df['Actual Load'].rolling(
                window=window,
                center=True,
                min_periods=max(3, window // 4)
            ).mean()
            
            rolling_std = df['Actual Load'].rolling(
                window=window,
                center=True,
                min_periods=max(3, window // 4)
            ).std()
            
            # Calculate z-scores for this window
            z_scores = abs((df['Actual Load'] - rolling_mean) / rolling_std)
            outlier_masks.append(z_scores > self.zscore_threshold)
        
        # 2. Add day-of-week pattern detection
        dow_means = df.groupby([df.index.dayofweek, df.index.hour])['Actual Load'].transform('mean')
        dow_stds = df.groupby([df.index.dayofweek, df.index.hour])['Actual Load'].transform('std')
        dow_z_scores = abs((df['Actual Load'] - dow_means) / dow_stds)
        outlier_masks.append(dow_z_scores > self.zscore_threshold)
        
        # Combine outlier masks (point must be flagged by at least 2 methods)
        final_outlier_mask = sum(outlier_masks) >= 2
        n_outliers = final_outlier_mask.sum()
        
        if n_outliers > 0:
            logger.warning(f"Found {n_outliers} outliers")
            
            # Replace outliers with pattern-based estimates
            for idx in df[final_outlier_mask].index:
                # Get pattern-based estimate
                hour_pattern = df.groupby(df.index.hour)['Actual Load'].mean()[idx.hour]
                dow_pattern = df.groupby(df.index.dayofweek)['Actual Load'].mean()[idx.dayofweek]
                recent_mean = df['Actual Load'].rolling(window=96, center=True).mean()[idx]
                
                # Weighted combination
                df.loc[idx, 'Actual Load'] = (
                    hour_pattern * 0.4 +
                    dow_pattern * 0.3 +
                    recent_mean * 0.3
                )
            
            logger.info(f"Replaced {n_outliers} outliers with pattern-based estimates")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced missing value handling with pattern-based filling."""
        logger.info("Handling missing values with pattern-based approach")
        
        initial_missing = df['Actual Load'].isna().sum()
        if initial_missing > 0:
            logger.warning(f"Found {initial_missing} missing values")
            
            df = df.copy()
            missing_mask = df['Actual Load'].isna()
            
            # 1. Try to fill with same hour from previous day
            for idx in df[missing_mask].index:
                day_ago = idx - pd.Timedelta(days=1)
                if day_ago in df.index and not pd.isna(df.loc[day_ago, 'Actual Load']):
                    df.loc[idx, 'Actual Load'] = df.loc[day_ago, 'Actual Load']
            
            # 2. Fill remaining with pattern-based estimates
            still_missing = df['Actual Load'].isna()
            if still_missing.any():
                # Calculate patterns
                hour_patterns = df.groupby(df.index.hour)['Actual Load'].mean()
                dow_patterns = df.groupby(df.index.dayofweek)['Actual Load'].mean()
                
                for idx in df[still_missing].index:
                    hour_val = hour_patterns[idx.hour]
                    dow_val = dow_patterns[idx.dayofweek]
                    overall_mean = df['Actual Load'].mean()
                    
                    df.loc[idx, 'Actual Load'] = (
                        hour_val * 0.5 +
                        dow_val * 0.3 +
                        overall_mean * 0.2
                    )
            
            # 3. Final interpolation for any remaining gaps
            df['Actual Load'] = df['Actual Load'].interpolate(method='time', limit=4)
            
            final_missing = df['Actual Load'].isna().sum()
            logger.info(f"Resolved {initial_missing - final_missing} missing values")
            
            if final_missing > 0:
                logger.warning(f"Unable to resolve {final_missing} missing values")
        
        return df
    
    def validate_historical_data(self, data: pd.DataFrame, min_days: int = 7) -> bool:
        """Validate that historical data meets minimum requirements."""
        try:
            if data is None or data.empty:
                logger.error("Historical data is empty")
                return False
            
            # Check if index is datetime
            if not isinstance(data.index, pd.DatetimeIndex):
                logger.error("Historical data index is not DatetimeIndex")
                return False
            
            # Check if timezone is set
            if data.index.tz is None:
                logger.warning("Historical data timezone is not set, setting to default")
                data.index = data.index.tz_localize(self.timezone)
            
            # Calculate data span in days with better precision
            time_delta = data.index.max() - data.index.min()
            data_span = time_delta.total_seconds() / (24 * 3600)
            
            # Use numpy's isclose for robust floating point comparison
            if not np.isclose(data_span, min_days, rtol=1e-2, atol=1e-2):  # 1% relative tolerance, 1% absolute tolerance
                if data_span < min_days:
                    logger.error(f"Historical data span ({data_span:.2f} days) is less than minimum required ({min_days} days)")
                    return False
            
            # Check frequency
            expected_periods = int(np.ceil(data_span * 24 * 60 / pd.Timedelta(self.freq).total_seconds() * 60))
            actual_periods = len(data)
            if actual_periods < expected_periods * 0.9:  # Allow 10% missing
                logger.warning(f"Historical data has fewer periods ({actual_periods}) than expected ({expected_periods})")
                # We'll still proceed, but with a warning
            
            logger.info(f"Historical data validation passed: {len(data)} records spanning {data_span:.2f} days")
            return True
            
        except Exception as e:
            logger.error(f"Error validating historical data: {str(e)}")
            return False

    def initialize_with_historical(self, data: pd.DataFrame) -> None:
        """Initialize the preprocessor with historical data.
        
        This preprocesses the historical data and stores its boundaries.
        
        Args:
            data: DataFrame with historical data
        """
        try:
            # Preprocess the historical data
            processed_data = self.preprocess(data)
            
            if processed_data is None or processed_data.empty:
                logger.error("Failed to preprocess historical data")
                return
            
            # Store data boundaries
            self.historical_start = processed_data.index.min()
            self.historical_end = processed_data.index.max()
            
            # Store statistics for outlier detection
            self._update_statistics(processed_data)
            
            logger.info(f"Initialized with historical data from {self.historical_start} to {self.historical_end}")
            
        except Exception as e:
            logger.error(f"Error initializing with historical data: {str(e)}")

    def _update_statistics(self, data: pd.DataFrame) -> None:
        """Update rolling statistics based on new data.
        
        Args:
            data: DataFrame with new data
        """
        try:
            # Calculate basic statistics
            self.data_min = data.min().min()
            self.data_max = data.max().max()
            self.data_mean = data.mean().mean()
            self.data_std = data.std().std()
            
            # Calculate rolling statistics if enough data
            if len(data) >= self.rolling_window:
                self.rolling_mean = data.rolling(window=self.rolling_window, center=True).mean()
                self.rolling_std = data.rolling(window=self.rolling_window, center=True).std()
            else:
                logger.warning(f"Not enough data for rolling statistics (need {self.rolling_window}, got {len(data)})")
                
            logger.info(f"Updated statistics: min={self.data_min:.2f}, max={self.data_max:.2f}, mean={self.data_mean:.2f}, std={self.data_std:.2f}")
            
        except Exception as e:
            logger.error(f"Error updating statistics: {str(e)}")
    
    def preprocess_live(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess new live data with improved handling of future timestamps."""
        try:
            logger.info("Starting preprocessing pipeline")
            logger.info(f"Input data shape: {df.shape}")
            
            if df is None or df.empty:
                logger.warning("Input DataFrame is empty")
                return None
            
            if 'Actual Load' not in df.columns:
                logger.error("DataFrame must contain 'Actual Load' column")
                raise ValueError("DataFrame must contain 'Actual Load' column")
            
            # Create a copy to avoid modifying the original
            df = df.copy()
            
            # Prepare index and handle duplicates
            df = self._prepare_index(df)
            df = self._handle_duplicates(df)
            
            # Identify the last valid timestamp
            last_valid_time = df['Actual Load'].last_valid_index()
            
            if last_valid_time is not None:
                # Handle historical data (up to last_valid_time)
                historical_mask = df.index <= last_valid_time
                historical_data = df[historical_mask].copy()
                
                # Apply full preprocessing to historical data
                historical_data = self._validate_values(historical_data)
                historical_data = self._handle_missing_values(historical_data)
                historical_data = self._remove_outliers(historical_data)
                
                # Handle future data (after last_valid_time)
                future_mask = df.index > last_valid_time
                if future_mask.any():
                    future_data = df[future_mask].copy()
                    logger.info(f"Processing {len(future_data)} future timestamps")
                    
                    # For future data, we'll estimate values using patterns
                    if len(historical_data) > 0:
                        # Calculate patterns from historical data
                        hourly_pattern = historical_data.groupby(historical_data.index.hour)['Actual Load'].mean()
                        daily_pattern = historical_data.groupby(historical_data.index.dayofweek)['Actual Load'].mean()
                        overall_mean = historical_data['Actual Load'].mean()
                        
                        # Estimate future values
                        for idx in future_data.index:
                            future_data.loc[idx, 'Actual Load'] = (
                                hourly_pattern[idx.hour] * 0.5 +
                                daily_pattern[idx.dayofweek] * 0.3 +
                                overall_mean * 0.2
                            )
                    else:
                        logger.warning("No historical data available for pattern estimation")
                        future_data['Actual Load'] = df['Actual Load'].mean()
                    
                    # Combine historical and future data
                    df = pd.concat([historical_data, future_data])
            else:
                # If no valid timestamp found, apply basic preprocessing
                df = self._validate_values(df)
                df = self._handle_missing_values(df)
            
            # Update statistics
            self._update_statistics(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error preprocessing live data: {str(e)}")
            raise 