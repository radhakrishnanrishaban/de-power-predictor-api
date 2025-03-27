from fastapi import APIRouter, HTTPException, Query, Path, status
from datetime import datetime, timedelta
import pytz
from typing import Optional
import logging
import pandas as pd
import numpy as np
import math

from src.data.manager import DataManager
from src.api.models.schemas import LoadResponse, LoadMetadata

logger = logging.getLogger(__name__)
router = APIRouter()
data_manager = DataManager()

def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean DataFrame values for JSON serialization"""
    df = df.copy()
    for col in df.select_dtypes(include=['float64', 'float32']).columns:
        # Convert to Python objects first to handle NaN/inf properly
        df[col] = df[col].where(~(pd.isna(df[col]) | np.isinf(df[col])), None)
        # Convert remaining values to float
        df[col] = df[col].apply(lambda x: float(x) if x is not None else None)
    return df

@router.get("/load/latest", response_model=LoadResponse)
async def get_latest_load():
    """Get latest load data"""
    try:
        data = data_manager.get_latest_load()
        
        if data is None or data.empty:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No data available"
            )
        
        # Sanitize the DataFrame
        data = sanitize_dataframe(data)
            
        return LoadResponse(
            timestamp=datetime.now(pytz.UTC),
            data=data.to_dict(orient='records'),  # Convert to dict after sanitization
            metadata=LoadMetadata(
                points=len(data),
                start_time=data.index[0],
                end_time=data.index[-1]
            )
        )
    except HTTPException as he:
        # Re-raise HTTP exceptions
        raise he
    except Exception as e:
        # Log and raise any other exceptions as 500 errors
        logger.error(f"Error fetching latest load data: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/load/historical/{days}", response_model=LoadResponse)
async def get_historical_load(
    days: int = Path(..., ge=1, le=30, description="Days of historical data to fetch"),
    use_cache: bool = Query(True, description="Whether to use cached data")
):
    """Get historical load data"""
    try:
        end_time = datetime.now(pytz.UTC)
        start_time = end_time - timedelta(days=days)
        
        logger.info(f"Fetching historical load data for {days} days (cache: {use_cache})")
        
        data = data_manager.get_load_data(
            start_time=start_time,
            end_time=end_time,
            use_cache=use_cache
        )
        
        if data is None or data.empty:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No historical data available for the past {days} days"
            )
        
        # Sanitize the DataFrame
        data = sanitize_dataframe(data)
            
        return LoadResponse(
            timestamp=datetime.now(pytz.UTC),
            data=data.to_dict(orient='records'),
            metadata=LoadMetadata(
                points=len(data),
                start_time=data.index[0],
                end_time=data.index[-1],
                days_requested=days
            )
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error fetching historical load data: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/load/cache/stats")
async def get_cache_statistics():
    """Get statistics about cached data."""
    try:
        stats = data_manager.get_cache_stats()
        return {
            "timestamp": datetime.now(pytz.UTC).isoformat(),
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error getting cache stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve cache statistics"
        )

def _calculate_quality_score(data) -> float:
    """Calculate a quality score for the data."""
    try:
        # Calculate percentage of non-null values
        completeness = 1 - (data.isnull().sum().sum() / (len(data) * len(data.columns)))
        
        # Check time series consistency
        time_diff = data.index.to_series().diff().dropna()
        expected_diff = pd.Timedelta(minutes=15)
        consistency = (time_diff == expected_diff).mean()
        
        # Combine scores
        quality_score = (completeness + consistency) / 2
        return round(quality_score * 100, 2)
    except Exception as e:
        logger.warning(f"Error calculating quality score: {str(e)}")
        return 0.0