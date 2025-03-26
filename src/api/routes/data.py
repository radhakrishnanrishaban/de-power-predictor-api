from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
import pandas as pd
from typing import List, Optional
from pydantic import BaseModel, ConfigDict
from src.data.clients.entsoe_client import EntsoeClient
from src.config import Config

router = APIRouter(prefix="/data", tags=["data"])

class LoadDataResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    timestamp: datetime
    actual_load: Optional[float]
    forecast_load: Optional[float]

class LoadDataListResponse(BaseModel):
    data: List[LoadDataResponse]
    start_time: datetime
    end_time: datetime
    resolution: str = "15min"

@router.get("/load", response_model=LoadDataListResponse)
async def get_load_data(
    hours_back: int = 48,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
):
    """
    Fetch load data from ENTSO-E API.
    
    Args:
        hours_back: Number of hours to look back from now (if start_date and end_date not provided)
        start_date: Optional start date for specific time range
        end_date: Optional end date for specific time range
    """
    try:
        client = EntsoeClient(
            api_key=Config.ENTSOE_API_KEY,
            country_code=Config.COUNTRY_CODE
        )

        # Set time range
        if start_date and end_date:
            query_start = start_date
            query_end = end_date
        else:
            query_end = datetime.now()
            query_start = query_end - timedelta(hours=hours_back)

        # Fetch data
        load_data = await client.get_load_data(
            start_date=query_start,
            end_date=query_end
        )

        # Convert to response format
        data_list = []
        for timestamp, row in load_data.iterrows():
            data_list.append(
                LoadDataResponse(
                    timestamp=timestamp,
                    actual_load=row.get('Actual Load'),
                    forecast_load=row.get('Forecast Load')
                )
            )

        return LoadDataListResponse(
            data=data_list,
            start_time=query_start,
            end_time=query_end
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching load data: {str(e)}"
        )

@router.get("/latest", response_model=LoadDataResponse)
async def get_latest_load():
    """Get the most recent load data point."""
    try:
        client = EntsoeClient(
            api_key=Config.ENTSOE_API_KEY,
            country_code=Config.COUNTRY_CODE
        )

        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)
        
        load_data = await client.get_load_data(
            start_date=start_time,
            end_date=end_time
        )

        # Get the latest data point
        latest_timestamp = load_data.index[-1]
        latest_data = load_data.iloc[-1]

        return LoadDataResponse(
            timestamp=latest_timestamp,
            actual_load=latest_data.get('Actual Load'),
            forecast_load=latest_data.get('Forecast Load')
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching latest load data: {str(e)}"
        )

@router.get("/statistics")
async def get_load_statistics(days: int = 7):
    """Get statistical summary of load data for the specified number of days."""
    try:
        client = EntsoeClient(
            api_key=Config.ENTSOE_API_KEY,
            country_code=Config.COUNTRY_CODE
        )

        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        load_data = await client.get_load_data(
            start_date=start_time,
            end_date=end_time
        )

        # Calculate statistics
        stats = {
            "mean_load": float(load_data['Actual Load'].mean()),
            "max_load": float(load_data['Actual Load'].max()),
            "min_load": float(load_data['Actual Load'].min()),
            "std_load": float(load_data['Actual Load'].std()),
            "data_points": len(load_data),
            "period_start": start_time,
            "period_end": end_time
        }

        return stats

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calculating load statistics: {str(e)}"
        )