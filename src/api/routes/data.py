from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta

from src.data.clients.entsoe_client import EntsoeClient
from src.data.preprocess.preprocessor import LoadDataPreprocessor
from src.api.models.schemas import LoadDataResponse, MetadataResponse  # If you have schemas defined

router = APIRouter()
client = EntsoeClient()
preprocessor = LoadDataPreprocessor()

@router.get("/load/latest", response_model=LoadDataResponse)
async def get_latest_load():
    """Get latest preprocessed load data"""
    try:
        data = client.get_latest_load()
        
        # Handle empty data before processing
        if data is None or data.empty:
            raise HTTPException(
                status_code=404,
                detail="No data available"
            )
            
        processed_data = preprocessor.preprocess(data)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "data": processed_data.to_dict(orient='records'),
            "metadata": {
                "points": len(processed_data),
                "start_time": processed_data.index.min().isoformat(),
                "end_time": processed_data.index.max().isoformat()
            }
        }
    except HTTPException as he:
        raise he  # Re-raise HTTP exceptions as-is
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/load/historical/{days}", response_model=LoadDataResponse)
async def get_historical_load(days: int):
    """Get historical load data for specified number of days"""
    try:
        # Validate days parameter first, before any API calls
        if days <= 0:
            raise HTTPException(
                status_code=400,
                detail="Invalid days parameter. Must be a positive integer."
            )
            
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        try:
            raw_data = client.fetch_load_data(
                start_date=start_date.strftime('%Y%m%d'),
                end_date=end_date.strftime('%Y%m%d')
            )
        except ValueError as ve:
            # Handle validation errors from the client
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            # Handle other client errors
            raise HTTPException(status_code=500, detail=str(e))
        
        # Handle empty data
        if raw_data is None or raw_data.empty:
            raise HTTPException(
                status_code=404,
                detail="No historical data available"
            )
            
        processed_data = preprocessor.preprocess(raw_data)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "data": processed_data.to_dict(orient='records'),
            "metadata": {
                "days_requested": days,
                "points": len(processed_data),
                "start_time": processed_data.index.min().isoformat(),
                "end_time": processed_data.index.max().isoformat()
            }
        }
    except HTTPException as he:
        raise he  # Re-raise HTTP exceptions as-is
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))