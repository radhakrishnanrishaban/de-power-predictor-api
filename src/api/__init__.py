from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.config import Config
from src.api.routes import data_router, forecast_router

app = FastAPI(
    title="German Energy Forecast API",
    description="API for forecasting German energy consumption",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers with prefixes
app.include_router(data_router, prefix="/api/v1/data", tags=["data"])
app.include_router(forecast_router, prefix="/api/v1/forecast", tags=["forecast"])

@app.get("/health")
async def health_check():
    return {"status": "healthy"}