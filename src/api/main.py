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

# Include routers
app.include_router(data_router)
app.include_router(forecast_router)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}