from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.config import Config
from src.api.routes import forecast, data

app = FastAPI(
    title="German Power Load Predictor API",
    description="API for predicting power load in Germany",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the routers with proper prefixes
app.include_router(
    forecast.router,
    prefix="/api/v1",
    tags=["forecast"]
)

app.include_router(
    data.router,
    prefix="/api/v1/data",
    tags=["data"]
)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}