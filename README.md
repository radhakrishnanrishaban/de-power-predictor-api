# German Energy Forecast API

Backend API service for German energy consumption forecasting.

## Setup

1. Clone the repository
2. Create virtual environment: `python -m venv venv`
3. Activate virtual environment:
   - Windows: `venv\Scripts\activate`
   - Unix: `source venv/bin/activate`
4. Install package: `pip install -e .`
5. Copy `.env.example` to `.env` and add your ENTSO-E API key
6. Run the API: `uvicorn src.api.main:app --reload`

## API Documentation

Once running, visit:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Development

Run tests: `pytest`
