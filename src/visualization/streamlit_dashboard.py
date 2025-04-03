import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pytz
import time
import sys
import os
import numpy as np
from pathlib import Path
import plotly.express as px
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.deployment.prediction_service import PredictionService
from src.data.clients.entsoe_client import EntsoeClient

# Set page config with dark theme
st.set_page_config(
    page_title="Energy Forecast Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Add custom CSS to improve styling
st.markdown("""
    <style>
    /* Remove all default Streamlit spacing */
    .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
        padding-left: 0rem;
        padding-right: 0rem;
    }
    .element-container {
        padding: 0 !important;
    }
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    /* Adjust plot container */
    .plot-container {
        padding: 0 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Define all constants at the top of the file
FORECAST_HOURS = 24
CACHE_DURATION = 15 * 60
HISTORY_DAYS = 3  # Reduced from 10 to 3 days of historical data
REFRESH_INTERVAL = 30 * 60  # 30 minutes in seconds

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/last_updated")
async def get_last_updated():
    return {"timestamp": int(time.time())}

@app.get("/data")
async def get_data():
    data = fetch_data()  # Your existing fetch_data function
    return {
        "timestamps": data.index.tolist(),
        "actual_load": data['Actual Load'].tolist(),
        "entsoe_forecast": data['ENTSOE Forecast'].tolist(),
        "model_forecast": data['Model Forecast'].tolist()
    }

@st.cache_resource(ttl=3600)  # Cache for 1 hour
def get_prediction_service():
    try:
        service = PredictionService()
        success = service.initialize()
        if not success:
            st.error("Failed to initialize prediction service")
            return None
        logger.info("Prediction service initialized successfully")
        return service
    except Exception as e:
        st.error(f"Error initializing prediction service: {str(e)}")
        logger.error(f"Prediction service initialization failed: {str(e)}")
        return None

@st.cache_resource(ttl=3600)  # Cache for 1 hour
def get_entsoe_client():
    try:
        client = EntsoeClient()
        logger.info("ENTSOE client initialized successfully")
        return client
    except Exception as e:
        st.error(f"Error initializing ENTSOE client: {str(e)}")
        logger.error(f"ENTSOE client initialization failed: {str(e)}")
        return None

prediction_service = get_prediction_service()
entsoe_client = get_entsoe_client()

def initialize_session_state():
    if 'last_update_time' not in st.session_state:
        st.session_state.last_update_time = 0
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'update_counter' not in st.session_state:
        st.session_state.update_counter = 0

initialize_session_state()


# Create placeholder for the main chart only
chart_placeholder = st.empty()

def should_refresh():
    current_time = time.time()
    time_since_last_update = current_time - st.session_state.last_update_time
    return time_since_last_update >= REFRESH_INTERVAL

def calculate_mape_by_duration(data: pd.DataFrame):
    """Calculate MAPE for different time durations"""
    try:
        # Modify durations to ensure we have data
        durations = {
            '1h': timedelta(hours=1),
            '3h': timedelta(hours=3),
            '6h': timedelta(hours=6),
            '12h': timedelta(hours=12)
        }
        
        results = {'Duration': [], 'ENTSO-E\'s Model': [], 'Our Model': []}
        
        for duration_name, duration in durations.items():
            end_time = data.index.max()
            start_time = end_time - duration
            mask = (data.index >= start_time) & (data.index <= end_time)
            period_data = data[mask].copy()  # Make a copy to avoid SettingWithCopyWarning
            
            if len(period_data) > 0 and 'Actual Load' in period_data.columns:
                actual = period_data['Actual Load'].dropna()
                
                if 'ENTSOE Forecast' in period_data.columns:
                    entsoe = period_data['ENTSOE Forecast'].dropna()
                    common_idx = actual.index.intersection(entsoe.index)
                    if len(common_idx) > 0:
                        entsoe_mape = np.mean(np.abs((actual[common_idx] - entsoe[common_idx]) / actual[common_idx])) * 100
                    else:
                        entsoe_mape = np.nan
                
                if 'Model Forecast' in period_data.columns:
                    model = period_data['Model Forecast'].dropna()
                    common_idx = actual.index.intersection(model.index)
                    if len(common_idx) > 0:
                        model_mape = np.mean(np.abs((actual[common_idx] - model[common_idx]) / actual[common_idx])) * 100
                    else:
                        model_mape = np.nan
                
                logger.info(f"Period data shape: {period_data.shape}")
                logger.info(f"Actual data points: {len(actual)}")
                logger.info(f"ENTSOE forecast points: {len(entsoe)}")
                logger.info(f"Model forecast points: {len(model)}")
                logger.info(f"Common points with ENTSOE: {len(common_idx)}")
                
                results['Duration'].append(duration_name)
                results['ENTSO-E\'s Model'].append(entsoe_mape)
                results['Our Model'].append(model_mape)
        
        return pd.DataFrame(results)
    except Exception as e:
        logger.error(f"Error calculating MAPE by duration: {str(e)}")
        return None



def update_dashboard():
    try:
        if (should_refresh() or 
            st.session_state.data is None or 
            st.session_state.update_counter > 0):
            logger.info("Triggering data refresh")
            new_data = fetch_data()
            if new_data is not None:
                st.session_state.data = new_data
                st.session_state.last_update_time = time.time()
                st.session_state.update_counter = 0
            else:
                logger.warning("fetch_data returned None, retaining old data")
        
        data = st.session_state.data
        if data is not None:
            fig = make_subplots(rows=1, cols=1)
            
            # Set x-axis range to today only
            today = pd.Timestamp.now(tz=pytz.UTC).replace(hour=0, minute=0, second=0, microsecond=0)
            tomorrow = today + pd.Timedelta(days=1)
            
            # Plot actual load
            if 'Actual Load' in data.columns:
                mask_actual = (~data['Actual Load'].isna()) & (data.index >= today) & (data.index <= tomorrow)
                fig.add_trace(
                    go.Scatter(
                        x=data.index[mask_actual],
                        y=data['Actual Load'][mask_actual],
                        name='Actual Load [MW]',
                        line=dict(color='blue')
                    )
                )
            
            # Plot ENTSO-E forecast
            if 'ENTSOE Forecast' in data.columns:
                mask_entsoe = (~data['ENTSOE Forecast'].isna()) & (data.index >= today) & (data.index <= tomorrow)
                fig.add_trace(
                    go.Scatter(
                        x=data.index[mask_entsoe],
                        y=data['ENTSOE Forecast'][mask_entsoe],
                        name="ENTSO-E's previous-day forecasted load [MW]",
                        line=dict(color='brown')
                    )
                )
            
            # Plot model forecast
            if 'Model Forecast' in data.columns:
                mask_forecast = (~data['Model Forecast'].isna()) & (data.index >= today) & (data.index <= tomorrow)
                fig.add_trace(
                    go.Scatter(
                        x=data.index[mask_forecast],
                        y=data['Model Forecast'][mask_forecast],
                        name='Our previous-day forecasted load [MW]',
                        line=dict(color='green')
                    )
                )
            
            # Add vertical line for "Now"
            now = pd.Timestamp.now(tz=pytz.UTC).replace(second=0, microsecond=0)
            now_timestamp = (now - pd.Timestamp("1970-01-01", tz=pytz.UTC)).total_seconds() * 1000
            fig.add_vline(x=now_timestamp, line_dash="dash", line_color="red", annotation_text="Now", annotation_position="top")
            
            # Update layout with better aspect ratio settings
            fig.update_layout(
                title={
                    'text': '',
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                plot_bgcolor='#0E1117',
                paper_bgcolor='#0E1117',
                font=dict(color='#FFFFFF'),
                xaxis=dict(
                    gridcolor='#1f1f1f',
                    range=[today, tomorrow],
                    type='date'
                ),
                yaxis=dict(
                    gridcolor='#1f1f1f',
                    title='Load [MW]',
                ),
                hovermode='x unified',
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#FFFFFF')
                ),
                height=350,  # Reduced height further
                margin=dict(l=20, r=20, t=30, b=20),  # Minimal margins
                autosize=True  # Changed to True
            )
            
            # Last updated text
            time_since_update = int(time.time() - st.session_state.last_update_time)
            update_text = (f"Last updated: {time_since_update} seconds ago" 
                         if time_since_update < 60 
                         else f"Last updated: {time_since_update // 60} minutes ago")
            st.markdown(f"""
                <div style='
                    text-align: right;
                    color: #666;
                    padding: 10px;
                    font-size: 0.8em;
                    position: fixed;
                    # bottom: 10px;
                    right: 10px;
                    background-color: rgba(14, 17, 23, 0.8);
                    border-radius: 5px;
                '>
                    {update_text}
                </div>
            """, unsafe_allow_html=True)

            # Display the chart with container width
            chart_placeholder.plotly_chart(fig, use_container_width=True, config={
                'displayModeBar': False,
                'responsive': True,
                'scrollZoom': False  # Disable scroll zoom to prevent accidental zooming
            })
        
        else:
            st.error("No data available to display")
            logger.error("No data available for display")
    except Exception as e:
        st.error(f"Error updating dashboard: {str(e)}")
        logger.error(f"Error updating dashboard: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

def generate_dummy_data(start_time, end_time, freq='15min'):
    date_range = pd.date_range(start=start_time, end=end_time, freq=freq, tz=pytz.UTC)
    hours = np.array([dt.hour for dt in date_range])
    base_load = 45000 + 15000 * np.sin(np.pi * (hours - 6) / 12)
    noise = np.random.normal(0, 2000, len(date_range))
    weekday = np.array([dt.weekday() < 5 for dt in date_range]).astype(int)
    weekly_effect = weekday * 5000
    load = base_load + noise + weekly_effect
    return pd.Series(load, index=date_range)



@st.cache_data(ttl=CACHE_DURATION)
def fetch_data():
    """Fetch and prepare data for the dashboard"""
    logger.info("Entering fetch_data()")
    
    # Start with UTC time for consistency - use pandas Timestamp
    end_time_utc = pd.Timestamp.now(tz='UTC').replace(second=0, microsecond=0)
    # Round to nearest 15 minutes
    minutes = end_time_utc.minute
    end_time_utc = end_time_utc.replace(minute=15 * (minutes // 15))
    
    try:
        start_time_utc = end_time_utc - pd.Timedelta(days=HISTORY_DAYS)  # Reduced history
        if not prediction_service or not entsoe_client:
            raise ValueError("Services not properly initialized")
        
        # Initialize the DataFrame first
        data = pd.DataFrame(index=pd.date_range(
            start=start_time_utc, 
            end=end_time_utc + pd.Timedelta(hours=FORECAST_HOURS), 
            freq='15min', 
            tz=pytz.UTC
        ))
        
        # Fetch data in parallel using threads
        def fetch_actual_load():
            try:
                berlin_tz = pytz.timezone('Europe/Berlin')
                start_time_berlin = start_time_utc.tz_convert(berlin_tz)
                end_time_berlin = end_time_utc.tz_convert(berlin_tz)
                
                actual_load_df = entsoe_client.get_load_data(start_time_berlin, end_time_berlin)
                if isinstance(actual_load_df, pd.DataFrame) and not actual_load_df.empty:
                    load_column = 'Actual Load' if 'Actual Load' in actual_load_df.columns else actual_load_df.columns[0]
                    actual_load = actual_load_df[load_column]
                    if actual_load.index.tzinfo is None:
                        actual_load.index = actual_load.index.tz_localize(berlin_tz)
                    actual_load.index = actual_load.index.tz_convert(pytz.UTC)
                    return actual_load
            except Exception as e:
                logger.warning(f"Actual load fetch failed: {str(e)}")
            return None
        
        def fetch_forecast():
            try:
                forecast = prediction_service.get_forecast(hours=FORECAST_HOURS)
                if forecast is not None:
                    if forecast.index.tzinfo is None:
                        forecast.index = forecast.index.tz_localize('Europe/Berlin')
                    forecast.index = forecast.index.tz_convert(pytz.UTC)
                    return forecast
            except Exception as e:
                logger.warning(f"Forecast fetch failed: {str(e)}")
            return None
        
        def fetch_entsoe_forecast():
            try:
                forecast = prediction_service.deployer.pipeline.get_entsoe_forecast()
                if forecast is not None:
                    if forecast.index.tzinfo is None:
                        forecast.index = forecast.index.tz_localize('Europe/Berlin')
                    forecast.index = forecast.index.tz_convert(pytz.UTC)
                    return forecast
            except Exception as e:
                logger.warning(f"ENTSOE forecast fetch failed: {str(e)}")
            return None
        
        # Fetch data in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_data = {
                executor.submit(fetch_actual_load): 'Actual Load',
                executor.submit(fetch_forecast): 'Model Forecast',
                executor.submit(fetch_entsoe_forecast): 'ENTSOE Forecast'
            }
            
            for future in as_completed(future_to_data):
                data_type = future_to_data[future]
                try:
                    result = future.result()
                    if result is not None:
                        if isinstance(result, pd.DataFrame):
                            data.loc[result.index, data_type] = result[data_type]
                        else:
                            data.loc[result.index, data_type] = result
                except Exception as e:
                    logger.error(f"Error fetching {data_type}: {str(e)}")
        
        if not data.dropna(how='all').empty:
            logger.info("Data prepared successfully")
            logger.info(f"Downloaded {len(data.dropna(how='all'))} data points")
            return data
        else:
            logger.warning("No data available, using dummy data")
            return generate_dummy_data_frame(start_time_utc, end_time_utc)
            
    except Exception as e:
        logger.error(f"Error in fetch_data: {str(e)}")
        return None

def generate_dummy_data_frame(start_time, end_time):
    """Generate dummy data for demonstration"""
    dummy_end = end_time + pd.Timedelta(hours=FORECAST_HOURS)
    date_range = pd.date_range(start=start_time, end=dummy_end, freq='15min', tz=pytz.UTC)
    return pd.DataFrame({
        'Actual Load': generate_dummy_data(start_time, end_time),
        'Model Forecast': generate_dummy_data(end_time, dummy_end) * 1.1,
        'ENTSOE Forecast': generate_dummy_data(end_time, dummy_end) * 1.05
    }, index=date_range)

update_dashboard()