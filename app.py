import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import os
from datetime import datetime, timedelta
from src.utils import create_features, convert_pm25_to_aqi, ISB_LAT, ISB_LON

# --- Configuration ---
API_KEY = os.getenv("OPENWEATHER_API_KEY", "3e5573c559d066b9120b40bc0c08617d")
MODEL_PATH = 'models/aqi_model.joblib'
SCALER_PATH = 'models/scaler.joblib'

# --- Load Artifacts ---
@st.cache_resource
def load_model_and_scaler():
    """Load the trained model and scaler."""
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except FileNotFoundError:
        st.error(f"Model or scaler not found. Please ensure '{MODEL_PATH}' and '{SCALER_PATH}' exist.")
        return None, None

model, scaler = load_model_and_scaler()

# --- API Fetching ---
@st.cache_data(ttl=3600) # Cache for 1 hour
def fetch_api_data():
    """Fetch all required data from OpenWeatherMap APIs."""
    # 1. Forecast Data (Next 72 hours)
    pollution_forecast_url = f"http://api.openweathermap.org/data/2.5/air_pollution/forecast?lat={ISB_LAT}&lon={ISB_LON}&appid={API_KEY}"
    weather_forecast_url = f"http://api.openweathermap.org/data/2.5/forecast/hourly?lat={ISB_LAT}&lon={ISB_LON}&appid={API_KEY}&cnt=72"
    
    # 2. Historical Data (Last 24 hours for lags)
    hist_end_ts = int(datetime.now().timestamp())
    hist_start_ts = int((datetime.now() - timedelta(hours=25)).timestamp())
    
    pollution_history_url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={ISB_LAT}&lon={ISB_LON}&start={hist_start_ts}&end={hist_end_ts}&appid={API_KEY}"
    weather_history_url = f"https://history.openweathermap.org/data/2.5/history/city?lat={ISB_LAT}&lon={ISB_LON}&type=hour&start={hist_start_ts}&end={hist_end_ts}&appid={API_KEY}"

    try:
        pol_forecast_data = requests.get(pollution_forecast_url).json()
        wea_forecast_data = requests.get(weather_forecast_url).json()
        pol_history_data = requests.get(pollution_history_url).json()
        wea_history_data = requests.get(weather_history_url).json()
        return pol_forecast_data, wea_forecast_data, pol_history_data, wea_history_data
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        return None, None, None, None

# --- Data Processing ---
def process_data_for_prediction(pol_forecast, wea_forecast, pol_history, wea_history):
    """Combine and process API data to create the feature row for prediction."""
    # Historical Data
    hist_pol_df = pd.DataFrame(pol_history['list'])
    hist_pol_df['pm2_5'] = hist_pol_df['components'].apply(lambda x: x.get('pm2_5'))
    
    hist_wea_df = pd.DataFrame(wea_history['list'])
    hist_wea_df['temp'] = hist_wea_df['main'].apply(lambda x: x.get('temp'))
    hist_wea_df['humidity'] = hist_wea_df['main'].apply(lambda x: x.get('humidity'))
    hist_wea_df['wind_speed'] = hist_wea_df['wind'].apply(lambda x: x.get('speed'))
    hist_wea_df['wind_deg'] = hist_wea_df['wind'].apply(lambda x: x.get('deg'))
    
    historical_df = pd.merge_asof(
        hist_pol_df[['dt', 'pm2_5']].sort_values('dt'),
        hist_wea_df[['dt', 'temp', 'humidity', 'wind_speed', 'wind_deg']].sort_values('dt'),
        on='dt', direction='nearest'
    )

    # Forecast Data
    fore_pol_df = pd.DataFrame(pol_forecast['list'])
    fore_pol_df['pm2_5'] = fore_pol_df['components'].apply(lambda x: x.get('pm2_5'))
    
    fore_wea_df = pd.DataFrame(wea_forecast['list'])
    fore_wea_df['temp'] = fore_wea_df['main'].apply(lambda x: x.get('temp'))
    fore_wea_df['humidity'] = fore_wea_df['main'].apply(lambda x: x.get('humidity'))
    fore_wea_df['wind_speed'] = fore_wea_df['wind'].apply(lambda x: x.get('speed'))
    fore_wea_df['wind_deg'] = fore_wea_df['wind'].apply(lambda x: x.get('deg'))

    forecast_df = pd.merge_asof(
        fore_pol_df[['dt', 'pm2_5']].sort_values('dt'),
        fore_wea_df[['dt', 'temp', 'humidity', 'wind_speed', 'wind_deg']].sort_values('dt'),
        on='dt', direction='nearest'
    )

    # Combine and create features
    combined_df = pd.concat([historical_df, forecast_df], ignore_index=True)
    features_df = create_features(combined_df, is_training=False)
    
    # Select the first row corresponding to the current time for prediction
    # This row's lag features are built from the past, and weather features are from the present
    prediction_row_index = len(historical_df)
    X_pred = features_df.iloc[[prediction_row_index]]
    
    # Get current AQI
    current_pm25 = historical_df.iloc[-1]['pm2_5']
    current_aqi = convert_pm25_to_aqi(current_pm25)

    return X_pred, current_aqi, forecast_df

# --- Streamlit App ---
st.set_page_config(page_title="AQI Predictor", layout="wide")
st.title('AQI 72-Hour Forecast for Islamabad')

if model is not None and scaler is not None:
    pol_forecast, wea_forecast, pol_history, wea_history = fetch_api_data()

    if all([pol_forecast, wea_forecast, pol_history, wea_history]):
        X_pred, current_aqi, forecast_df = process_data_for_prediction(
            pol_forecast, wea_forecast, pol_history, wea_history
        )
        
        # Align columns with the training columns
        training_cols = scaler.get_feature_names_out()
        X_pred_aligned = X_pred[training_cols]

        # Scale the input
        X_pred_scaled = scaler.transform(X_pred_aligned)

        # Make Prediction
        y_pred_pm25 = model.predict(X_pred_scaled)[0] # Get the single row of 72 predictions

        # Convert to AQI
        predicted_aqi = [convert_pm25_to_aqi(p) for p in y_pred_pm25]

        # --- Display Results ---
        st.metric("Current AQI in Islamabad", value=f"{current_aqi}")

        if max(predicted_aqi) > 150:
            st.warning("High AQI levels expected in the next 72 hours!")

        # Create a DataFrame for plotting (only first 72 hours)
        future_timestamps = pd.to_datetime(forecast_df['dt'].iloc[:72], unit='s')
        plot_df = pd.DataFrame({
            'Timestamp': future_timestamps,
            'Predicted AQI': predicted_aqi
        })

        st.subheader("Predicted AQI for the Next 72 Hours")
        st.line_chart(plot_df.set_index('Timestamp'))

        with st.expander("View Raw Prediction Data"):
            st.dataframe(plot_df)
    else:
        st.warning("Could not retrieve data for prediction. Please check API key and network.")
else:
    st.info("Model is loading or not available.")
