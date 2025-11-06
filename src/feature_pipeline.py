import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from src.utils import ISB_LAT, ISB_LON, create_features

# Load API key from environment variable or use the one from the plan
API_KEY = os.getenv("OPENWEATHER_API_KEY", "3e5573c559d066b9120b40bc0c08617d")

def fetch_data(url):
    """Fetches data from a given URL and returns the JSON response."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from {url}: {e}")
        return None

def main():
    """Main function to run the feature pipeline."""
    # 1. Define time range (last 350 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=350)
    start_timestamp = int(start_date.timestamp())
    end_timestamp = int(end_date.timestamp())

    # 2. Fetch Air Pollution Data
    print("Fetching air pollution data...")
    pollution_url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={ISB_LAT}&lon={ISB_LON}&start={start_timestamp}&end={end_timestamp}&appid={API_KEY}"
    pollution_data = fetch_data(pollution_url)
    
    if not pollution_data or 'list' not in pollution_data:
        print("Could not fetch or parse pollution data. Exiting.")
        return

    pollution_df = pd.DataFrame(pollution_data['list'])
    # Extract pm2_5 from components
    pollution_df['pm2_5'] = pollution_df['components'].apply(lambda x: x.get('pm2_5'))
    pollution_df = pollution_df[['dt', 'pm2_5']]

    # 3. Fetch Weather Data
    print("Fetching weather data...")
    weather_url = f"https://history.openweathermap.org/data/2.5/history/city?lat={ISB_LAT}&lon={ISB_LON}&type=hour&start={start_timestamp}&end={end_timestamp}&appid={API_KEY}"
    weather_data = fetch_data(weather_url)
    
    if not weather_data or 'list' not in weather_data:
        print("Could not fetch weather data. Exiting.")
        return

    weather_df = pd.DataFrame(weather_data['list'])
    weather_df['temp'] = weather_df['main'].apply(lambda x: x.get('temp'))
    weather_df['humidity'] = weather_df['main'].apply(lambda x: x.get('humidity'))
    weather_df['wind_speed'] = weather_df['wind'].apply(lambda x: x.get('speed'))
    weather_df['wind_deg'] = weather_df['wind'].apply(lambda x: x.get('deg'))
    weather_df = weather_df[['dt', 'temp', 'humidity', 'wind_speed', 'wind_deg']]

    # 4. Merge DataFrames
    print("Merging data...")
    # Convert 'dt' to numeric for merging, as there can be slight discrepancies
    pollution_df['dt'] = pd.to_numeric(pollution_df['dt'])
    weather_df['dt'] = pd.to_numeric(weather_df['dt'])
    # Merge on the nearest 'dt' value, as timestamps might not align perfectly
    merged_df = pd.merge_asof(pollution_df.sort_values('dt'), weather_df.sort_values('dt'), on='dt', direction='nearest')

    # 5. Create Features
    print("Creating features...")
    features_df = create_features(merged_df, is_training=True)

    # 6. Save Features
    parquet_path = 'data/processed/features.parquet'
    csv_path = 'data/processed/features.csv'
    
    print(f"Saving features to {parquet_path}...")
    features_df.to_parquet(parquet_path)
    
    print(f"Saving features to {csv_path}...")
    features_df.to_csv(csv_path)
    
    print("Feature pipeline completed successfully.")

if __name__ == "__main__":
    main()
