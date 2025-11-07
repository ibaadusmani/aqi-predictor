import pandas as pd
import numpy as np

ISB_LAT = 33.7380
ISB_LON = 73.0845

def create_features(df, is_training=True):
    """
    Create time-series features from a DataFrame.
    """
    # Create readable timestamp column
    df['timestamp'] = pd.to_datetime(df['dt'], unit='s')
    
    # Set index for time series operations but keep timestamp as column too
    df_indexed = df.set_index('timestamp')
    
    df_indexed['hour'] = df_indexed.index.hour
    df_indexed['day_of_week'] = df_indexed.index.dayofweek
    df_indexed['month'] = df_indexed.index.month

    # Cyclical features for hour
    df_indexed['hour_sin'] = np.sin(2 * np.pi * df_indexed['hour'] / 24)
    df_indexed['hour_cos'] = np.cos(2 * np.pi * df_indexed['hour'] / 24)

    # Lag features for pm2.5
    df_indexed['pm25_lag_1hr'] = df_indexed['pm2_5'].shift(1)
    df_indexed['pm25_lag_3hr'] = df_indexed['pm2_5'].shift(3)
    df_indexed['pm25_lag_24hr'] = df_indexed['pm2_5'].shift(24)

    # Rolling averages for pm2.5
    df_indexed['pm25_rolling_avg_6hr'] = df_indexed['pm2_5'].rolling(window=6).mean()
    df_indexed['pm25_rolling_avg_24hr'] = df_indexed['pm2_5'].rolling(window=24).mean()

    # Interaction features
    df_indexed['temp_wind_interaction'] = df_indexed['temp'] * df_indexed['wind_speed']
    
    # Drop only hour column
    df_indexed = df_indexed.drop(columns=['hour'])
    
    # Reset index to get timestamp back as a column
    df_indexed = df_indexed.reset_index()
    # Reset index to get timestamp back as a column
    df_indexed = df_indexed.reset_index()

    if is_training:
        # Create target variables
        for i in range(1, 73):
            df_indexed[f'pm25_t+{i}'] = df_indexed['pm2_5'].shift(-i)
        
        # Drop rows with NaN in feature columns (lag/rolling features at the start)
        # Keep rows with NaN only in target columns (last 72 hours - useful for inference)
        feature_cols = [col for col in df_indexed.columns if not col.startswith('pm25_t+')]
        df_indexed = df_indexed.dropna(subset=feature_cols)

    return df_indexed

def convert_pm25_to_aqi(pm25):
    """
    Convert PM2.5 concentration to AQI.
    """
    if pm25 is None:
        return None

    if 0 <= pm25 <= 12.0:
        return _linear_conversion(pm25, 0, 50, 0, 12.0)
    elif 12.1 <= pm25 <= 35.4:
        return _linear_conversion(pm25, 51, 100, 12.1, 35.4)
    elif 35.5 <= pm25 <= 55.4:
        return _linear_conversion(pm25, 101, 150, 35.5, 55.4)
    elif 55.5 <= pm25 <= 150.4:
        return _linear_conversion(pm25, 151, 200, 55.5, 150.4)
    elif 150.5 <= pm25 <= 250.4:
        return _linear_conversion(pm25, 201, 300, 150.5, 250.4)
    elif 250.5 <= pm25 <= 350.4:
        return _linear_conversion(pm25, 301, 400, 250.5, 350.4)
    elif 350.5 <= pm25 <= 500.4:
        return _linear_conversion(pm25, 401, 500, 350.5, 500.4)
    else:
        return 500

def _linear_conversion(val, i_low, i_high, c_low, c_high):
    return round(((val - c_low) / (c_high - c_low)) * (i_high - i_low) + i_low)
