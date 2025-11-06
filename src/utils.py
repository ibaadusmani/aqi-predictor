import pandas as pd
import numpy as np

ISB_LAT = 33.7380
ISB_LON = 73.0845

def create_features(df, is_training=True):
    """
    Create time-series features from a DataFrame.
    """
    df['timestamp'] = pd.to_datetime(df['dt'], unit='s')
    df = df.set_index('timestamp')
    
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month

    # Cyclical features for hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Lag features for pm2.5
    df['pm25_lag_1hr'] = df['pm2_5'].shift(1)
    df['pm25_lag_3hr'] = df['pm2_5'].shift(3)
    df['pm25_lag_24hr'] = df['pm2_5'].shift(24)

    # Rolling averages for pm2.5
    df['pm25_rolling_avg_6hr'] = df['pm2_5'].rolling(window=6).mean()
    df['pm25_rolling_avg_24hr'] = df['pm2_5'].rolling(window=24).mean()

    # Interaction features
    df['temp_wind_interaction'] = df['temp'] * df['wind_speed']
    
    # Drop original dt and hour columns
    df = df.drop(columns=['dt', 'hour'])

    if is_training:
        # Create target variables
        for i in range(1, 73):
            df[f'pm25_t+{i}'] = df['pm2_5'].shift(-i)
        
        # Drop rows with NaNs created by lags/shifts
        df = df.dropna()

    return df

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
