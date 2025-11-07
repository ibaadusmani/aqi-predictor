import requests
import pandas as pd
from datetime import datetime, timedelta

API_KEY = "3e5573c559d066b9120b40bc0c08617d"
ISB_LAT = 33.7380
ISB_LON = 73.0845

# Store all results
results = []

print("Testing OpenWeatherMap API endpoints...")
print(f"Current date: {datetime.now()}")
print("=" * 60)

# Test 1: Current air pollution (no history)
print("\n1. CURRENT Air Pollution API (current conditions only):")
current_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={ISB_LAT}&lon={ISB_LON}&appid={API_KEY}"
response = requests.get(current_url)
if response.status_code == 200:
    data = response.json()
    if 'list' in data and len(data['list']) > 0:
        for item in data['list']:
            dt = datetime.fromtimestamp(item['dt'])
            pm25 = item['components']['pm2_5']
            results.append({
                'api_type': 'current',
                'datetime': dt,
                'pm2_5': pm25,
                'status': 'SUCCESS'
            })
            print(f"  Status: SUCCESS")
            print(f"  Timestamp: {dt}")
            print(f"  PM2.5: {pm25}")
else:
    print(f"  Status: FAILED - {response.status_code}")
    results.append({'api_type': 'current', 'status': f'FAILED-{response.status_code}'})

# Test 2: Forecast air pollution (future predictions)
print("\n2. FORECAST Air Pollution API (next 96 hours):")
forecast_url = f"http://api.openweathermap.org/data/2.5/air_pollution/forecast?lat={ISB_LAT}&lon={ISB_LON}&appid={API_KEY}"
response = requests.get(forecast_url)
if response.status_code == 200:
    data = response.json()
    if 'list' in data:
        for item in data['list']:
            dt = pd.to_datetime(item['dt'], unit='s')
            pm25 = item['components']['pm2_5']
            results.append({
                'api_type': 'forecast',
                'datetime': dt,
                'pm2_5': pm25,
                'status': 'SUCCESS'
            })
        print(f"  Status: SUCCESS")
        print(f"  Number of records: {len(data['list'])}")
        df_temp = pd.DataFrame(data['list'])
        df_temp['datetime'] = pd.to_datetime(df_temp['dt'], unit='s')
        print(f"  Date range: {df_temp['datetime'].min()} to {df_temp['datetime'].max()}")
else:
    print(f"  Status: FAILED - {response.status_code}")
    results.append({'api_type': 'forecast', 'status': f'FAILED-{response.status_code}'})

# Test 3: Historical air pollution (3 days ago)
print("\n3. HISTORY Air Pollution API (3 days ago):")
three_days_ago = datetime.now() - timedelta(days=3)
start_ts = int(three_days_ago.replace(hour=0, minute=0, second=0).timestamp())
end_ts = int(three_days_ago.replace(hour=23, minute=59, second=59).timestamp())
history_url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={ISB_LAT}&lon={ISB_LON}&start={start_ts}&end={end_ts}&appid={API_KEY}"
response = requests.get(history_url)
if response.status_code == 200:
    data = response.json()
    if 'list' in data:
        for item in data['list']:
            dt = pd.to_datetime(item['dt'], unit='s')
            pm25 = item['components']['pm2_5']
            results.append({
                'api_type': 'history_3days_ago',
                'datetime': dt,
                'pm2_5': pm25,
                'status': 'SUCCESS'
            })
        print(f"  Status: SUCCESS")
        print(f"  Number of records: {len(data['list'])}")
        if len(data['list']) > 0:
            df_temp = pd.DataFrame(data['list'])
            df_temp['datetime'] = pd.to_datetime(df_temp['dt'], unit='s')
            print(f"  Date range: {df_temp['datetime'].min()} to {df_temp['datetime'].max()}")
        else:
            print(f"  No data returned!")
else:
    print(f"  Status: FAILED - {response.status_code}")
    results.append({'api_type': 'history_3days_ago', 'status': f'FAILED-{response.status_code}'})

# Test 4: Historical air pollution (yesterday)
print("\n4. HISTORY Air Pollution API (yesterday):")
yesterday = datetime.now() - timedelta(days=1)
start_ts = int(yesterday.replace(hour=0, minute=0, second=0).timestamp())
end_ts = int(yesterday.replace(hour=23, minute=59, second=59).timestamp())
history_url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={ISB_LAT}&lon={ISB_LON}&start={start_ts}&end={end_ts}&appid={API_KEY}"
response = requests.get(history_url)
if response.status_code == 200:
    data = response.json()
    if 'list' in data:
        for item in data['list']:
            dt = pd.to_datetime(item['dt'], unit='s')
            pm25 = item['components']['pm2_5']
            results.append({
                'api_type': 'history_yesterday',
                'datetime': dt,
                'pm2_5': pm25,
                'status': 'SUCCESS'
            })
        print(f"  Status: SUCCESS")
        print(f"  Number of records: {len(data['list'])}")
        if len(data['list']) > 0:
            df_temp = pd.DataFrame(data['list'])
            df_temp['datetime'] = pd.to_datetime(df_temp['dt'], unit='s')
            print(f"  Date range: {df_temp['datetime'].min()} to {df_temp['datetime'].max()}")
        else:
            print(f"  No data returned!")
else:
    print(f"  Status: FAILED - {response.status_code}")
    results.append({'api_type': 'history_yesterday', 'status': f'FAILED-{response.status_code}'})

# Test 5: Historical air pollution (today)
print("\n5. HISTORY Air Pollution API (today):")
today = datetime.now()
start_ts = int(today.replace(hour=0, minute=0, second=0).timestamp())
end_ts = int(today.timestamp())
history_url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={ISB_LAT}&lon={ISB_LON}&start={start_ts}&end={end_ts}&appid={API_KEY}"
response = requests.get(history_url)
if response.status_code == 200:
    data = response.json()
    if 'list' in data:
        for item in data['list']:
            dt = pd.to_datetime(item['dt'], unit='s')
            pm25 = item['components']['pm2_5']
            results.append({
                'api_type': 'history_today',
                'datetime': dt,
                'pm2_5': pm25,
                'status': 'SUCCESS'
            })
        print(f"  Status: SUCCESS")
        print(f"  Number of records: {len(data['list'])}")
        if len(data['list']) > 0:
            df_temp = pd.DataFrame(data['list'])
            df_temp['datetime'] = pd.to_datetime(df_temp['dt'], unit='s')
            print(f"  Date range: {df_temp['datetime'].min()} to {df_temp['datetime'].max()}")
        else:
            print(f"  No data returned!")
else:
    print(f"  Status: FAILED - {response.status_code}")
    results.append({'api_type': 'history_today', 'status': f'FAILED-{response.status_code}'})

# Test 6: Historical weather (yesterday)
print("\n6. HISTORY Weather API (yesterday):")
yesterday = datetime.now() - timedelta(days=1)
start_ts = int(yesterday.replace(hour=0, minute=0, second=0).timestamp())
end_ts = int(yesterday.replace(hour=23, minute=59, second=59).timestamp())
weather_url = f"https://history.openweathermap.org/data/2.5/history/city?lat={ISB_LAT}&lon={ISB_LON}&type=hour&start={start_ts}&end={end_ts}&appid={API_KEY}"
response = requests.get(weather_url)
print(f"  Status Code: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    if 'list' in data:
        print(f"  Number of records: {len(data['list'])}")
        if len(data['list']) > 0:
            df_temp = pd.DataFrame(data['list'])
            df_temp['datetime'] = pd.to_datetime(df_temp['dt'], unit='s')
            print(f"  Date range: {df_temp['datetime'].min()} to {df_temp['datetime'].max()}")
else:
    print(f"  Response: {response.text[:200]}")

print("\n" + "=" * 60)
print("Testing complete!")

# Save all results to CSV
if results:
    results_df = pd.DataFrame(results)
    output_file = 'api_test_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    print(f"Total records: {len(results_df)}")
    print("\nSummary by API type:")
    print(results_df['api_type'].value_counts())
else:
    print("\nNo results to save!")
