import pandas as pd
import numpy as np
import glob
import os
import argparse
from pathlib import Path

def create_features_for_device(df, device_meta=None, weather_df=None):
    """
    Apply feature engineering to a single device's dataframe.
    Assumes df is sorted by Timedate.
    """
    # Fix potential column name mismatches
    col_mapping = {
        'timedate': 'Timedate', 
        'deviceid': 'deviceId'
    }
    df = df.rename(columns={c: col_mapping.get(str(c).lower(), c) for c in df.columns})

    if 'Timedate' not in df.columns:
        raise KeyError(f"'Timedate' not found. Available columns: {list(df.columns)}")

    # Parse Timedate
    df['Timedate'] = pd.to_datetime(df['Timedate'])
    df = df.sort_values(by='Timedate')
    
    # Create date column for joining with daily weather
    df['date'] = df['Timedate'].dt.date.astype(str)

    # 1. Merge with metadata and weather if provided
    if device_meta is not None:
        for col in ['latitude', 'longitude']:
            df[col] = device_meta[col]
            
    if weather_df is not None and device_meta is not None:
        # Filter weather for this specific location to speed up join
        loc_weather = weather_df[
            (weather_df['latitude'] == device_meta['latitude']) & 
            (weather_df['longitude'] == device_meta['longitude'])
        ].copy()
        
        if not loc_weather.empty:
            df = df.merge(loc_weather, on=['date', 'latitude', 'longitude'], how='left')
            
            # New Weather Features (Rolling)
            weather_cols = ['temperature_2m_mean', 'precipitation_sum', 'wind_speed_10m_max']
            for col in weather_cols:
                if col in df.columns:
                    # Daily weather is the same for all 288 rows of a day. 
                    # 7-day rolling mean = 7 * 288 rows
                    df[f'weather_{col}_roll_7d'] = df[col].rolling(window=7*288, min_periods=1).mean()

    # 2. Row-level (time-based) features
    df['hour'] = df['Timedate'].dt.hour
    df['dayofweek'] = df['Timedate'].dt.dayofweek
    df['month'] = df['Timedate'].dt.month
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    # Cyclic encoding for time features (hour and month)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
    
    # 3. Time-series features (Rolling window features)
    # Select columns to compute rolling features on (based on task3.pdf)
    target_cols = ['t1', 't2', 't5', 'x1'] # External temp, Internal temp, Load HEX temp, Operating freq
    available_cols = [c for c in target_cols if c in df.columns]
    
    for col in available_cols:
        # Create 1-hour rolling features (12 * 5 mins)
        df[f'{col}_roll_mean_1h'] = df[col].rolling(window=12, min_periods=1).mean()
        df[f'{col}_roll_std_1h'] = df[col].rolling(window=12, min_periods=1).std().fillna(0)
        
        # Create 24-hour rolling features (288 * 5 mins)
        df[f'{col}_roll_mean_24h'] = df[col].rolling(window=288, min_periods=1).mean()
        
        # Lags
        df[f'{col}_lag_1h'] = df[col].shift(12)
        df[f'{col}_lag_24h'] = df[col].shift(288)

    # 4. Target specific features (x2)
    if 'x2' in df.columns:
        df['x2_roll_mean_1h'] = df['x2'].rolling(window=12, min_periods=1).mean()
        df['x2_lag_24h'] = df['x2'].shift(288)

    return df

def aggregate_and_engineer(input_dir, output_dir):
    """
    Bucket data by deviceId, then process each device including external data joins.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    temp_dir = os.path.join(output_dir, 'temp_by_device')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    # Load auxiliary data
    print("Loading auxiliary data (devices and weather)...")
    try:
        devices_meta = pd.read_csv('data/devices.csv')
        # Set index for fast lookup
        devices_meta = devices_meta.set_index('deviceId')
        weather_df = pd.read_csv('data/weather_daily_updates.csv')
        weather_df['date'] = pd.to_datetime(weather_df['date']).dt.date.astype(str)
    except Exception as e:
        print(f"Warning: Could not load auxiliary data: {e}")
        devices_meta = None
        weather_df = None

    split_files = glob.glob(os.path.join(input_dir, '*.csv'))
    print(f"Found {len(split_files)} split files.")
    
    # Step 1: Bucket by deviceId
    print("Step 1: Bucketing data by deviceId...")
    state_file = os.path.join(temp_dir, 'processed_chunks.txt')
    processed_chunks = set()
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            processed_chunks = set(line.strip() for line in f)
            print(f"Resuming: Found {len(processed_chunks)} already processed chunks.")
    
    known_devices = set()
    for file in glob.glob(os.path.join(temp_dir, '*.csv')):
        known_devices.add(os.path.basename(file).replace('.csv', ''))
    
    for i, file in enumerate(sorted(split_files)):
        filename = os.path.basename(file)
        if filename in processed_chunks:
            continue
            
        print(f"Processing split {i}/{len(split_files)}: {filename}...")
        chunk = pd.read_csv(file)
        col_mapping = {'timedate': 'Timedate', 'deviceid': 'deviceId'}
        chunk = chunk.rename(columns={c: col_mapping.get(str(c).lower(), c) for c in chunk.columns})
        
        if 'deviceId' not in chunk.columns: continue
            
        for device_id, group in chunk.groupby('deviceId'):
            safe_id = str(device_id).replace('/', '_').replace('\\', '_')
            out_file = os.path.join(temp_dir, f"{safe_id}.csv")
            mode = 'a' if safe_id in known_devices else 'w'
            header = not (safe_id in known_devices)
            group.to_csv(out_file, mode=mode, header=header, index=False)
            known_devices.add(safe_id)
            
        with open(state_file, 'a') as f:
            f.write(filename + '\n')
        processed_chunks.add(filename)

    # Step 2: Feature Engineering per device
    print(f"Step 2: Feature engineering for {len(known_devices)} devices...")
    device_files = glob.glob(os.path.join(temp_dir, '*.csv'))
    final_output_dir = os.path.join(output_dir, 'engineered')
    if not os.path.exists(final_output_dir):
        os.makedirs(final_output_dir)
        
    for i, file in enumerate(device_files):
        safe_id = os.path.basename(file).replace('.csv', '')
        if i % 50 == 0: print(f"Engineering device {i}/{len(device_files)}: {safe_id}")
            
        df = pd.read_csv(file)
        
        # Determine device metadata
        d_meta = None
        if devices_meta is not None:
            # Try to find deviceId in index (handling potential safe_id translation)
            # This logic assumes deviceId doesn't actually contain slash/backslash 
            # Or matches what's in devices.csv
            for actual_id in devices_meta.index:
                if str(actual_id).replace('/', '_').replace('\\', '_') == safe_id:
                    d_meta = devices_meta.loc[actual_id]
                    break
        
        df_engineered = create_features_for_device(df, device_meta=d_meta, weather_df=weather_df)
        
        out_file = os.path.join(final_output_dir, os.path.basename(file))
        df_engineered.to_csv(out_file, index=False)
        
    print("Done! Engineered data is in:", final_output_dir)
    print("You may want to clean up the temporary bucketing directory:", temp_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature Engineering for split time-series data")
    parser.add_argument("--input_dir", default="data/split", help="Directory containing the split CSVs")
    parser.add_argument("--output_dir", default="data/devices", help="Directory to save engineered features")
    
    args = parser.parse_args()
    
    aggregate_and_engineer(args.input_dir, args.output_dir)
