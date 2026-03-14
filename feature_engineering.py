import pandas as pd
import numpy as np
import glob
import os
import argparse
from pathlib import Path

def create_features_for_device(df):
    """
    Apply feature engineering to a single device's dataframe.
    Assumes df is sorted by Timedate.
    """
    # Parse Timedate
    df['Timedate'] = pd.to_datetime(df['Timedate'])
    df = df.sort_values(by='Timedate')
    
    # 1. Row-level (time-based) features
    df['hour'] = df['Timedate'].dt.hour
    df['dayofweek'] = df['Timedate'].dt.dayofweek
    df['month'] = df['Timedate'].dt.month
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    # Cyclic encoding for time features (hour and month)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
    
    # 2. Time-series features (Rolling window features)
    # We will compute rolling means and stds for important temperature and load features
    # Assuming standard 5-minute intervals: 12 intervals = 1 hour, 288 = 24 hours
    
    # Select columns to compute rolling features on (based on task3.pdf)
    target_cols = ['t1', 't2', 't5', 'x1'] # External temp, Internal temp, Load HEX temp, Operating freq
    
    # Ensure they exist in df
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

    # 3. Target specific features (x2)
    if 'x2' in df.columns:
        df['x2_roll_mean_1h'] = df['x2'].rolling(window=12, min_periods=1).mean()
        df['x2_lag_24h'] = df['x2'].shift(288)

    return df

def aggregate_and_engineer(input_dir, output_dir):
    """
    Since device data can span multiple split files, we must gather all data for a specific device,
    sort it by time, engineer features, and then save it.
    
    To avoid memory issues, we will first bucket data by deviceId into intermediate files,
    then process each device file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    temp_dir = os.path.join(output_dir, 'temp_by_device')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    split_files = glob.glob(os.path.join(input_dir, '*.csv'))
    print(f"Found {len(split_files)} split files.")
    
    # Step 1: Bucket by deviceId
    print("Step 1: Bucketing data by deviceId...")
    
    # Keep track of known devices to append
    known_devices = set()
    
    for i, file in enumerate(split_files):
        print(f"Processing split {i}/{len(split_files)} for bucketing...")
        chunk = pd.read_csv(file)
        
        if 'deviceId' not in chunk.columns:
            print(f"Skipping {file} - no deviceId column.")
            continue
            
        # Group by deviceId and append to device-specific files
        for device_id, group in chunk.groupby('deviceId'):
            safe_device_id = str(device_id).replace('/', '_').replace('\\', '_')
            out_file = os.path.join(temp_dir, f"{safe_device_id}.csv")
            
            # If we've seen this device before, append without header. Otherwise write with header.
            mode = 'a' if device_id in known_devices else 'w'
            header = not (device_id in known_devices)
            
            group.to_csv(out_file, mode=mode, header=header, index=False)
            known_devices.add(device_id)

    # Step 2: Feature Engineering per device
    print(f"Step 2: Feature engineering for {len(known_devices)} devices...")
    device_files = glob.glob(os.path.join(temp_dir, '*.csv'))
    
    final_output_dir = os.path.join(output_dir, 'engineered')
    if not os.path.exists(final_output_dir):
        os.makedirs(final_output_dir)
        
    for i, file in enumerate(device_files):
        if i % 50 == 0:
            print(f"Engineering device {i}/{len(device_files)}...")
            
        df = pd.read_csv(file)
        df_engineered = create_features_for_device(df)
        
        out_file = os.path.join(final_output_dir, os.path.basename(file))
        df_engineered.to_csv(out_file, index=False)
        
    print("Done! Engineered data is saved by-device in:", final_output_dir)
    print("You may want to clean up the temporary bucketing directory:", temp_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature Engineering for split time-series data")
    parser.add_argument("--input_dir", default="data/split", help="Directory containing the split CSVs")
    parser.add_argument("--output_dir", default="data/devices", help="Directory to save engineered features")
    
    args = parser.parse_args()
    
    aggregate_and_engineer(args.input_dir, args.output_dir)
