import os
import pandas as pd
import numpy as np
import pickle
import sys

# Dodanie folderu nadrzędnego do ścieżki
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import FeaturePipeline

def main():
    data_dir = 'data'
    data_path = os.path.join(data_dir, 'data.csv') 
    devices_path = os.path.join(data_dir, 'devices.csv')
    weather_path = os.path.join(data_dir, 'weather_daily_updates.csv')
    pipeline_path = os.path.join(data_dir, 'pipeline.pkl')
    
    print("Loading datasets...")
    devices_df = pd.read_csv(devices_path)
    weather_df = pd.read_csv(weather_path)
    
    print(f"Reading {data_path} in chunks to avoid OOM...")
    train_chunks = []
    val_chunks = []
    test_chunks = []
    
    # Process in chunks of 500,000 rows
    chunk_size = 500000
    try:
        reader = pd.read_csv(data_path, chunksize=chunk_size)
        for i, chunk in enumerate(reader):
            print(f"Processing chunk {i+1}...")
            
            # Merge with devices to get coordinates/metadata
            chunk = pd.merge(chunk, devices_df, on='deviceId', how='left')
            
            # Split by period immediately to save memory
            train_chunks.append(chunk[chunk['period'] == 'train'].copy())
            val_chunks.append(chunk[chunk['period'] == 'valid'].copy())
            test_chunks.append(chunk[chunk['period'] == 'test'].copy())
            
            # Optional: if memory is still an issue, we could save these to temp local parquets here
            
    except Exception as e:
        print(f"Error during chunked reading: {e}")
        return

    print("Concatenating processed chunks...")
    train_df = pd.concat(train_chunks, ignore_index=True) if train_chunks else pd.DataFrame()
    val_df = pd.concat(val_chunks, ignore_index=True) if val_chunks else pd.DataFrame()
    test_df = pd.concat(test_chunks, ignore_index=True) if test_chunks else pd.DataFrame()
    
    # Free memory
    del train_chunks, val_chunks, test_chunks
    
    print(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    print("Converting timedate...")
    for df in [train_df, val_df, test_df]:
        if not df.empty:
            df['timedate'] = pd.to_datetime(df['timedate'])
    
    pipeline = FeaturePipeline()
    
    if not train_df.empty:
        print("Fitting pipeline on training data...")
        # Use a subset for fitting PACF if train_df is too large
        pipeline.fit(train_df)
        
        print("Saving fitted Pipeline...")
        with open(pipeline_path, 'wb') as f:
            pickle.dump(pipeline, f)
    else:
        if os.path.exists(pipeline_path):
            print(f"Loading pre-fitted Pipeline from {pipeline_path}...")
            with open(pipeline_path, 'rb') as f:
                pipeline = pickle.load(f)
        else:
            print("WARNING: No train data and no saved pipeline!")

    # Process and save
    sets = [('TRAIN', train_df, 'train_processed.parquet'), 
            ('VAL', val_df, 'val_processed.parquet'), 
            ('TEST', test_df, 'test_processed.parquet')]
            
    for name, df, filename in sets:
        if not df.empty:
            print(f"Transforming and saving {name} set...")
            df = pipeline.merge_weather(df, weather_df)
            df = pipeline.transform(df)
            df.to_parquet(os.path.join(data_dir, filename), index=False)
            del df # Free memory immediately
        
    print("Data Preparation Complete!")

if __name__ == "__main__":
    main()