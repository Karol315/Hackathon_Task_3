import os
import pandas as pd
import numpy as np
import pickle
import sys
import shutil

# Dodanie folderu nadrzędnego do ścieżki
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import FeaturePipeline

def process_and_save_set(name, folder, weather_df, pipeline, data_dir, filename):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.parquet')]
    if not files:
        print(f"No data for {name}")
        return
    
    print(f"Concatenating and transforming {name} set...")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df['timedate'] = pd.to_datetime(df['timedate'])
    
    # Transform
    df = pipeline.merge_weather(df, weather_df)
    df = pipeline.transform(df)
    
    print(f"Saving {name} to {filename}...")
    df.to_parquet(os.path.join(data_dir, filename), index=False)
    del df

def main():
    data_dir = 'data'
    data_path = os.path.join(data_dir, 'data.csv') 
    devices_path = os.path.join(data_dir, 'devices.csv')
    weather_path = os.path.join(data_dir, 'weather_daily_updates.csv')
    pipeline_path = os.path.join(data_dir, 'pipeline.pkl')
    
    tmp_dir = os.path.join(data_dir, 'tmp_process')
    folders = {
        'train': os.path.join(tmp_dir, 'train'),
        'valid': os.path.join(tmp_dir, 'valid'),
        'test': os.path.join(tmp_dir, 'test')
    }

    # Reset tmp dirs
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    for p in folders.values():
        os.makedirs(p, exist_ok=True)

    print("Loading metadata...")
    devices_df = pd.read_csv(devices_path)
    weather_df = pd.read_csv(weather_path)
    
    print(f"Reading {data_path} - flushing chunks to disk...")
    chunk_size = 500000
    try:
        reader = pd.read_csv(data_path, chunksize=chunk_size)
        for i, chunk in enumerate(reader):
            # Print status every 10 chunks
            if i % 10 == 0:
                print(f"Processing chunk {i}...")
            
            chunk = pd.merge(chunk, devices_df, on='deviceId', how='left')
            
            for period, folder in folders.items():
                part = chunk[chunk['period'] == period]
                if not part.empty:
                    part.to_parquet(os.path.join(folder, f"{i}.parquet"))
            
            del chunk # Aggressively free memory
            
    except Exception as e:
        print(f"Error during chunked reading: {e}")
        return

    # Initialize / Load Pipeline
    pipeline = FeaturePipeline()
    
    # To fit the pipeline, we need the train set in memory once
    train_files = [os.path.join(folders['train'], f) for f in os.listdir(folders['train'])]
    if train_files:
        print("Loading TRAIN set for fitting...")
        train_df = pd.concat([pd.read_parquet(f) for f in train_files], ignore_index=True)
        train_df['timedate'] = pd.to_datetime(train_df['timedate'])
        
        print("Fitting pipeline...")
        pipeline.fit(train_df)
        
        with open(pipeline_path, 'wb') as f:
            pickle.dump(pipeline, f)
        del train_df # Release memory
    else:
        if os.path.exists(pipeline_path):
            with open(pipeline_path, 'rb') as f:
                pipeline = pickle.load(f)

    # Process each set one by one to keep memory low
    process_and_save_set('TRAIN', folders['train'], weather_df, pipeline, data_dir, 'train_processed.parquet')
    process_and_save_set('VAL', folders['valid'], weather_df, pipeline, data_dir, 'val_processed.parquet')
    process_and_save_set('TEST', folders['test'], weather_df, pipeline, data_dir, 'test_processed.parquet')

    # Cleanup tmp
    shutil.rmtree(tmp_dir)
    print("Data Preparation Complete!")

if __name__ == "__main__":
    main()