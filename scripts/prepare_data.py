import os
import pandas as pd
import numpy as np
import pickle
import sys
import gc

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import FeaturePipeline

def main():
    data_dir = 'data'
    data_path = os.path.join(data_dir, 'data.csv') 
    devices_path = os.path.join(data_dir, 'devices.csv')
    weather_path = os.path.join(data_dir, 'weather_daily_updates.csv')
    pipeline_path = os.path.join(data_dir, 'pipeline.pkl')
    
    print("Loading metadata...")
    devices_df = pd.read_csv(devices_path)
    weather_df = pd.read_csv(weather_path)
    
    pipeline = FeaturePipeline()
    
    print("Reading first chunk to fit pipeline...")
    first_chunk = pd.read_csv(data_path, nrows=100000)
    first_chunk = pd.merge(first_chunk, devices_df, on='deviceId', how='left')
    train_sample = first_chunk[first_chunk['period'] == 'train']
    if not train_sample.empty:
        pipeline.fit(train_sample)
        with open(pipeline_path, 'wb') as f:
            pickle.dump(pipeline, f)
    else:
        print("No train data in first chunk, fitting on dummy data.")
        pipeline.means['x2'] = 0
        pipeline.stds['x2'] = 1
        pipeline.fitted = True

    del first_chunk, train_sample
    gc.collect()

    # Zamiast trzymać w pamięci układ folderów, będziemy zapisywać do pojedynczych plików parquet w trybie 'append'
    # (Fastparquet to obsługuje, po prostu zapisujemy chunks)
    
    train_path = os.path.join(data_dir, 'train_processed.parquet')
    val_path = os.path.join(data_dir, 'val_processed.parquet')
    test_path = os.path.join(data_dir, 'test_processed.parquet')
    
    for p in [train_path, val_path, test_path]:
        if os.path.exists(p):
            os.remove(p)

    print(f"Reading {data_path} in safe chunks of 100k...")
    chunk_size = 100000
    
    try:
        reader = pd.read_csv(data_path, chunksize=chunk_size, engine='c')
        for i, chunk in enumerate(reader):
            if i % 10 == 0:
                print(f"Processing chunk {i}...")
            
            chunk = pd.merge(chunk, devices_df, on='deviceId', how='left')
            chunk = pipeline.merge_weather(chunk, weather_df)
            chunk = pipeline.transform(chunk)
            
            # Zrzut od razu do docelowych plików z dopisywaniem (append)
            train_part = chunk[chunk['period'] == 'train']
            val_part = chunk[chunk['period'] == 'valid']
            test_part = chunk[chunk['period'] == 'test']
            
            if not train_part.empty:
                train_part.to_parquet(train_path, engine='fastparquet', append=os.path.exists(train_path))
            if not val_part.empty:
                val_part.to_parquet(val_path, engine='fastparquet', append=os.path.exists(val_path))
            if not test_part.empty:
                test_part.to_parquet(test_path, engine='fastparquet', append=os.path.exists(test_path))
                
            del chunk, train_part, val_part, test_part
            gc.collect() # Wymuszenie czyszczenia
            
    except Exception as e:
        print(f"Error during chunked reading: {e}")
        return

    print("Data Preparation Complete - Data written to final Parquet files natively!")

if __name__ == "__main__":
    main()