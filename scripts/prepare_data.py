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
    data_path = os.path.join(data_dir, 'data.csv') # Zwróć uwagę, czy na pewno główny plik nazywa się test.csv
    devices_path = os.path.join(data_dir, 'devices.csv')
    weather_path = os.path.join(data_dir, 'weather_daily_updates.csv')
    pipeline_path = os.path.join(data_dir, 'pipeline.pkl')
    
    print("Loading datasets...")
    # Load devices mapping
    devices_df = pd.read_csv(devices_path)
    
    # Load weather
    weather_df = pd.read_csv(weather_path)
    
    print("Reading main data.csv (this might take a while)...")
    try:
        df_sample = pd.read_csv(data_path, nrows=100)
        print("Columns in main data:", df_sample.columns.tolist())
        df = pd.read_csv(data_path)
    except MemoryError:
        print("Memory Error! Switching to chunked processing...")
        # Placeholder for chunked processing logic
        return

    print("Merging devices metadata...")
    df = pd.merge(df, devices_df, on='deviceId', how='left')
    
    print("Converting timedate and splitting...")
    df['timedate'] = pd.to_datetime(df['timedate'])
    
    # Rozdzielenie danych na zbiory
    train_df = df[df['period'] == 'train'].copy()
    val_df = df[df['period'] == 'valid'].copy()
    test_df = df[df['period'] == 'test'].copy()
    
    print(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Initialize Pipeline
    pipeline = FeaturePipeline()
    
    # Logika dopasowania Pipeline'u (Fit)
    if not train_df.empty:
        print("Fitting pipeline on training data...")
        pipeline.fit(train_df)
        
        # Zapisz wyuczony Pipeline
        print("Saving fitted Pipeline to pickle...")
        with open(pipeline_path, 'wb') as f:
            pickle.dump(pipeline, f)
    else:
        print("No training data found in this batch.")
        # Jeśli nie ma danych treningowych, MUSIMY wczytać stary pipeline, żeby transformować resztę
        if os.path.exists(pipeline_path):
            print(f"Loading pre-fitted Pipeline from {pipeline_path}...")
            with open(pipeline_path, 'rb') as f:
                pipeline = pickle.load(f)
        else:
            print("WARNING: No fitted Pipeline found on disk and no train data to fit. Transformations will fail!")

    # Przetwarzanie i zapis zbioru TRAIN
    if not train_df.empty:
        print("Transforming and saving TRAIN set...")
        train_df = pipeline.merge_weather(train_df, weather_df)
        train_df = pipeline.transform(train_df)
        train_df.to_parquet(os.path.join(data_dir, 'train_processed.parquet'), index=False)
    
    # Przetwarzanie i zapis zbioru VAL
    if not val_df.empty:
        print("Transforming and saving VAL set...")
        val_df = pipeline.merge_weather(val_df, weather_df)
        val_df = pipeline.transform(val_df)
        val_df.to_parquet(os.path.join(data_dir, 'val_processed.parquet'), index=False)
    
    # Przetwarzanie i zapis zbioru TEST
    if not test_df.empty:
        print("Transforming and saving TEST set...")
        test_df = pipeline.merge_weather(test_df, weather_df)
        test_df = pipeline.transform(test_df)
        test_df.to_parquet(os.path.join(data_dir, 'test_processed.parquet'), index=False)
        
    print("Data Preparation Complete!")

if __name__ == "__main__":
    main()