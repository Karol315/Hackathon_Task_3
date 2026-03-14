import pandas as pd
import numpy as np

class FeaturePipeline:
    def __init__(self):
        self.fitted = False
        self.means = {}
        self.stds = {}

    def fit(self, train_df):
        # Proste statystyki do normowania, zamiast całego analizowania PACF
        self.means['x2'] = train_df['x2'].mean()
        self.stds['x2'] = train_df['x2'].std() if train_df['x2'].std() != 0 else 1.0
        self.fitted = True

    def transform(self, df):
        if not self.fitted:
            raise ValueError("Pipeline must be fitted on training data first.")
            
        df = df.copy()
        df['timedate'] = pd.to_datetime(df['timedate'])
        
        # Sortowanie wewnątrz urządzeń jest kluczowe, ale robimy to szybko
        df = df.sort_values(['deviceId', 'timedate'])
        
        # 1. Proste transformacje
        df['x2_log'] = np.log1p(df['x2'].clip(lower=0))
        df['x2_diff'] = df.groupby('deviceId')['x2'].diff().fillna(0)
        
        # 2. Czas
        df['hour'] = df['timedate'].dt.hour
        df['day_of_week'] = df['timedate'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # 3. Lags (Maksymalnie obcięte, żeby nie marnować pamięci)
        for lag in [1, 24]:
            df[f'x2_lag_{lag}'] = df.groupby('deviceId')['x2'].shift(lag).fillna(0)
            
        # 4. Normalizacja
        df['x2_norm'] = (df['x2'] - self.means['x2']) / self.stds['x2']
        
        return df

    def merge_weather(self, df, weather_df):
        df = df.copy()
        weather_df = weather_df.copy()
        
        df['date'] = df['timedate'].dt.date
        weather_df['date'] = pd.to_datetime(weather_df['date']).dt.date
        
        df['lat_round'] = df['latitude'].round(1)
        df['lon_round'] = df['longitude'].round(1)
        weather_df['lat_round'] = weather_df['latitude'].round(1)
        weather_df['lon_round'] = weather_df['longitude'].round(1)
        
        weather_df_clean = weather_df.drop(columns=['latitude', 'longitude'], errors='ignore')
        weather_df_clean = weather_df_clean.drop_duplicates(['date', 'lat_round', 'lon_round'])
        
        df = pd.merge(df, weather_df_clean, on=['date', 'lat_round', 'lon_round'], how='left')
        
        # Szybkie uzupełnianie braków (tylko ffill, bez groupby jeśli to możliwe, ale dla bezpieczeństwa można użyć bfill na całości)
        weather_cols = [c for c in weather_df_clean.columns if c not in ['date', 'lat_round', 'lon_round']]
        df[weather_cols] = df[weather_cols].ffill().bfill().fillna(0)
        
        return df