import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter
from statsmodels.tsa.stattools import pacf

class FeaturePipeline:
    def __init__(self):
        self.global_stats = {}
        self.fitted = False

    def fit(self, train_df):
        """
        Calculates global statistics ONLY on the training set to avoid leakage.
        """
        # Calculate global means/stds for normalization
        self.global_stats['x2_mean'] = train_df['x2'].mean()
        self.global_stats['x2_std'] = train_df['x2'].std()
        
        # Determine optimal lags using PACF on a sample
        sample_device = train_df['deviceId'].iloc[0]
        # Użyto .ffill() zamiast fillna(method='ffill')
        sample_series = train_df[train_df['deviceId'] == sample_device]['x2'].ffill().dropna()
        
        if len(sample_series) > 50:
            pacf_vals = pacf(sample_series, nlags=20)
            optimal_lags = np.where(np.abs(pacf_vals) > 0.2)[0].tolist()
            # Remove 0 lag if it exists (correlation with itself is always 1)
            if 0 in optimal_lags:
                optimal_lags.remove(0)
            
            # Fallback if no lags > 0.2 were found
            self.global_stats['optimal_lags'] = optimal_lags if optimal_lags else [1, 12, 24]
        else:
            self.global_stats['optimal_lags'] = [1, 12, 24] # Defaults
            
        self.fitted = True

    def transform(self, df):
        """
        Applies transformations to the dataframe.
        """
        if not self.fitted:
            raise ValueError("Pipeline must be fitted on training data first.")
            
        df = df.copy()
        df['timedate'] = pd.to_datetime(df['timedate'])
        df = df.sort_values(['deviceId', 'timedate'])
        
        # 1. Stationarity & Transformations
        # Log-returns for x2 (energy consumption)
        # We add 1 to avoid log(0) - assuming x2 >= 0
        df['x2_log'] = np.log1p(df['x2'].clip(lower=0)) # Zabezpieczenie przed wartościami ujemnymi
        df['x2_diff'] = df.groupby('deviceId')['x2'].diff()
        df['x2_log_return'] = df.groupby('deviceId')['x2_log'].diff()
        
        # 2. Volatility Features (ARCH/GARCH inspiration)
        for window in [12, 144]: 
            df[f'x2_volatility_{window}'] = df.groupby('deviceId')['x2_log_return'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
            
        # 3. Filtering (Spectral Analysis inspiration)
        def apply_butter_lowpass(series, cutoff=0.1, fs=1.0, order=5):
            # Fill missing values to avoid filter errors
            series_filled = series.ffill().bfill()
            if len(series_filled) < max(20, order * 3): # Zabezpieczenie długości sygnału
                return series
            try:
                b, a = butter(order, cutoff, fs=fs, btype='low', analog=False)
                return lfilter(b, a, series_filled)
            except Exception:
                return series # Fallback w razie np. braku zmienności

        # Applying filter per device
        df['x2_trend'] = df.groupby('deviceId')['x2'].transform(apply_butter_lowpass)
        
        # 4. Lag Features (PACF based)
        for lag in self.global_stats['optimal_lags']:
            df[f'x2_lag_{lag}'] = df.groupby('deviceId')['x2'].shift(lag)
            
        # 5. Time Features & Events
        df['hour'] = df['timedate'].dt.hour
        df['day_of_week'] = df['timedate'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # 6. Global stats integration (Row-wise)
        # Handle case where std might be 0 to avoid division by zero
        std_val = self.global_stats['x2_std'] if self.global_stats['x2_std'] != 0 else 1.0
        df['x2_norm'] = (df['x2'] - self.global_stats['x2_mean']) / std_val
        
        return df

    def merge_weather(self, df, weather_df):
        """
        Fuzzy merge weather data based on coordinates and date.
        """
        df = df.copy()
        weather_df = weather_df.copy()
        
        df['date'] = df['timedate'].dt.date
        weather_df['date'] = pd.to_datetime(weather_df['date']).dt.date
        
        # Round coordinates for fuzzy matching
        df['lat_round'] = df['latitude'].round(1)
        df['lon_round'] = df['longitude'].round(1)
        weather_df['lat_round'] = weather_df['latitude'].round(1)
        weather_df['lon_round'] = weather_df['longitude'].round(1)
        
        # Drop raw lat/lon from weather to avoid conflict
        # Upewniamy się, że drop_duplicates działa na czystych danych
        weather_df_clean = weather_df.drop(columns=['latitude', 'longitude'], errors='ignore')
        weather_df_clean = weather_df_clean.drop_duplicates(['date', 'lat_round', 'lon_round'])
        
        # Merge
        df = pd.merge(df, weather_df_clean, on=['date', 'lat_round', 'lon_round'], how='left')
        
        # Fill missing weather
        weather_cols = [c for c in weather_df_clean.columns if c not in ['date', 'lat_round', 'lon_round']]
        
        # Grupowe ffill() i bfill() w nowym standardzie Pandas
        df[weather_cols] = df.groupby('deviceId')[weather_cols].transform(
            lambda x: x.ffill().bfill()
        )
        
        return df