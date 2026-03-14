import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class PumpTimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for Time Series Forecasting.
    Groups data by device to avoid mixing sequences between different pumps.
    Yields sliding windows of features to predict a target variable (e.g., monthly average).
    """
    def __init__(self, df: pd.DataFrame, target_col: str, device_col: str, 
                 sequence_length: int = 30, feature_cols: list = None):
        """
        Args:
            df (pd.DataFrame): The dataframe containing all features and targets.
            target_col (str): The name of the column containing the target to predict.
            device_col (str): The name of the column identifying the pump/device.
            sequence_length (int): Number of time steps to look back.
            feature_cols (list, optional): Columns to use as input features. 
                                           If None, uses all columns except target_col and device_col.
        """
        super().__init__()
        self.sequence_length = sequence_length
        self.device_col = device_col
        self.target_col = target_col
        
        if feature_cols is None:
            self.feature_cols = [col for col in df.columns if col not in [target_col, device_col]]
        else:
            self.feature_cols = feature_cols
            
        self.samples = []
        self._prepare_data(df)
        
    def _prepare_data(self, df):
        # We process each device independently
        for device_id, group in df.groupby(self.device_col):
            # Ensure the data is sorted by time if there's a time column
            # For simplicity, we assume incoming DataFrame rows are already correctly ordered chronologically.
            
            features = group[self.feature_cols].values
            targets = group[self.target_col].values
            
            # Create sliding windows
            num_samples = len(group) - self.sequence_length
            if num_samples <= 0:
                continue
                
            for i in range(num_samples):
                # Input sequence: from index i to i + sequence_length
                x_seq = features[i : i + self.sequence_length]
                # Target: corresponding to the step following the sequence 
                # (or aggregated target aligned with the sequence end).
                y_val = targets[i + self.sequence_length]
                
                self.samples.append((x_seq, y_val))
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
