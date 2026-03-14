import os
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from src.model import GlobalLSTM, GlobalTransformer
from src.trainer import MonthlyTrainer

class PumpDataset(Dataset):
    def __init__(self, df, feature_cols, target_col, seq_length=30):
        self.seq_length = seq_length
        self.features = []
        self.targets = []
        
        # Group by device to avoid mixing sequences between pumps
        for _, group in df.groupby('deviceId'):
            # Ensure sort for sequence
            group = group.sort_values('timedate')
            # Handle NaNs and Infs per group
            group = group.replace([np.inf, -np.inf], np.nan)
            group[feature_cols] = group[feature_cols].ffill().bfill().fillna(0)
            group[target_col] = group[target_col].ffill().bfill().fillna(0)
            
            f_data = group[feature_cols].values
            t_data = group[target_col].values
            
            for i in range(len(group) - seq_length):
                feat_seq = f_data[i:i+seq_length]
                target_val = t_data[i+seq_length]
                
                # Double check for any remaining NaNs in the sequence
                if not np.isnan(feat_seq).any() and not np.isnan(target_val):
                    self.features.append(feat_seq)
                    self.targets.append(target_val)
                
        if self.features:
            self.features = torch.tensor(np.array(self.features), dtype=torch.float32)
            self.targets = torch.tensor(np.array(self.targets), dtype=torch.float32).view(-1, 1)
        else:
            self.features = torch.empty(0)
            self.targets = torch.empty(0)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

def main():
    parser = argparse.ArgumentParser(description="Monthly Average Training for Hackathon Task 3")
    parser.add_argument('--model', type=str, default='lstm', choices=['lstm', 'transformer'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    data_dir = 'data'
    train_path = os.path.join(data_dir, 'train_processed.parquet')
    val_path = os.path.join(data_dir, 'val_processed.parquet')
    
    if not os.path.exists(train_path):
        print("Processed data not found! Please run scripts/prepare_data.py first.")
        return

    print("Loading processed parquets...")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    
    # Define features to exclude (non-numerical or targets)
    exclude = ['deviceId', 'timedate', 'date', 'latitude', 'longitude', 'lat_round', 'lon_round', 'x2', 'period']
    # Target is the raw x2 (or the log_return, but user usually wants to predict the level)
    # However, for stationarity, we might predict diff. 
    # Let's use 'x2' as standard target but we've added 'x2_diff' and 'x2_log_return' as features.
    target_col = 'x2'
    
    feature_cols = [c for c in train_df.columns if c not in exclude]
    print(f"Using {len(feature_cols)} features.")

    # Sequence parameters
    SEQ_LENGTH = 24 # 2 hours of history if 5-min intervals
    
    print("Creating PyTorch Datasets...")
    train_ds = PumpDataset(train_df, feature_cols, target_col, seq_length=SEQ_LENGTH)
    val_ds = PumpDataset(val_df, feature_cols, target_col, seq_length=SEQ_LENGTH)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    
    # Model Init
    input_dim = len(feature_cols)
    hidden_dim = 128
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if args.model == 'lstm':
        model = GlobalLSTM(input_dim, hidden_dim)
    else:
        model = GlobalTransformer(input_dim, d_model=hidden_dim, nhead=8)
        
    trainer = MonthlyTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.lr,
        device=device
    )
    
    print(f"Starting Training ({args.model.upper()})...")
    history = trainer.train(epochs=args.epochs)
    
    # Save Model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), f'models/{args.model}_monthly.pth')
    print("Training Complete!")

if __name__ == "__main__":
    main()
