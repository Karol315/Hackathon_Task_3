import os
import argparse
import pandas as pd
import torch
import pickle
from torch.utils.data import DataLoader, Dataset
from src.model import GlobalLSTM, GlobalTransformer

class InferenceDataset(Dataset):
    def __init__(self, df, feature_cols, seq_length=24):
        self.seq_length = seq_length
        self.features = []
        self.metadata = [] # To keep track of device and time
        
        for device_id, group in df.groupby('deviceId'):
            group = group.sort_values('timedate')
            f_data = group[feature_cols].values
            times = group['timedate'].values
            
            # For monthly prediction, we might want the last sequence of each month
            # or just all sequences and then aggregate. 
            # The prompt says "vector of outputs that can be merged to task3.pdf post"
            # task3.pdf usually has [deviceId, month, predicted_x2]
            
            group['year_month'] = pd.to_datetime(group['timedate']).dt.to_period('M')
            for month, m_group in group.groupby('year_month'):
                if len(m_group) >= seq_length:
                    # Take the last sequence of the month for prediction
                    last_idx = len(m_group)
                    feat_seq = f_data[m_group.index[-1]-seq_length+1 : m_group.index[-1]+1]
                    if len(feat_seq) == seq_length:
                        self.features.append(feat_seq)
                        self.metadata.append({'deviceId': device_id, 'month': str(month)})

        if self.features:
            self.features = torch.tensor(self.features, dtype=torch.float32)
        else:
            self.features = torch.empty(0)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], idx

def main():
    parser = argparse.ArgumentParser(description="Inference for Hackathon Task 3")
    parser.add_argument('--input', type=str, required=True, help='Path to CSV file (schema like data.csv)')
    parser.add_argument('--model_type', type=str, default='lstm', choices=['lstm', 'transformer'])
    parser.add_argument('--model_path', type=str, required=True, help='Path to .pth weights')
    parser.add_argument('--output', type=str, default='predictions.csv')
    args = parser.parse_args()

    # 1. Load context
    with open('data/pipeline.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    
    devices_df = pd.read_csv('data/devices.csv')
    weather_df = pd.read_csv('data/weather_daily_updates.csv')

    # 2. Load and Preprocess data
    print(f"Loading and processing {args.input}...")
    df = pd.read_csv(args.input)
    df = pd.merge(df, devices_df, on='deviceId', how='left')
    df['timedate'] = pd.to_datetime(df['timedate'])
    
    # Apply pipeline (it should handle fuzzy merge and transformations)
    df = pipeline.merge_weather(df, weather_df)
    df = pipeline.transform(df)
    
    # 3. Setup Dataset
    exclude = ['deviceId', 'timedate', 'date', 'latitude', 'longitude', 'lat_round', 'lon_round', 'x2', 'year_month']
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype != 'object']
    
    ds = InferenceDataset(df, feature_cols)
    if len(ds) == 0:
        print("No valid sequences found for inference.")
        return
        
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    # 4. Load Model
    input_dim = len(feature_cols)
    hidden_dim = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.model_type == 'lstm':
        model = GlobalLSTM(input_dim, hidden_dim)
    else:
        model = GlobalTransformer(input_dim, d_model=hidden_dim, nhead=8)
        
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # 5. Run Inference
    print("Running model inference...")
    results = []
    with torch.no_grad():
        for x, indices in loader:
            x = x.to(device)
            preds = model(x).cpu().numpy().flatten()
            for i, p in enumerate(preds):
                meta = ds.metadata[indices[i]]
                meta['predicted_x2_avg'] = p
                results.append(meta)

    # 6. Save Results
    res_df = pd.DataFrame(results)
    res_df.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")

if __name__ == "__main__":
    main()
