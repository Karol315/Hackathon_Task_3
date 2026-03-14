import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class GlobalLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        
        # Monthly Average Prediction (Single Scalar)
        self.fc_mean = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        # Take the last time step output to predict the monthly average
        last_out = out[:, -1, :]
        mean = self.fc_mean(last_out)
        return mean

class GlobalTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Monthly Average Prediction (Single Scalar)
        self.fc_mean = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        out = self.transformer_encoder(x)
        
        last_out = out[:, -1, :] 
        mean = self.fc_mean(last_out)
        return mean
