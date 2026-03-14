import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class MonthlyTrainer:
    def __init__(self, model, train_loader, val_loader, learning_rate=1e-3, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Mean Absolute Error as requested/standard for energy
        self.criterion = nn.L1Loss()

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(x)
            
            loss = self.criterion(output, y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                loss = self.criterion(output, y)
                total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def train(self, epochs=50):
        history = {'train_loss': [], 'val_loss': []}
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs} - Train MAE: {train_loss:.4f} - Val MAE: {val_loss:.4f}")
            
        return history
