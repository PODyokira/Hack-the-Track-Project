"""
LSTM Model for Predictive Forecasting
Predicts tire degradation, lap times, and future performance metrics
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
from pathlib import Path
import pickle

class TimeSeriesDataset(Dataset):
    """Dataset for time-series sequences"""
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class LSTMPredictor(nn.Module):
    """LSTM model for time-series prediction"""
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_dim=1, dropout=0.2):
        super(LSTMPredictor, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # Take the last output
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

class TireDegradationPredictor:
    """Predict tire degradation and lap time changes"""
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = LSTMPredictor(input_dim, hidden_dim, num_layers, output_dim=1).to(self.device)
        self.scaler_features = StandardScaler()
        self.scaler_target = StandardScaler()
        self.input_dim = input_dim
    
    def train(self, sequences, targets, epochs=50, batch_size=64, lr=0.001):
        """Train the LSTM model"""
        print(f"Training LSTM on {len(sequences)} sequences...")
        
        # Reshape sequences for scaling: (n_samples * seq_len, n_features)
        n_samples, seq_len, n_features = sequences.shape
        sequences_reshaped = sequences.reshape(-1, n_features)
        
        # Scale features
        sequences_scaled = self.scaler_features.fit_transform(sequences_reshaped)
        sequences_scaled = sequences_scaled.reshape(n_samples, seq_len, n_features)
        
        # Scale targets
        targets_scaled = self.scaler_target.fit_transform(targets.reshape(-1, 1)).flatten()
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            sequences_scaled, targets_scaled, test_size=0.2, random_state=42
        )
        
        # Create data loaders
        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for batch_sequences, batch_targets in train_loader:
                batch_sequences = batch_sequences.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                optimizer.zero_grad()
                predictions = self.model(batch_sequences).squeeze()
                loss = criterion(predictions, batch_targets)
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                for batch_sequences, batch_targets in val_loader:
                    batch_sequences = batch_sequences.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
                    predictions = self.model(batch_sequences).squeeze()
                    loss = criterion(predictions, batch_targets)
                    val_loss += loss.item()
                    
                    val_predictions.extend(predictions.cpu().numpy())
                    val_targets.extend(batch_targets.cpu().numpy())
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            # Calculate MAE
            val_predictions = np.array(val_predictions)
            val_targets = np.array(val_targets)
            val_predictions_unscaled = self.scaler_target.inverse_transform(val_predictions.reshape(-1, 1)).flatten()
            val_targets_unscaled = self.scaler_target.inverse_transform(val_targets.reshape(-1, 1)).flatten()
            val_mae = mean_absolute_error(val_targets_unscaled, val_predictions_unscaled)
            
            scheduler.step(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val MAE: {val_mae:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        print(f"Training complete. Best validation loss: {best_val_loss:.6f}, Best MAE: {val_mae:.4f}")
        return best_val_loss
    
    def predict(self, sequences):
        """Predict future values"""
        self.model.eval()
        
        # Scale sequences
        n_samples, seq_len, n_features = sequences.shape
        sequences_reshaped = sequences.reshape(-1, n_features)
        sequences_scaled = self.scaler_features.transform(sequences_reshaped)
        sequences_scaled = sequences_scaled.reshape(n_samples, seq_len, n_features)
        
        sequences_tensor = torch.FloatTensor(sequences_scaled).to(self.device)
        
        with torch.no_grad():
            predictions_scaled = self.model(sequences_tensor).cpu().numpy()
        
        # Inverse transform
        predictions = self.scaler_target.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
        return predictions
    
    def predict_degradation(self, recent_laps_data, window_size=10):
        """Predict tire degradation for next lap based on recent laps"""
        if len(recent_laps_data) < window_size:
            # Pad with zeros if not enough data
            padding = np.zeros((window_size - len(recent_laps_data), recent_laps_data.shape[1]))
            recent_laps_data = np.vstack([padding, recent_laps_data])
        
        # Take last window_size samples
        sequence = recent_laps_data[-window_size:].reshape(1, window_size, -1)
        degradation = self.predict(sequence)[0]
        return degradation
    
    def save(self, model_path, scaler_path):
        """Save model and scalers"""
        torch.save(self.model.state_dict(), model_path)
        with open(scaler_path, 'wb') as f:
            pickle.dump({
                'feature_scaler': self.scaler_features,
                'target_scaler': self.scaler_target,
                'input_dim': self.input_dim
            }, f)
        print(f"LSTM model saved to {model_path}")
    
    def load(self, model_path, scaler_path, input_dim, hidden_dim=128, num_layers=2):
        """Load model and scalers"""
        self.model = LSTMPredictor(input_dim, hidden_dim, num_layers, output_dim=1).to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        with open(scaler_path, 'rb') as f:
            scalers = pickle.load(f)
            self.scaler_features = scalers['feature_scaler']
            self.scaler_target = scalers['target_scaler']
            self.input_dim = scalers['input_dim']
        print(f"LSTM model loaded from {model_path}")

def train_lstm_model(data_dir='processed_data', output_dir='models'):
    """Main function to train LSTM model"""
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load data
    print("Loading LSTM sequences...")
    sequences = np.load(data_path / 'lstm_sequences.npy')
    targets = np.load(data_path / 'lstm_targets.npy')
    
    print(f"Sequences shape: {sequences.shape}")
    print(f"Targets shape: {targets.shape}")
    
    # Get input dimension
    input_dim = sequences.shape[2]
    
    # Train model
    predictor = TireDegradationPredictor(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=2,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    predictor.train(sequences, targets, epochs=100, batch_size=64)
    
    # Save model
    predictor.save(
        output_path / 'lstm_model.pth',
        output_path / 'lstm_scalers.pkl'
    )
    
    # Save metadata
    metadata = {
        'input_dim': input_dim,
        'window_size': sequences.shape[1],
        'hidden_dim': 128,
        'num_layers': 2
    }
    with open(output_path / 'lstm_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return predictor

if __name__ == "__main__":
    train_lstm_model()



