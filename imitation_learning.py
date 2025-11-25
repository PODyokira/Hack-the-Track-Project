"""
Imitation Learning Model - "Copy the Champion" Approach
Trains a deep neural network to mimic expert driver behavior
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
from pathlib import Path
import pickle

class RacingDataset(Dataset):
    """Dataset for racing telemetry data"""
    def __init__(self, states, actions):
        self.states = torch.FloatTensor(states)
        self.actions = torch.FloatTensor(actions)
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

class ImitationLearningModel(nn.Module):
    """Deep Neural Network for Imitation Learning"""
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256, 128]):
        super(ImitationLearningModel, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, action_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        return self.network(state)

class ImitationLearningTrainer:
    def __init__(self, state_dim, action_dim, device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = ImitationLearningModel(state_dim, action_dim).to(self.device)
        self.scaler_state = StandardScaler()
        self.scaler_action = StandardScaler()
        
    def train(self, states, actions, epochs=50, batch_size=256, lr=0.001):
        """Train the imitation learning model"""
        print(f"Training on {len(states)} expert demonstrations...")
        
        # Normalize states and actions
        states_scaled = self.scaler_state.fit_transform(states)
        actions_scaled = self.scaler_action.fit_transform(actions)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            states_scaled, actions_scaled, test_size=0.2, random_state=42
        )
        
        # Create data loaders
        train_dataset = RacingDataset(X_train, y_train)
        val_dataset = RacingDataset(X_val, y_val)
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
            for batch_states, batch_actions in train_loader:
                batch_states = batch_states.to(self.device)
                batch_actions = batch_actions.to(self.device)
                
                optimizer.zero_grad()
                predictions = self.model(batch_states)
                loss = criterion(predictions, batch_actions)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_states, batch_actions in val_loader:
                    batch_states = batch_states.to(self.device)
                    batch_actions = batch_actions.to(self.device)
                    
                    predictions = self.model(batch_states)
                    loss = criterion(predictions, batch_actions)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            scheduler.step(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        print(f"Training complete. Best validation loss: {best_val_loss:.6f}")
        return best_val_loss
    
    def predict(self, states):
        """Predict actions given states"""
        self.model.eval()
        states_scaled = self.scaler_state.transform(states)
        states_tensor = torch.FloatTensor(states_scaled).to(self.device)
        
        with torch.no_grad():
            actions_scaled = self.model(states_tensor).cpu().numpy()
        
        actions = self.scaler_action.inverse_transform(actions_scaled)
        return actions
    
    def save(self, model_path, scaler_path):
        """Save model and scalers"""
        torch.save(self.model.state_dict(), model_path)
        with open(scaler_path, 'wb') as f:
            pickle.dump({
                'state_scaler': self.scaler_state,
                'action_scaler': self.scaler_action
            }, f)
        print(f"Model saved to {model_path}")
    
    def load(self, model_path, scaler_path, state_dim, action_dim):
        """Load model and scalers"""
        self.model = ImitationLearningModel(state_dim, action_dim).to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        with open(scaler_path, 'rb') as f:
            scalers = pickle.load(f)
            self.scaler_state = scalers['state_scaler']
            self.scaler_action = scalers['action_scaler']
        print(f"Model loaded from {model_path}")

def train_imitation_learning_model(data_dir='processed_data', output_dir='models'):
    """Main function to train imitation learning model"""
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load data
    print("Loading imitation learning data...")
    il_data = pd.read_csv(data_path / 'imitation_learning_data.csv')
    
    with open(data_path / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    state_features = metadata['state_features']
    action_features = metadata['action_features']
    
    # Prepare data
    state_cols = [col for col in state_features if col in il_data.columns]
    action_cols = [col for col in action_features if col in il_data.columns]
    
    # Extract dataframes
    states_df = il_data[state_cols].copy()
    actions_df = il_data[action_cols].copy()
    
    # Convert all columns to numeric, coercing errors to NaN
    for col in states_df.columns:
        states_df[col] = pd.to_numeric(states_df[col], errors='coerce')
    for col in actions_df.columns:
        actions_df[col] = pd.to_numeric(actions_df[col], errors='coerce')
    
    # Remove NaN rows using pandas isna() which handles all types
    valid_mask = ~(states_df.isna().any(axis=1) | actions_df.isna().any(axis=1))
    states_df = states_df[valid_mask]
    actions_df = actions_df[valid_mask]
    
    # Convert to numpy arrays
    states = states_df.values.astype(np.float32)
    actions = actions_df.values.astype(np.float32)
    
    print(f"Training data: {len(states)} samples")
    print(f"State features: {state_cols}")
    print(f"Action features: {action_cols}")
    
    # Train model
    trainer = ImitationLearningTrainer(
        state_dim=len(state_cols),
        action_dim=len(action_cols),
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    trainer.train(states, actions, epochs=100, batch_size=256)
    
    # Save model
    trainer.save(
        output_path / 'imitation_learning_model.pth',
        output_path / 'imitation_learning_scalers.pkl'
    )
    
    # Save model metadata
    model_metadata = {
        'state_features': state_cols,
        'action_features': action_cols,
        'state_dim': len(state_cols),
        'action_dim': len(action_cols)
    }
    with open(output_path / 'il_model_metadata.json', 'w') as f:
        json.dump(model_metadata, f, indent=2)
    
    return trainer

if __name__ == "__main__":
    train_imitation_learning_model()

