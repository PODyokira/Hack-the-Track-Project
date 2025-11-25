"""
Isolation Forest Model for Anomaly Detection
Detects driver fatigue, loss of focus, or inconsistent driving patterns
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class DriverAnomalyDetector:
    """Isolation Forest based anomaly detector for driver behavior"""
    def __init__(self, contamination=0.1, n_estimators=100, random_state=42):
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def train(self, data, driver_id=None):
        """Train the anomaly detection model"""
        print(f"Training anomaly detector on {len(data)} samples...")
        
        if driver_id:
            print(f"Training on driver {driver_id} data only")
            data = data[data.get('vehicle_number', None) == driver_id] if 'vehicle_number' in data.columns else data
        
        # Scale features
        data_scaled = self.scaler.fit_transform(data)
        
        # Train Isolation Forest
        self.model.fit(data_scaled)
        self.is_fitted = True
        
        # Get predictions on training data
        predictions = self.model.predict(data_scaled)
        n_anomalies = np.sum(predictions == -1)
        anomaly_rate = n_anomalies / len(predictions)
        
        print(f"Training complete. Detected {n_anomalies} anomalies ({anomaly_rate*100:.2f}%)")
        return predictions
    
    def predict(self, data):
        """Predict anomalies in new data"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        # Scale features
        data_scaled = self.scaler.transform(data)
        
        # Predict anomalies (-1 = anomaly, 1 = normal)
        predictions = self.model.predict(data_scaled)
        
        # Get anomaly scores (lower = more anomalous)
        scores = self.model.score_samples(data_scaled)
        
        return predictions, scores
    
    def detect_driver_fatigue(self, recent_data, threshold_percentile=10):
        """Detect if driver is showing signs of fatigue based on recent data"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before detection")
        
        predictions, scores = self.predict(recent_data)
        
        # Calculate threshold based on percentile
        threshold = np.percentile(scores, threshold_percentile)
        
        # Count anomalies in recent data
        n_anomalies = np.sum(predictions == -1)
        anomaly_rate = n_anomalies / len(predictions)
        
        # Average anomaly score
        avg_score = np.mean(scores)
        
        # Fatigue indicators
        is_fatigue = anomaly_rate > 0.15 or avg_score < threshold
        
        return {
            'is_fatigue': is_fatigue,
            'anomaly_rate': anomaly_rate,
            'avg_anomaly_score': avg_score,
            'n_anomalies': n_anomalies,
            'threshold': threshold
        }
    
    def analyze_consistency(self, data, window_size=50):
        """Analyze driving consistency over time"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before analysis")
        
        predictions, scores = self.predict(data)
        
        # Calculate rolling consistency
        consistency_scores = []
        for i in range(0, len(scores), window_size):
            window_scores = scores[i:i+window_size]
            # Higher score = more consistent
            consistency = np.mean(window_scores)
            consistency_scores.append(consistency)
        
        return {
            'consistency_scores': consistency_scores,
            'overall_consistency': np.mean(consistency_scores),
            'consistency_std': np.std(consistency_scores),
            'anomaly_predictions': predictions,
            'anomaly_scores': scores
        }
    
    def get_anomaly_features(self, data, predictions):
        """Identify which features contribute most to anomalies"""
        anomaly_mask = predictions == -1
        normal_mask = predictions == 1
        
        if np.sum(anomaly_mask) == 0:
            return {}
        
        anomaly_data = data[anomaly_mask]
        normal_data = data[normal_mask]
        
        feature_contributions = {}
        for col in data.columns:
            anomaly_mean = np.mean(anomaly_data[col])
            normal_mean = np.mean(normal_data[col])
            contribution = abs(anomaly_mean - normal_mean) / (abs(normal_mean) + 1e-6)
            feature_contributions[col] = {
                'contribution': contribution,
                'anomaly_mean': anomaly_mean,
                'normal_mean': normal_mean
            }
        
        # Sort by contribution
        sorted_features = sorted(
            feature_contributions.items(),
            key=lambda x: x[1]['contribution'],
            reverse=True
        )
        
        return dict(sorted_features)
    
    def save(self, model_path, scaler_path):
        """Save model and scaler"""
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Anomaly detector saved to {model_path}")
    
    def load(self, model_path, scaler_path):
        """Load model and scaler"""
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        self.is_fitted = True
        print(f"Anomaly detector loaded from {model_path}")

def train_anomaly_detector(data_dir='processed_data', output_dir='models', driver_id=None):
    """Main function to train anomaly detection model"""
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load data
    print("Loading anomaly detection data...")
    anomaly_data = pd.read_csv(data_path / 'anomaly_detection_data.csv')
    
    print(f"Data shape: {anomaly_data.shape}")
    print(f"Features: {anomaly_data.columns.tolist()}")
    
    # Remove NaN rows
    anomaly_data = anomaly_data.dropna()
    print(f"Data after removing NaN: {anomaly_data.shape}")
    
    # Train model
    detector = DriverAnomalyDetector(contamination=0.1, n_estimators=100)
    predictions = detector.train(anomaly_data, driver_id=driver_id)
    
    # Analyze feature contributions
    feature_contributions = detector.get_anomaly_features(anomaly_data, predictions)
    print("\nTop features contributing to anomalies:")
    for feature, info in list(feature_contributions.items())[:5]:
        print(f"  {feature}: contribution={info['contribution']:.4f}")
    
    # Save model
    detector.save(
        output_path / 'anomaly_detector.pkl',
        output_path / 'anomaly_scaler.pkl'
    )
    
    # Save feature contributions
    contributions_dict = {
        k: {
            'contribution': float(v['contribution']),
            'anomaly_mean': float(v['anomaly_mean']),
            'normal_mean': float(v['normal_mean'])
        }
        for k, v in feature_contributions.items()
    }
    with open(output_path / 'anomaly_feature_contributions.json', 'w') as f:
        json.dump(contributions_dict, f, indent=2)
    
    return detector

if __name__ == "__main__":
    train_anomaly_detector()



