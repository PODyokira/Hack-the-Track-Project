"""
Integrated Racing Strategy System
Combines Imitation Learning, LSTM, Isolation Forest, and PPO
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
import json
import pickle

from data_preprocessing import RacingDataPreprocessor
from imitation_learning import ImitationLearningTrainer
from lstm_predictor import TireDegradationPredictor
from anomaly_detection import DriverAnomalyDetector
from ppo_rl import PPOAgent, RacingEnvironment

class IntegratedRacingSystem:
    """Complete integrated system for race strategy and driver analysis"""
    
    def __init__(self, models_dir='models', data_dir='processed_data'):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        
        # Load all models
        self.il_model = None
        self.lstm_model = None
        self.anomaly_detector = None
        self.ppo_agent = None
        
        self.load_models()
    
    def load_models(self):
        """Load all trained models"""
        print("Loading integrated racing system models...")
        
        # Load IL model
        try:
            with open(self.models_dir / 'il_model_metadata.json', 'r') as f:
                il_metadata = json.load(f)
            
            self.il_model = ImitationLearningTrainer(
                state_dim=il_metadata['state_dim'],
                action_dim=il_metadata['action_dim']
            )
            self.il_model.load(
                self.models_dir / 'imitation_learning_model.pth',
                self.models_dir / 'imitation_learning_scalers.pkl',
                il_metadata['state_dim'],
                il_metadata['action_dim']
            )
            print("[OK] Imitation Learning model loaded")
        except Exception as e:
            print(f"[ERROR] Could not load IL model: {e}")
        
        # Load LSTM model (optional)
        try:
            with open(self.models_dir / 'lstm_metadata.json', 'r') as f:
                lstm_metadata = json.load(f)
            
            self.lstm_model = TireDegradationPredictor(
                input_dim=lstm_metadata['input_dim'],
                hidden_dim=lstm_metadata['hidden_dim'],
                num_layers=lstm_metadata['num_layers']
            )
            self.lstm_model.load(
                self.models_dir / 'lstm_model.pth',
                self.models_dir / 'lstm_scalers.pkl',
                lstm_metadata['input_dim'],
                lstm_metadata['hidden_dim'],
                lstm_metadata['num_layers']
            )
            print("[OK] LSTM predictor loaded")
        except Exception:
            # LSTM is optional - silently skip if not available
            self.lstm_model = None
        
        # Load Anomaly Detector (optional)
        try:
            self.anomaly_detector = DriverAnomalyDetector()
            self.anomaly_detector.load(
                self.models_dir / 'anomaly_detector.pkl',
                self.models_dir / 'anomaly_scaler.pkl'
            )
            print("[OK] Anomaly detector loaded")
        except Exception:
            # Anomaly detector is optional - silently skip if not available
            self.anomaly_detector = None
        
        # Load PPO Agent
        try:
            with open(self.models_dir / 'il_model_metadata.json', 'r') as f:
                il_metadata = json.load(f)
            
            self.ppo_agent = PPOAgent(
                state_dim=6,  # Environment state dim
                action_dim=2,  # [pit_decision, driving_aggression]
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            self.ppo_agent.load(self.models_dir / 'ppo_agent.pth')
            print("[OK] PPO agent loaded")
        except Exception as e:
            print(f"[ERROR] Could not load PPO agent: {e}")
        
        print("\nIntegrated system ready!")
    
    def predict_driving_action(self, state):
        """Predict optimal driving action using IL model"""
        if self.il_model is None:
            return None
        
        # State should be: [lateral_g, longitudinal_g, sector_S1, sector_S2, sector_S3]
        state_array = np.array([state]).reshape(1, -1)
        action = self.il_model.predict(state_array)[0]
        
        # Based on il_model_metadata.json, actions are: [throttle_position, Steering_Angle]
        return {
            'throttle_position': action[0] if len(action) > 0 else 0.5,
            'steering_angle': action[1] if len(action) > 1 else 0.0  # Steering_Angle is at index 1
        }
    
    def predict_tire_degradation(self, recent_telemetry):
        """Predict tire degradation using LSTM"""
        if self.lstm_model is None:
            return None
        
        try:
            # Convert telemetry to sequence format
            # This is simplified - in practice, you'd need proper feature extraction
            if len(recent_telemetry) < 10:
                return None
            
            # Create sequence (window_size, features)
            sequence = recent_telemetry[-10:].reshape(1, 10, -1)
            degradation = self.lstm_model.predict(sequence)[0]
            
            return {
                'predicted_degradation': float(degradation),
                'tire_age_estimate': float(degradation * 2.0),  # Simplified conversion
                'recommendation': 'pit' if degradation > 0.5 else 'continue'
            }
        except Exception as e:
            # Return None if prediction fails
            return None
    
    def detect_driver_anomalies(self, recent_telemetry):
        """Detect driver fatigue or inconsistencies"""
        if self.anomaly_detector is None:
            return None
        
        try:
            # Convert to DataFrame if needed
            if isinstance(recent_telemetry, np.ndarray):
                # Assume it's already in the right format
                recent_df = pd.DataFrame(recent_telemetry)
            else:
                recent_df = recent_telemetry
            
            # Check if model is trained
            if not self.anomaly_detector.is_fitted:
                return None
            
            # Detect fatigue
            fatigue_analysis = self.anomaly_detector.detect_driver_fatigue(recent_df)
            
            # Analyze consistency
            consistency_analysis = self.anomaly_detector.analyze_consistency(recent_df)
            
            return {
                'is_fatigue': fatigue_analysis['is_fatigue'],
                'anomaly_rate': float(fatigue_analysis['anomaly_rate']),
                'consistency_score': float(consistency_analysis['overall_consistency']),
                'recommendation': 'rest' if fatigue_analysis['is_fatigue'] else 'continue'
            }
        except Exception as e:
            # Return None if detection fails
            return None
    
    def get_race_strategy(self, current_state=None):
        """Get optimal race strategy using PPO agent"""
        if self.ppo_agent is None:
            return None
        
        # PPO expects 6D state: [lap, lap_progress, tire_age, position, degradation, pit_stops]
        # If current_state is provided but wrong shape, use default
        if current_state is None or len(current_state) != 6:
            # Default race state
            current_state = [15, 0.5, 12, 3, 0.4, 0]  # Lap 15, mid-lap, tire age 12, position 3
        
        # Ensure it's a numpy array with correct shape
        current_state = np.array(current_state, dtype=np.float32)
        if len(current_state.shape) == 1:
            current_state = current_state.reshape(1, -1)[0]  # Flatten to 1D
        
        # Get action from PPO agent
        action, _ = self.ppo_agent.select_action(current_state, deterministic=True)
        
        return {
            'pit_decision': 'pit' if action[0] > 0 else 'stay_out',
            'driving_aggression': float(action[1]),
            'strategy': 'aggressive' if action[1] > 0.5 else 'conservative'
        }
    
    def get_comprehensive_analysis(self, current_state, recent_telemetry, race_state=None):
        """Get comprehensive analysis combining all models"""
        # race_state is for PPO (6D), current_state is for IL (5D)
        analysis = {
            'driving_action': self.predict_driving_action(current_state),
            'tire_degradation': self.predict_tire_degradation(recent_telemetry),
            'driver_anomalies': self.detect_driver_anomalies(recent_telemetry),
            'race_strategy': self.get_race_strategy(race_state)
        }
        
        # Generate recommendations
        recommendations = []
        
        if analysis['tire_degradation'] and analysis['tire_degradation']['recommendation'] == 'pit':
            recommendations.append("Consider pitting for fresh tires - degradation is high")
        
        if analysis['driver_anomalies'] and analysis['driver_anomalies']['is_fatigue']:
            recommendations.append("Driver showing signs of fatigue - consider rest")
        
        if analysis['race_strategy']:
            if analysis['race_strategy']['pit_decision'] == 'pit':
                recommendations.append("Optimal strategy suggests pit stop")
            if analysis['race_strategy']['strategy'] == 'aggressive':
                recommendations.append("Optimal strategy suggests aggressive driving")
        
        analysis['recommendations'] = recommendations
        
        return analysis

def train_complete_system(data_dir='barber', output_dir='models', processed_dir='processed_data'):
    """Train the complete integrated system"""
    print("=" * 60)
    print("Training Complete Integrated Racing System")
    print("=" * 60)
    
    # Step 1: Data Preprocessing
    print("\n[1/5] Data Preprocessing...")
    preprocessor = RacingDataPreprocessor(data_dir)
    preprocessor.save_processed_data(processed_dir)
    
    # Step 2: Train Imitation Learning
    print("\n[2/5] Training Imitation Learning Model...")
    try:
        import imitation_learning
        imitation_learning.train_imitation_learning_model(processed_dir, output_dir)
    except Exception as e:
        print(f"Error training IL model: {e}")
    
    # Step 3: Train LSTM
    print("\n[3/5] Training LSTM Predictor...")
    try:
        import lstm_predictor
        lstm_predictor.train_lstm_model(processed_dir, output_dir)
    except Exception as e:
        print(f"Error training LSTM model: {e}")
    
    # Step 4: Train Anomaly Detector
    print("\n[4/5] Training Anomaly Detector...")
    try:
        import anomaly_detection
        anomaly_detection.train_anomaly_detector(processed_dir, output_dir)
    except Exception as e:
        print(f"Error training anomaly detector: {e}")
    
    # Step 5: Train PPO Agent
    print("\n[5/5] Training PPO Agent...")
    try:
        import ppo_rl
        ppo_rl.train_ppo_agent(
            il_model_path=str(Path(output_dir) / 'imitation_learning_model.pth'),
            episodes=1000,
            output_dir=output_dir
        )
    except Exception as e:
        print(f"Error training PPO agent: {e}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nAll models saved to: {output_dir}")
    print("\nYou can now use the IntegratedRacingSystem class to make predictions.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        train_complete_system()
    else:
        # Example usage
        system = IntegratedRacingSystem()
        
        # Example state and telemetry
        example_state = [50.0, 0.5, 0.8, 0.3, 0.0, 0.0, 1.0]  # [speed, lap_progress, ...]
        example_telemetry = np.random.randn(20, 6)  # Recent telemetry data
        
        analysis = system.get_comprehensive_analysis(example_state, example_telemetry)
        print("\nComprehensive Analysis:")
        print(json.dumps(analysis, indent=2, default=str))

