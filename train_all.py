"""
Main training script - trains all models in sequence
Run this script to train the complete system
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from integrated_system import train_complete_system

if __name__ == "__main__":
    print("=" * 70)
    print("Hack the Track - Complete System Training")
    print("=" * 70)
    print("\nThis will train all models:")
    print("  1. Data Preprocessing")
    print("  2. Imitation Learning Model")
    print("  3. LSTM Predictor")
    print("  4. Anomaly Detector")
    print("  5. PPO Agent")
    print("\nThis may take a while...")
    print("=" * 70)
    
    # Train complete system
    train_complete_system(
        data_dir='barber',
        output_dir='models',
        processed_dir='processed_data'
    )



