# Quick Start Guide

## 🚀 Getting Started in 3 Steps

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Train All Models

```bash
python train_all.py
```

This will:
- Process the race data from `barber/` directory
- Train Imitation Learning model (learns from best drivers)
- Train LSTM predictor (tire degradation forecasting)
- Train Anomaly Detector (driver fatigue detection)
- Train PPO agent (race strategy optimization)

**Note**: Training may take 30-60 minutes depending on your hardware.

### Step 3: Run Demo

```bash
python demo.py
```

This demonstrates all the system capabilities with example data.

## 📝 Individual Model Training

If you want to train models individually:

```python
# 1. Preprocess data
python data_preprocessing.py

# 2. Train IL model
python imitation_learning.py

# 3. Train LSTM
python lstm_predictor.py

# 4. Train Anomaly Detector
python anomaly_detection.py

# 5. Train PPO Agent
python ppo_rl.py
```

## 🎯 Using the System

```python
from integrated_system import IntegratedRacingSystem

# Load all models
system = IntegratedRacingSystem()

# Get driving recommendations
state = [speed, lap_progress, lateral_g, longitudinal_g, ...]
action = system.predict_driving_action(state)

# Predict tire degradation
degradation = system.predict_tire_degradation(recent_telemetry)

# Detect driver fatigue
anomalies = system.detect_driver_anomalies(recent_data)

# Get race strategy
strategy = system.get_race_strategy(current_state)

# Comprehensive analysis
analysis = system.get_comprehensive_analysis(state, telemetry)
```

## 📊 Expected Output Structure

After training, you should have:

```
models/
├── imitation_learning_model.pth
├── imitation_learning_scalers.pkl
├── il_model_metadata.json
├── lstm_model.pth
├── lstm_scalers.pkl
├── lstm_metadata.json
├── anomaly_detector.pkl
├── anomaly_scaler.pkl
├── anomaly_feature_contributions.json
└── ppo_agent.pth

processed_data/
├── imitation_learning_data.csv
├── lstm_sequences.npy
├── lstm_targets.npy
├── anomaly_detection_data.csv
└── metadata.json
```

## ⚠️ Troubleshooting

### "No module named 'torch'"
```bash
pip install torch
```

### "File not found" errors
Make sure the `barber/` directory contains the race data files.

### Out of memory errors
The telemetry data is large. The preprocessing script samples 10% by default. You can adjust this in `data_preprocessing.py`:
```python
telemetry_r1 = self.load_telemetry(race_num=1, sample_frac=0.05)  # Use 5% instead
```

### CUDA errors
The system will automatically use CPU if CUDA is not available. No action needed.

## 🎓 Next Steps

1. Read the full `README.md` for detailed documentation
2. Explore individual model files to understand the architecture
3. Modify hyperparameters in each training script for better performance
4. Add your own features and improvements!

## 💡 Tips

- Start with smaller sample fractions for faster iteration
- Monitor training progress - models use early stopping
- Check the `processed_data/metadata.json` for feature information
- Use `demo.py` as a template for your own applications

Good luck! 🏁



