# Testing Guide - Racing Strategy System

## Quick Start Testing

### 1. Run the Full Test Suite
```bash
python test_system.py
```
This will test all components individually and show comprehensive results.

### 2. Run the Presentation Demo
```bash
python hackathon_demo.py
```
This shows a complete demonstration suitable for presentations.

### 3. Run Simple Examples
```bash
python example_usage.py
```
This shows basic usage examples.

## Individual Component Testing

### Test Imitation Learning Model

```python
from imitation_learning import ImitationLearningTrainer
import json
import numpy as np

# Load model
with open('models/il_model_metadata.json', 'r') as f:
    metadata = json.load(f)

trainer = ImitationLearningTrainer(metadata['state_dim'], metadata['action_dim'])
trainer.load('models/imitation_learning_model.pth', 
             'models/imitation_learning_scalers.pkl',
             metadata['state_dim'], metadata['action_dim'])

# Test prediction
# State format: [lateral_g, longitudinal_g, sector_S1, sector_S2, sector_S3]
state = np.array([[0.8, 0.3, 0.0, 1.0, 0.0]])  # Sector 2, high lateral G
action = trainer.predict(state)

print(f"Throttle: {action[0][0]:.3f}")
print(f"Steering: {action[0][1]:.3f}")
```

### Test PPO Agent

```python
from ppo_rl import PPOAgent
import torch
import numpy as np

# Load agent
agent = PPOAgent(state_dim=6, action_dim=2)
agent.load('models/ppo_agent.pth')

# Test strategy
# State: [lap, lap_progress, tire_age, position, degradation, pit_stops]
race_state = np.array([15, 0.5, 12, 3, 0.4, 0])
action, _ = agent.select_action(race_state, deterministic=True)

print(f"Pit Decision: {'PIT' if action[0] > 0 else 'STAY OUT'}")
print(f"Aggression: {action[1]:.3f}")
```

### Test Integrated System

```python
from integrated_system import IntegratedRacingSystem
import numpy as np

# Initialize system
system = IntegratedRacingSystem()

# Test driving action
state = [0.8, 0.3, 0.0, 1.0, 0.0]  # [lateral_g, longitudinal_g, S1, S2, S3]
action = system.predict_driving_action(state)
print(f"Action: {action}")

# Test race strategy
race_state = [15, 0.5, 12, 3, 0.4, 0]  # [lap, progress, tire_age, position, degradation, pit_stops]
strategy = system.get_race_strategy(race_state)
print(f"Strategy: {strategy}")

# Comprehensive analysis
telemetry = np.random.randn(20, 6)
analysis = system.get_comprehensive_analysis(state, telemetry, race_state)
print(f"Analysis: {analysis}")
```

## Test Scenarios

### Scenario 1: High-Speed Corner
```python
state = [0.9, 0.2, 0.0, 1.0, 0.0]  # High lateral G, Sector 2
action = system.predict_driving_action(state)
# Expected: Moderate throttle, steering adjustment
```

### Scenario 2: Straight Section
```python
state = [0.1, 0.8, 1.0, 0.0, 0.0]  # Low lateral G, Sector 1
action = system.predict_driving_action(state)
# Expected: High throttle, minimal steering
```

### Scenario 3: Early Race Strategy
```python
race_state = [5, 0.5, 2, 5, 0.1, 0]  # Fresh tires, early race
strategy = system.get_race_strategy(race_state)
# Expected: Conservative strategy, possibly pit
```

### Scenario 4: Late Race Strategy
```python
race_state = [20, 0.5, 18, 2, 0.6, 0]  # Old tires, late race
strategy = system.get_race_strategy(race_state)
# Expected: Aggressive strategy, stay out or pit decision
```

## Expected Outputs

### Imitation Learning Model
- **Input**: State vector [lateral_g, longitudinal_g, sector_S1, sector_S2, sector_S3]
- **Output**: Action vector [throttle_position, steering_angle]
- **Range**: Throttle: 0-1, Steering: degrees

### PPO Agent
- **Input**: Race state [lap, lap_progress, tire_age, position, degradation, pit_stops]
- **Output**: Strategy [pit_decision, driving_aggression]
- **Range**: Pit decision: -1 to 1, Aggression: 0-1

## Troubleshooting

### Model Not Found Errors
If you see errors about missing models:
- **LSTM and Anomaly Detector**: These are optional. The core system (IL + PPO) works without them.
- **IL or PPO errors**: Make sure you've run `python train_all.py` first.

### Import Errors
Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### State Dimension Errors
- IL model expects 5D state: [lateral_g, longitudinal_g, S1, S2, S3]
- PPO agent expects 6D state: [lap, progress, tire_age, position, degradation, pit_stops]

## Validation Tests

Run these to verify everything works:

```bash
# 1. Test all components
python test_system.py

# 2. Test integrated system
python example_usage.py

# 3. Full demo
python hackathon_demo.py
```

All tests should complete without errors (warnings about optional models are OK).


