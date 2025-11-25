"""
Simple Example: How to Use the Racing System
"""

import numpy as np
from integrated_system import IntegratedRacingSystem

# Initialize the system (loads all trained models)
print("Loading racing system...")
system = IntegratedRacingSystem(models_dir='models', data_dir='processed_data')

# Example 1: Get optimal driving action
print("\n" + "="*60)
print("Example 1: Optimal Driving Action")
print("="*60)

# State: [lateral_g, longitudinal_g, sector_S1, sector_S2, sector_S3]
# This represents: High lateral G (0.8), low longitudinal G (0.3), in Sector 2
state = [0.8, 0.3, 0.0, 1.0, 0.0]

action = system.predict_driving_action(state)
if action:
    print(f"State: Lateral G={state[0]}, Longitudinal G={state[1]}, Sector 2")
    print(f"Recommended Action:")
    print(f"  - Throttle Position: {action['throttle_position']:.3f} (0-1 scale)")
    print(f"  - Steering Angle: {action['steering_angle']:.2f} degrees")

# Example 2: Get race strategy
print("\n" + "="*60)
print("Example 2: Race Strategy")
print("="*60)

# Current race state: [lap, lap_progress, tire_age, position, degradation, pit_stops]
race_state = [15, 0.5, 12, 3, 0.4, 0]  # Lap 15, mid-lap, tire age 12, position 3

strategy = system.get_race_strategy(race_state)
if strategy:
    print(f"Current Race State:")
    print(f"  - Lap: {race_state[0]}")
    print(f"  - Position: {race_state[3]}")
    print(f"  - Tire Age: {race_state[2]} laps")
    print(f"\nOptimal Strategy:")
    print(f"  - Pit Decision: {strategy['pit_decision'].upper()}")
    print(f"  - Aggression: {strategy['driving_aggression']:.2f}")
    print(f"  - Strategy Type: {strategy['strategy'].upper()}")

# Example 3: Comprehensive analysis
print("\n" + "="*60)
print("Example 3: Comprehensive Analysis")
print("="*60)

# Simulate recent telemetry data (20 samples, 6 features)
recent_telemetry = np.random.randn(20, 6)

# Race state for PPO: [lap, lap_progress, tire_age, position, degradation, pit_stops]
race_state = [15, 0.5, 12, 3, 0.4, 0]

analysis = system.get_comprehensive_analysis(state, recent_telemetry, race_state)

print("\nComplete Analysis:")
print(f"  - Driving Action: {'Available' if analysis.get('driving_action') else 'N/A'}")
print(f"  - Race Strategy: {'Available' if analysis.get('race_strategy') else 'N/A'}")
print(f"  - Tire Degradation: {'Available' if analysis.get('tire_degradation') else 'N/A'}")
print(f"  - Driver Anomalies: {'Available' if analysis.get('driver_anomalies') else 'N/A'}")

if analysis.get('recommendations'):
    print(f"\nRecommendations ({len(analysis['recommendations'])}):")
    for i, rec in enumerate(analysis['recommendations'], 1):
        print(f"  {i}. {rec}")

print("\n" + "="*60)
print("System Ready for Real-Time Use!")
print("="*60)

