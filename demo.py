"""
Demo script showing how to use the Integrated Racing System
Run this after training all models
"""

import numpy as np
import json
from pathlib import Path
from integrated_system import IntegratedRacingSystem

def demo_comprehensive_analysis():
    """Demonstrate comprehensive race analysis"""
    print("=" * 70)
    print("Integrated Racing System - Demo")
    print("=" * 70)
    
    # Initialize system
    print("\nLoading models...")
    try:
        system = IntegratedRacingSystem(models_dir='models', data_dir='processed_data')
    except Exception as e:
        print(f"Error loading models: {e}")
        print("\nPlease train the models first by running:")
        print("  python train_all.py")
        return
    
    print("\n" + "=" * 70)
    print("Example 1: Optimal Driving Action")
    print("=" * 70)
    
    # Example state: [speed, lap_progress, lateral_g, longitudinal_g, sector_S1, sector_S2, sector_S3]
    state = [60.0, 0.5, 0.8, 0.3, 0.0, 1.0, 0.0]  # Mid-lap, sector 2
    action = system.predict_driving_action(state)
    if action:
        print(f"State: Speed={state[0]} km/h, Lap Progress={state[1]*100:.1f}%, Sector 2")
        print(f"Recommended Action:")
        print(f"  - Throttle Position: {action['throttle_position']:.2f}")
        print(f"  - Brake Pressure: {action['brake_pressure']:.2f}")
        print(f"  - Steering Angle: {action['steering_angle']:.2f}")
    
    print("\n" + "=" * 70)
    print("Example 2: Tire Degradation Prediction")
    print("=" * 70)
    
    # Simulate recent telemetry data (window_size x features)
    recent_telemetry = np.random.randn(10, 6)  # 10 samples, 6 features
    degradation = system.predict_tire_degradation(recent_telemetry)
    if degradation:
        print(f"Tire Degradation Analysis:")
        print(f"  - Predicted Degradation: {degradation['predicted_degradation']:.3f}")
        print(f"  - Estimated Tire Age: {degradation['tire_age_estimate']:.1f} laps")
        print(f"  - Recommendation: {degradation['recommendation'].upper()}")
    
    print("\n" + "=" * 70)
    print("Example 3: Driver Anomaly Detection")
    print("=" * 70)
    
    # Simulate recent driving data
    import pandas as pd
    recent_data = pd.DataFrame(np.random.randn(50, 6), 
                              columns=['speed', 'Steering_Angle', 'throttle_position', 
                                      'total_brake_pressure', 'lateral_g', 'longitudinal_g'])
    anomalies = system.detect_driver_anomalies(recent_data)
    if anomalies:
        print(f"Driver Analysis:")
        print(f"  - Fatigue Detected: {'YES' if anomalies['is_fatigue'] else 'NO'}")
        print(f"  - Anomaly Rate: {anomalies['anomaly_rate']*100:.1f}%")
        print(f"  - Consistency Score: {anomalies['consistency_score']:.3f}")
        print(f"  - Recommendation: {anomalies['recommendation'].upper()}")
    
    print("\n" + "=" * 70)
    print("Example 4: Race Strategy")
    print("=" * 70)
    
    # Current race state: [lap, lap_progress, tire_age, position, degradation, pit_stops]
    current_state = [15, 0.3, 12, 3, 0.4, 0]  # Lap 15, tire age 12, position 3
    strategy = system.get_race_strategy(current_state)
    if strategy:
        print(f"Current Race State:")
        print(f"  - Lap: {current_state[0]}")
        print(f"  - Position: {current_state[3]}")
        print(f"  - Tire Age: {current_state[2]} laps")
        print(f"\nOptimal Strategy:")
        print(f"  - Pit Decision: {strategy['pit_decision'].upper()}")
        print(f"  - Driving Aggression: {strategy['driving_aggression']:.2f}")
        print(f"  - Strategy Type: {strategy['strategy'].upper()}")
    
    print("\n" + "=" * 70)
    print("Example 5: Comprehensive Analysis")
    print("=" * 70)
    
    # Get comprehensive analysis
    analysis = system.get_comprehensive_analysis(state, recent_telemetry)
    print("\nComprehensive Race Analysis:")
    print(json.dumps(analysis, indent=2, default=str))
    
    if analysis.get('recommendations'):
        print("\n" + "=" * 70)
        print("Recommendations:")
        print("=" * 70)
        for i, rec in enumerate(analysis['recommendations'], 1):
            print(f"{i}. {rec}")
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)

if __name__ == "__main__":
    demo_comprehensive_analysis()



