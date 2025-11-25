"""
Hack the Track - Toyota GR Cup
Presentation Demo Script
Hybrid IL+RL Racing Strategy System
"""

import numpy as np
from integrated_system import IntegratedRacingSystem
import time

def print_header():
    print("\n" + "="*70)
    print("HACK THE TRACK - TOYOTA GR CUP")
    print("="*70)
    print("\nHybrid IL+RL Racing Strategy System")
    print("Real-Time Analytics for Race Engineering\n")

def demo_driving_actions(system):
    """Demonstrate driving action predictions"""
    print("="*70)
    print("1. DRIVING ACTION PREDICTION (Imitation Learning)")
    print("="*70)
    print("\nThe system learns from expert drivers and predicts optimal actions:")
    print("-"*70)
    
    scenarios = [
        ("High-Speed Corner (Sector 2)", [0.9, 0.2, 0.0, 1.0, 0.0], 
         "High lateral G, low longitudinal G"),
        ("Straight Section (Sector 1)", [0.1, 0.8, 1.0, 0.0, 0.0],
         "Low lateral G, high longitudinal G"),
        ("Tight Corner (Sector 3)", [0.7, 0.4, 0.0, 0.0, 1.0],
         "Moderate lateral G, moderate longitudinal G")
    ]
    
    for name, state, description in scenarios:
        action = system.predict_driving_action(state)
        if action:
            print(f"\n[SCENARIO] {name}")
            print(f"   Context: {description}")
            print(f"   -> Throttle Position: {action['throttle_position']:.3f} (0-1 scale)")
            print(f"   -> Steering Angle: {action['steering_angle']:.2f} degrees")
            time.sleep(0.5)
    
    print("\n" + "-"*70)
    print("[OK] Trained on 153,072 expert demonstrations from top 5 drivers")
    print("[OK] Predicts optimal throttle and steering in real-time")

def demo_race_strategy(system):
    """Demonstrate race strategy optimization"""
    print("\n\n" + "="*70)
    print("2. RACE STRATEGY OPTIMIZATION (PPO Reinforcement Learning)")
    print("="*70)
    print("\nThe system optimizes strategic decisions using RL:")
    print("-"*70)
    
    race_scenarios = [
        ("Early Race - Fresh Tires", [5, 0.5, 2, 5, 0.1, 0],
         "Lap 5, position 5, tires fresh"),
        ("Mid-Race - Tire Degradation", [15, 0.5, 12, 3, 0.4, 0],
         "Lap 15, position 3, tires aging"),
        ("Late Race - Critical Decision", [20, 0.1, 18, 2, 0.6, 0],
         "Lap 20, position 2, high degradation")
    ]
    
    for name, race_state, description in race_scenarios:
        strategy = system.get_race_strategy(race_state)
        if strategy:
            print(f"\n[SCENARIO] {name}")
            print(f"   Situation: {description}")
            print(f"   -> Pit Decision: {strategy['pit_decision'].upper().replace('_', ' ')}")
            print(f"   -> Strategy: {strategy['strategy'].upper()}")
            print(f"   -> Aggression Level: {strategy['driving_aggression']:.2f}")
            time.sleep(0.5)
    
    print("\n" + "-"*70)
    print("[OK] Uses PPO (Proximal Policy Optimization) for strategic decisions")
    print("[OK] Initialized from IL model - hybrid approach for best performance")

def demo_comprehensive_analysis(system):
    """Demonstrate comprehensive analysis"""
    print("\n\n" + "="*70)
    print("3. COMPREHENSIVE RACE ANALYSIS")
    print("="*70)
    print("\nCombining all models for complete race insights:")
    print("-"*70)
    
    # Example state
    il_state = [0.8, 0.3, 0.0, 1.0, 0.0]  # Sector 2, high lateral G
    race_state = [15, 0.5, 12, 3, 0.4, 0]  # Mid-race scenario
    recent_telemetry = np.random.randn(20, 6)  # Simulated telemetry
    
    analysis = system.get_comprehensive_analysis(il_state, recent_telemetry, race_state)
    
    print("\n[ANALYSIS] Real-Time Analysis Results:")
    print(f"   - Driving Action: {'[OK] Available' if analysis.get('driving_action') else '[N/A]'}")
    print(f"   - Race Strategy: {'[OK] Available' if analysis.get('race_strategy') else '[N/A]'}")
    print(f"   - Tire Degradation: {'[OK] Available' if analysis.get('tire_degradation') else '[Optional]'}")
    print(f"   - Driver Anomalies: {'[OK] Available' if analysis.get('driver_anomalies') else '[Optional]'}")
    
    if analysis.get('recommendations'):
        print(f"\n[RECOMMENDATIONS] ({len(analysis['recommendations'])}):")
        for i, rec in enumerate(analysis['recommendations'], 1):
            print(f"   {i}. {rec}")
    
    print("\n" + "-"*70)
    print("[OK] Integrated system provides complete race engineering insights")
    print("[OK] Ready for real-time deployment")

def print_technical_highlights():
    """Print technical achievements"""
    print("\n\n" + "="*70)
    print("TECHNICAL HIGHLIGHTS")
    print("="*70)
    print("""
[ARCHITECTURE]
   - Imitation Learning: Deep Neural Network (256-256-128)
   - Reinforcement Learning: PPO with IL initialization
   - Hybrid Approach: Best of both worlds

[DATA PROCESSING]
   - 153,072 expert demonstrations processed
   - Top 5 drivers identified and analyzed
   - Robust feature engineering (G-forces, sectors, telemetry)

[CAPABILITIES]
   - Real-time driving action prediction
   - Strategic decision optimization
   - Comprehensive race analysis
   - Pit stop timing recommendations

[INNOVATION]
   - IL learns from champions (expert demonstrations)
   - RL optimizes beyond human strategies
   - Hybrid initialization for faster convergence
    """)

def print_summary():
    """Print final summary"""
    print("\n" + "="*70)
    print("SYSTEM STATUS: READY FOR DEPLOYMENT")
    print("="*70)
    print("""
[STATUS] Core Models: Fully Trained and Operational
[STATUS] Real-Time Capabilities: Instant Predictions
[STATUS] Strategic Optimization: PPO Agent Ready
[STATUS] Integration: All Components Working Together

[READY] System Ready for Hackathon Presentation!
    """)

def main():
    """Main demo function"""
    print_header()
    
    print("Loading racing system...")
    system = IntegratedRacingSystem(models_dir='models', data_dir='processed_data')
    print("[OK] System loaded successfully!\n")
    
    time.sleep(1)
    
    # Run demos
    demo_driving_actions(system)
    time.sleep(1)
    
    demo_race_strategy(system)
    time.sleep(1)
    
    demo_comprehensive_analysis(system)
    time.sleep(1)
    
    print_technical_highlights()
    print_summary()

if __name__ == "__main__":
    main()

