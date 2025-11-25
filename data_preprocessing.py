"""
Data Preprocessing and Feature Engineering for GR Cup Racing Data
This script processes telemetry data, lap times, and race results to create
features for ML models (IL, LSTM, Isolation Forest, PPO)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class RacingDataPreprocessor:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.track_length = 144672  # inches (from track info)
        
    def load_results(self, race_num=1):
        """Load race results to identify best drivers"""
        results_file = self.data_dir / f"03_Provisional Results_Race {race_num}_Anonymized.CSV"
        df = pd.read_csv(results_file, sep=';')
        # Parse lap times
        df['FL_TIME_SECONDS'] = df['FL_TIME'].apply(self._parse_lap_time)
        df = df.sort_values('FL_TIME_SECONDS')
        return df
    
    def _parse_lap_time(self, time_str):
        """Convert lap time string (e.g., '1:37.428') to seconds"""
        if pd.isna(time_str) or time_str == '':
            return np.nan
        parts = str(time_str).split(':')
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    
    def load_telemetry(self, race_num=1, sample_frac=0.1):
        """Load and pivot telemetry data (sampling for memory efficiency)"""
        telemetry_file = self.data_dir / f"R{race_num}_barber_telemetry_data.csv"
        print(f"Loading telemetry from {telemetry_file}...")
        
        # Read in chunks and sample
        chunk_size = 100000
        chunks = []
        for chunk in pd.read_csv(telemetry_file, chunksize=chunk_size):
            chunk = chunk.sample(frac=sample_frac) if sample_frac < 1.0 else chunk
            chunks.append(chunk)
        
        df = pd.concat(chunks, ignore_index=True)
        
        # Pivot to wide format
        print("Pivoting telemetry data...")
        # Create a mapping of vehicle_id to vehicle_number before pivoting
        vehicle_mapping = df[['vehicle_id', 'vehicle_number']].drop_duplicates()
        
        # Pivot using vehicle_id, lap, timestamp as index
        telemetry_pivot = df.pivot_table(
            index=['vehicle_id', 'lap', 'timestamp'],
            columns='telemetry_name',
            values='telemetry_value',
            aggfunc='first'
        ).reset_index()
        
        # Merge vehicle_number back after pivot
        telemetry_pivot = telemetry_pivot.merge(
            vehicle_mapping,
            on='vehicle_id',
            how='left'
        )
        
        # Ensure vehicle_number is numeric (int) for proper matching
        if 'vehicle_number' in telemetry_pivot.columns:
            telemetry_pivot['vehicle_number'] = pd.to_numeric(telemetry_pivot['vehicle_number'], errors='coerce').astype('Int64')
        
        return telemetry_pivot
    
    def load_lap_data(self, race_num=1):
        """Load lap time and sector data"""
        lap_time_file = self.data_dir / f"R{race_num}_barber_lap_time.csv"
        lap_start_file = self.data_dir / f"R{race_num}_barber_lap_start.csv"
        
        lap_times = pd.read_csv(lap_time_file)
        lap_starts = pd.read_csv(lap_start_file)
        
        return lap_times, lap_starts
    
    def identify_expert_drivers(self, results_df, top_n=5):
        """Identify top N drivers based on fastest lap times"""
        expert_drivers = results_df.head(top_n)['NUMBER'].tolist()
        # Convert to int to ensure proper matching
        expert_drivers = [int(d) for d in expert_drivers]
        return expert_drivers
    
    def get_vehicle_id_mapping(self, race_num=1):
        """Get mapping from car number (NUMBER) to vehicle_id using analysis file"""
        analysis_file = self.data_dir / f"23_AnalysisEnduranceWithSections_Race {race_num}_Anonymized.CSV"
        if not analysis_file.exists():
            return {}
        
        try:
            analysis_df = pd.read_csv(analysis_file, sep=';')
            # Get unique car numbers
            car_numbers = analysis_df['NUMBER'].unique()
            
            # Try to find vehicle_id mapping from lap time files
            lap_time_file = self.data_dir / f"R{race_num}_barber_lap_time.csv"
            if lap_time_file.exists():
                lap_times = pd.read_csv(lap_time_file)
                # The vehicle_id should be consistent, but we need to match by some other method
                # For now, return empty dict - we'll use a different approach
                pass
            
            return {}
        except Exception as e:
            print(f"Warning: Could not load analysis file for mapping: {e}")
            return {}
    
    def create_driving_features(self, telemetry_df):
        """Create features for driving behavior analysis"""
        df = telemetry_df.copy()
        
        # Ensure numeric columns
        numeric_cols = ['accx_can', 'accy_can', 'aps', 'pbrake_r', 'pbrake_f', 
                       'Steering_Angle', 'speed', 'nmot', 'gear']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calculate derived features
        if 'accx_can' in df.columns and 'accy_can' in df.columns:
            df['total_acceleration'] = np.sqrt(df['accx_can']**2 + df['accy_can']**2)
            df['lateral_g'] = df['accy_can'].abs()
            df['longitudinal_g'] = df['accx_can'].abs()
        
        if 'aps' in df.columns:
            df['throttle_position'] = df['aps'] / 100.0  # Normalize to 0-1
        
        if 'pbrake_f' in df.columns and 'pbrake_r' in df.columns:
            df['total_brake_pressure'] = df['pbrake_f'] + df['pbrake_r']
            df['brake_balance'] = df['pbrake_f'] / (df['pbrake_f'] + df['pbrake_r'] + 1e-6)
        
        if 'Laptrigger_lapdist_dls' in df.columns:
            df['lap_distance'] = pd.to_numeric(df['Laptrigger_lapdist_dls'], errors='coerce')
            df['lap_progress'] = df['lap_distance'] / self.track_length
            df['sector'] = pd.cut(df['lap_progress'], bins=[0, 0.28, 0.71, 1.0], 
                                 labels=['S1', 'S2', 'S3'])
        
        # Calculate rolling statistics for consistency
        if 'speed' in df.columns:
            df = df.sort_values(['vehicle_id', 'lap', 'timestamp'])
            df['speed_rolling_mean'] = df.groupby(['vehicle_id', 'lap'])['speed'].transform(
                lambda x: x.rolling(window=10, min_periods=1).mean()
            )
            df['speed_rolling_std'] = df.groupby(['vehicle_id', 'lap'])['speed'].transform(
                lambda x: x.rolling(window=10, min_periods=1).std()
            )
        
        if 'Steering_Angle' in df.columns:
            df['steering_consistency'] = df.groupby(['vehicle_id', 'lap'])['Steering_Angle'].transform(
                lambda x: x.rolling(window=10, min_periods=1).std()
            )
        
        return df
    
    def create_tire_degradation_features(self, lap_times_df):
        """Create features for tire degradation prediction"""
        df = lap_times_df.copy()
        
        # Parse lap times
        if 'LAP_TIME' in df.columns:
            df['lap_time_seconds'] = df['LAP_TIME'].apply(self._parse_lap_time)
        
        # Calculate tire degradation indicators
        df = df.sort_values(['NUMBER', 'LAP_NUMBER'])
        df['lap_time_delta'] = df.groupby('NUMBER')['lap_time_seconds'].diff()
        df['cumulative_laps'] = df.groupby('NUMBER').cumcount() + 1
        
        # Rolling average lap times (tire degradation proxy)
        df['rolling_avg_lap_time'] = df.groupby('NUMBER')['lap_time_seconds'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )
        
        # Sector degradation
        for sector in ['S1', 'S2', 'S3']:
            if f'{sector}_SECONDS' in df.columns:
                df[f'{sector}_degradation'] = df.groupby('NUMBER')[f'{sector}_SECONDS'].transform(
                    lambda x: x - x.rolling(window=3, min_periods=1).min()
                )
        
        return df
    
    def prepare_imitation_learning_data(self, telemetry_df, expert_drivers, race_num=1):
        """Prepare data for imitation learning - expert demonstrations"""
        # Since vehicle_number doesn't match car numbers, we'll use a different approach:
        # Identify expert drivers by finding vehicles with fastest lap times in telemetry
        
        print(f"Identifying expert vehicles from telemetry data (car numbers: {expert_drivers})...")
        
        # Load analysis file to get car number to performance mapping
        analysis_file = self.data_dir / f"23_AnalysisEnduranceWithSections_Race {race_num}_Anonymized.CSV"
        expert_vehicle_ids = set()
        
        if analysis_file.exists():
            try:
                analysis_df = pd.read_csv(analysis_file, sep=';')
                # Get best lap times for each car number
                analysis_df['LAP_TIME_SECONDS'] = analysis_df[' LAP_TIME'].apply(self._parse_lap_time)
                
                # Find fastest lap for each car number in expert list
                for car_num in expert_drivers:
                    car_data = analysis_df[analysis_df['NUMBER'] == car_num]
                    if len(car_data) > 0:
                        # Get the fastest lap time for this car
                        fastest_lap = car_data['LAP_TIME_SECONDS'].min()
                        print(f"  Car {car_num}: fastest lap = {fastest_lap:.3f}s")
                
                # Since we can't directly map car numbers to vehicle_id, 
                # we'll use all vehicles and filter by performance later
                # OR we can identify fastest vehicles from telemetry directly
            except Exception as e:
                print(f"Warning: Could not use analysis file: {e}")
        
        # Alternative approach: Use all telemetry data and identify fastest vehicles
        # by analyzing lap times from the data itself
        if len(expert_vehicle_ids) == 0:
            print("Using alternative approach: identifying fastest vehicles from telemetry...")
            # Group by vehicle_id and calculate average/max speed or other performance metrics
            if 'speed' in telemetry_df.columns and 'vehicle_id' in telemetry_df.columns:
                vehicle_performance = telemetry_df.groupby('vehicle_id').agg({
                    'speed': ['mean', 'max'],
                }).reset_index()
                vehicle_performance.columns = ['vehicle_id', 'avg_speed', 'max_speed']
                vehicle_performance = vehicle_performance.sort_values('avg_speed', ascending=False)
                
                # Take top N vehicles as "experts"
                top_vehicles = vehicle_performance.head(len(expert_drivers))['vehicle_id'].tolist()
                expert_vehicle_ids = set(top_vehicles)
                print(f"  Identified expert vehicles: {top_vehicles}")
        
        # Filter telemetry for expert vehicles
        if len(expert_vehicle_ids) > 0:
            expert_data = telemetry_df[telemetry_df['vehicle_id'].isin(expert_vehicle_ids)].copy()
            print(f"  Filtered to {len(expert_data)} samples from {len(expert_vehicle_ids)} expert vehicles")
        else:
            # Fallback: use all data (not ideal but better than nothing)
            print("Warning: Could not identify expert vehicles, using all data")
            expert_data = telemetry_df.copy()
        
        if len(expert_data) == 0:
            raise ValueError("No expert data found! Check telemetry data and vehicle identification.")
        
        # Create state-action pairs
        # State: [speed, lap_progress, lateral_g, longitudinal_g, sector_encoded]
        # Action: [throttle_position, brake_pressure, steering_angle]
        
        state_features = ['speed', 'lap_progress', 'lateral_g', 'longitudinal_g']
        action_features = ['throttle_position', 'total_brake_pressure', 'Steering_Angle']
        
        # Encode sector as one-hot
        if 'sector' in expert_data.columns:
            sector_dummies = pd.get_dummies(expert_data['sector'], prefix='sector')
            expert_data = pd.concat([expert_data, sector_dummies], axis=1)
            state_features.extend(sector_dummies.columns.tolist())
        
        # Select and clean features
        all_features = state_features + action_features
        available_features = [f for f in all_features if f in expert_data.columns]
        
        # Check which features have good coverage
        feature_coverage = expert_data[available_features].notna().sum()
        print(f"\nFeature coverage:")
        for feat in available_features:
            coverage_pct = (feature_coverage[feat] / len(expert_data)) * 100
            print(f"  {feat}: {feature_coverage[feat]:,} ({coverage_pct:.1f}%)")
        
        # Use only features with reasonable coverage (>10% non-null)
        good_features = [f for f in available_features if feature_coverage[f] > len(expert_data) * 0.1]
        
        if len(good_features) == 0:
            raise ValueError("No features with sufficient coverage found!")
        
        print(f"\nUsing {len(good_features)} features with >10% coverage")
        
        # Fill NaN values for features we'll use (forward fill then backward fill)
        expert_data_clean = expert_data[['vehicle_id', 'lap', 'timestamp'] + good_features].copy()
        
        # Fill NaN values with forward fill and backward fill
        for col in good_features:
            if expert_data_clean[col].isna().any():
                # Forward fill within each vehicle/lap group
                expert_data_clean[col] = expert_data_clean.groupby(['vehicle_id', 'lap'])[col].ffill()
                # Backward fill
                expert_data_clean[col] = expert_data_clean.groupby(['vehicle_id', 'lap'])[col].bfill()
                # Fill remaining with median
                if expert_data_clean[col].isna().any():
                    median_val = expert_data_clean[col].median()
                    expert_data_clean[col] = expert_data_clean[col].fillna(median_val if not pd.isna(median_val) else 0)
        
        # Final check - drop rows where critical features are still NaN
        critical_features = ['speed', 'lap_progress']  # These are essential
        critical_available = [f for f in critical_features if f in good_features]
        if critical_available:
            expert_data_clean = expert_data_clean.dropna(subset=critical_available)
        
        print(f"Final data shape after cleaning: {expert_data_clean.shape}")
        
        # Separate state and action from good features
        state_cols = [f for f in state_features if f in good_features]
        action_cols = [f for f in action_features if f in good_features]
        
        return expert_data_clean, state_cols, action_cols
    
    def prepare_lstm_data(self, telemetry_df, window_size=10):
        """Prepare time-series data for LSTM (tire degradation, lap time prediction)"""
        df = telemetry_df.copy()
        df = df.sort_values(['vehicle_id', 'lap', 'timestamp'])
        
        # Features for prediction
        feature_cols = ['speed', 'total_acceleration', 'throttle_position', 
                       'total_brake_pressure', 'lateral_g', 'longitudinal_g']
        available_cols = [c for c in feature_cols if c in df.columns]
        
        if len(available_cols) == 0:
            print("Warning: No LSTM features available, using fallback features")
            available_cols = [c for c in df.columns if c not in ['vehicle_id', 'lap', 'timestamp', 'vehicle_number', 'sector']]
            available_cols = available_cols[:6]  # Take first 6 numeric columns
        
        print(f"LSTM using features: {available_cols}")
        
        # Fill NaN values for these columns
        for col in available_cols:
            if df[col].isna().any():
                df[col] = df.groupby(['vehicle_id', 'lap'])[col].ffill().bfill()
                df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 0)
        
        # Group by vehicle and lap
        sequences = []
        targets = []
        
        for (vehicle_id, lap), group in df.groupby(['vehicle_id', 'lap']):
            if len(group) < window_size:
                continue
            
            # Create sequences
            group_features = group[available_cols].values
            
            # Check for NaN
            if np.isnan(group_features).any():
                continue
            
            for i in range(len(group_features) - window_size):
                seq = group_features[i:i+window_size]
                target = group_features[i+window_size, 0] if len(available_cols) > 0 else group_features[i+window_size, -1]
                
                # Skip if sequence or target contains NaN
                if not (np.isnan(seq).any() or np.isnan(target)):
                    sequences.append(seq)
                    targets.append(target)
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        print(f"LSTM sequences shape: {sequences.shape}, targets shape: {targets.shape}")
        print(f"NaN check - sequences: {np.isnan(sequences).any()}, targets: {np.isnan(targets).any()}")
        
        return sequences, targets
    
    def prepare_anomaly_detection_data(self, telemetry_df, driver_id=None):
        """Prepare data for anomaly detection (Isolation Forest)"""
        df = telemetry_df.copy()
        
        if driver_id:
            df = df[df['vehicle_number'] == driver_id]
        
        # Features for anomaly detection
        feature_cols = ['speed', 'Steering_Angle', 'throttle_position', 
                       'total_brake_pressure', 'lateral_g', 'longitudinal_g',
                       'steering_consistency']
        
        available_cols = [c for c in feature_cols if c in df.columns]
        
        if len(available_cols) == 0:
            # Fallback to any numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            available_cols = [c for c in numeric_cols if c not in ['vehicle_id', 'lap', 'vehicle_number']][:7]
        
        # Ensure vehicle_id and lap are present for groupby
        required_cols = ['vehicle_id', 'lap']
        missing_required = [col for col in required_cols if col not in df.columns]
        if missing_required:
            raise KeyError(f"Missing required columns for anomaly detection: {missing_required}")
        df_features = df[required_cols + available_cols].copy()
        
        # Fill NaN values instead of dropping
        for col in available_cols:
            if df_features[col].isna().any():
                df_features[col] = df_features.groupby(['vehicle_id', 'lap'])[col].ffill().bfill()
                df_features[col] = df_features[col].fillna(df_features[col].median() if not df_features[col].isna().all() else 0)
        
        # Only drop rows where all features are NaN
        df_features = df_features.dropna(how='all')
        
        print(f"Anomaly detection data: {len(df_features)} samples with {len(available_cols)} features")
        
        return df_features
    
    def save_processed_data(self, output_dir='processed_data'):
        """Process and save all preprocessed data"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("Loading race results...")
        results_r1 = self.load_results(race_num=1)
        results_r2 = self.load_results(race_num=2)
        
        print("Identifying expert drivers...")
        expert_drivers = self.identify_expert_drivers(results_r1, top_n=5)
        print(f"Expert drivers: {expert_drivers}")
        
        print("Loading telemetry data (this may take a while)...")
        telemetry_r1 = self.load_telemetry(race_num=1, sample_frac=0.1)  # Sample 10% for speed
        telemetry_r1 = self.create_driving_features(telemetry_r1)
        
        print("Preparing imitation learning data...")
        il_data, state_cols, action_cols = self.prepare_imitation_learning_data(
            telemetry_r1, expert_drivers, race_num=1
        )
        il_data.to_csv(output_path / 'imitation_learning_data.csv', index=False)
        
        print("Preparing LSTM data...")
        lstm_sequences, lstm_targets = self.prepare_lstm_data(telemetry_r1)
        np.save(output_path / 'lstm_sequences.npy', lstm_sequences)
        np.save(output_path / 'lstm_targets.npy', lstm_targets)
        
        print("Preparing anomaly detection data...")
        anomaly_data = self.prepare_anomaly_detection_data(telemetry_r1)
        anomaly_data.to_csv(output_path / 'anomaly_detection_data.csv', index=False)
        
        # Save metadata
        metadata = {
            'expert_drivers': expert_drivers,
            'state_features': state_cols,
            'action_features': action_cols,
            'track_length': self.track_length
        }
        import json
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"All processed data saved to {output_path}")
        return output_path

if __name__ == "__main__":
    data_dir = Path(__file__).parent / "barber"
    preprocessor = RacingDataPreprocessor(data_dir)
    preprocessor.save_processed_data()

