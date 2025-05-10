import pandas as pd

# Load files
features_df = pd.read_csv('/datasets/vins_logging/features.csv')  # Assumes columns: timestamp, feature_id, feature_x, feature_y, landmark_x, landmark_y, landmark_z
poses_df = pd.read_csv('/workspaces/output/vins_result_loop.csv', header=None, index_col=False, names=['timestamp', 'px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz'])
landmarks_df = pd.read_csv('/datasets/vins_logging/landmarks.csv')  # Assumes columns: timestamp, feature_id, x, y, z

# Convert timestamps to numeric
features_df['timestamp'] = pd.to_numeric(features_df['timestamp'], downcast='integer')
poses_df['timestamp'] = pd.to_numeric(poses_df['timestamp'], errors='coerce', downcast='integer')
landmarks_df['timestamp'] = pd.to_numeric(landmarks_df['timestamp'], downcast='integer')

# Determine overlapping time window
start_ts = max(features_df['timestamp'].min(), poses_df['timestamp'].min(), landmarks_df['timestamp'].min())
end_ts = min(features_df['timestamp'].max(), poses_df['timestamp'].max(), landmarks_df['timestamp'].max())

print(f"Start timestamp: {start_ts}")
print(f"End timestamp: {end_ts}")
print(f"Duration: {(end_ts - start_ts) * 1e-9:.2f} seconds")

# Filter by timestamp range
features_synced = features_df[(features_df['timestamp'] >= start_ts) & (features_df['timestamp'] <= end_ts)]
poses_synced = poses_df[(poses_df['timestamp'] >= start_ts) & (poses_df['timestamp'] <= end_ts)]
landmarks_in_range = landmarks_df[(landmarks_df['timestamp'] >= start_ts) & (landmarks_df['timestamp'] <= end_ts)]

# Keep only latest landmark per feature_id
landmarks_filtered = landmarks_in_range.sort_values('timestamp').groupby('feature_id').tail(1).sort_values('feature_id')

# Count number of observations per feature_id in features_synced
obs_counts = features_synced['feature_id'].value_counts().rename("num_observations")

# Add observation count to landmarks_filtered
landmarks_filtered = landmarks_filtered.merge(obs_counts, on='feature_id', how='left')
landmarks_filtered['num_observations'] = landmarks_filtered['num_observations'].fillna(0).astype(int)

# Drop timestamp column
landmarks_filtered = landmarks_filtered.drop(columns=['timestamp'])

# Save results
features_synced.to_csv('/datasets/vins_logging/features_synced.csv', index=False)
poses_synced.to_csv('/datasets/vins_logging/poses_synced.csv', index=False)
landmarks_filtered.to_csv('/datasets/vins_logging/landmarks_filtered.csv', index=False)

print("Filtered CSVs saved: features_synced.csv, poses_synced.csv, landmarks_filtered.csv")
