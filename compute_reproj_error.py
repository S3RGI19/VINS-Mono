import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# --- Camera intrinsics ---
fx, fy = 458.654, 457.296
cx, cy = 367.215, 248.375

# --- Extrinsics: T_BC (from body to camera) ---
R_BC = np.array([
    [ 0.01486554, -0.99988093,  0.0041403 ],
    [ 0.99955725,  0.01496721,  0.02571553],
    [-0.02577444,  0.00375619,  0.99966073]
])
t_BC = np.array([-0.02164015, -0.06467699, 0.00981073])

# --- Load data ---
features = pd.read_csv('/datasets/vins_logging/features_synced.csv')
poses = pd.read_csv('/datasets/vins_logging/poses_synced.csv')
landmarks = pd.read_csv('/datasets/vins_logging/landmarks_filtered.csv')

# --- Prepare lookup tables ---
landmark_dict = landmarks.set_index('feature_id')[['x', 'y', 'z']].to_dict(orient='index')
poses_sorted = poses.sort_values('timestamp').reset_index(drop=True)
pose_timestamps = poses_sorted['timestamp'].values

# --- Helper: find closest pose in time ---
def find_closest_pose(ts):
    idx = np.abs(pose_timestamps - ts).argmin()
    return poses_sorted.iloc[idx]

def distort_points(u, v, fx, fy, cx, cy, k1, k2, p1, p2):
    # Normalize image coordinates
    x = (u - cx) / fx
    y = (v - cy) / fy

    r2 = x**2 + y**2
    x_dist = x * (1 + k1*r2 + k2*r2**2) + 2*p1*x*y + p2*(r2 + 2*x**2)
    y_dist = y * (1 + k1*r2 + k2*r2**2) + p1*(r2 + 2*y**2) + 2*p2*x*y

    # Back to pixel coordinates
    u_dist = fx * x_dist + cx
    v_dist = fy * y_dist + cy
    return u_dist, v_dist


# --- Compute reprojection errors ---
results = []
for _, row in features.iterrows():
    ts = row['timestamp']
    fid = row['feature_id']
    fx_obs, fy_obs = row['u'], row['v']

    if fid not in landmark_dict:
        continue

    pose = find_closest_pose(ts)

    # --- Pose: world to body ---
    t_wb = np.array([pose['px'], pose['py'], pose['pz']])
    q_wb = R.from_quat([pose['qx'], pose['qy'], pose['qz'], pose['qw']])
    R_wb = q_wb.as_matrix()

    # --- World to camera transform ---
    R_wc = R_wb @ R_BC
    t_wc = R_wb @ t_BC + t_wb

    # --- Transform landmark to camera frame ---
    pw = np.array([landmark_dict[fid]['x'], landmark_dict[fid]['y'], landmark_dict[fid]['z']])
    pc = R_wc.T @ (pw - t_wc)

    if pc[2] <= 0:
        continue

    u = fx * pc[0] / pc[2] + cx
    v = fy * pc[1] / pc[2] + cy
    error = np.sqrt((u - fx_obs) ** 2 + (v - fy_obs) ** 2)

    results.append({'timestamp': ts, 'feature_id': fid, 'reprojection_error': error})

# --- Save and analyze ---
reproj_df = pd.DataFrame(results)
reproj_df.to_csv('/datasets/vins_logging/reprojection_errors.csv', index=False)
print("Saved corrected reprojection errors to /datasets/vins_logging/reprojection_errors.csv")

# --- Statistics ---
print("\n--- Reprojection Error Statistics ---")
print(reproj_df['reprojection_error'].describe(percentiles=[.25, .5, .75, .9, .99]))
print("Num total observations:", len(reproj_df))

# --- Histogram (clipped) ---
plt.figure(figsize=(8, 4))
clipped_errors = reproj_df[reproj_df['reprojection_error'] < 100]['reprojection_error']
plt.hist(clipped_errors, bins=100, color='steelblue', edgecolor='black')
plt.title('Reprojection Error Histogram (<100 px)')
plt.xlabel('Reprojection Error (px)')
plt.ylabel('Count')
plt.grid(True)
plt.tight_layout()
plt.savefig('/datasets/vins_logging/reproj_histogram_clipped.png')
plt.show()

# --- Error vs Time ---
plt.figure(figsize=(10, 4))
plt.plot(reproj_df['timestamp'], reproj_df['reprojection_error'], '.', alpha=0.3, markersize=1)
plt.title('Reprojection Error over Time')
plt.xlabel('Timestamp (ns)')
plt.ylabel('Reprojection Error (px)')
plt.grid(True)
plt.tight_layout()
plt.savefig('/datasets/vins_logging/reproj_vs_time.png')
plt.show()

# --- Error across image plane ---
features_with_error = features.merge(reproj_df, on=['timestamp', 'feature_id'])
features_clipped = features_with_error[features_with_error['reprojection_error'] < 100]

plt.figure(figsize=(8, 6))
sc = plt.scatter(
    features_clipped['u'], features_clipped['v'],
    c=features_clipped['reprojection_error'],
    cmap='viridis', s=2, alpha=0.6
)
plt.gca().invert_yaxis()
plt.colorbar(sc, label='Reprojection Error (px)')
plt.title('Reprojection Error Across Image Plane (<100 px)')
plt.xlabel('u (pixels)')
plt.ylabel('v (pixels)')
plt.grid(True)
plt.tight_layout()
plt.savefig('/datasets/vins_logging/reproj_error_image_plane.png')
plt.show()

import matplotlib.cm as cm

# --- Percentile binning ---
percentiles = [0, 20, 40, 60, 80, 100]
percentile_vals = np.percentile(features_with_error['reprojection_error'], percentiles)

for i in range(len(percentiles) - 1):
    low = percentile_vals[i]
    high = percentile_vals[i + 1]

    bin_df = features_with_error[
        (features_with_error['reprojection_error'] >= low) &
        (features_with_error['reprojection_error'] < high)
    ]

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        bin_df['u'], bin_df['v'],
        c=bin_df['reprojection_error'],
        cmap='viridis', s=2, alpha=0.6
    )
    plt.gca().invert_yaxis()
    plt.colorbar(sc, label='Reprojection Error (px)')
    plt.title(f'Reprojection Error in Image Plane: {percentiles[i]}–{percentiles[i+1]} Percentile\n[{low:.2f}–{high:.2f} px]')
    plt.xlabel('u (pixels)')
    plt.ylabel('v (pixels)')
    plt.grid(True)
    plt.tight_layout()
    fname = f'/datasets/vins_logging/reproj_error_image_plane_p{percentiles[i]}_{percentiles[i+1]}.png'
    plt.savefig(fname)
    plt.show()
    print(f"Saved plot to: {fname}")
