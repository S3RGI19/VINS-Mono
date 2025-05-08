import re
import csv
from collections import defaultdict
import numpy as np

def read_csv(file_name):
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data

def first_timestamp(data):
    if 'timestamp' in data[0][0]:
        data = data[1:]
    timestamps = [float(row[0]) for row in data]
    timestamps = np.array(timestamps)
    timestamps = np.unique(timestamps)
    timestamps = timestamps.tolist()
    return timestamps
def load_timestamps(file):
    with open(file, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
        if 'timestamp' in data[0][0]:  # skip header if exists
            data = data[1:]
        return set(row[0] for row in data)
cam0_file_name = '/datasets/euroc/raw/V1_01_easy/mav0/cam0/data.csv'
cam0_timestamps = load_timestamps(cam0_file_name)

log_file = '/workspaces/VINS-Mono/debug_log_optimized.txt'
# log_file = '/workspaces/src/VINS-Mono/debug.txt'
feature_csv = 'feature_opt.csv'
landmark_csv = 'landmarks_opt.csv'

# Regex patterns
feature_pattern = re.compile(
    r"#timestamp: ([\d\.]+), feature_id: (\d+), x: ([\d\.-]+), y: ([\d\.-]+)"
)
landmark_pattern = re.compile(
    r"landmark: ([\d\.\-eE]+), ([\d\.\-eE]+), ([\d\.\-eE]+)"
)
optimization_pattern = re.compile(r"Estimator optimization start")

# Containers
features = []
landmarks = defaultdict(lambda: {'xyz': None, 'count': 0})

with open(log_file, 'r') as f:
    for line in f:
        f_match = feature_pattern.search(line)
        l_match = landmark_pattern.search(line)

        if f_match:
            timestamp, feature_id, x, y = f_match.groups()
            feature_id = int(feature_id)

            # Save feature record
            features.append([timestamp, feature_id, x, y, -1.0])  # -1.0 as default level

            # Update landmark if present (overwrite with latest)
            if l_match:
                try :
                    lx, ly, lz = map(float, l_match.groups())
                except:
                    print(f" l_match.groups(): {l_match.groups()}")
                landmarks[feature_id]['xyz'] = (lx, ly, lz)  # always overwrite
                landmarks[feature_id]['count'] += 1

# Sort features by timestamp
features.sort(key=lambda row: float(row[0]))

# Write feature.csv
with open(feature_csv, 'w', newline='') as f_csv:
    writer = csv.writer(f_csv)
    # writer.writerow(["timestamp", "feature_id", "x", "y", "level"])
    writer.writerows(features)

# Write landmark.csv
with open(landmark_csv, 'w', newline='') as l_csv:
    writer = csv.writer(l_csv)
    # writer.writerow(["id", "x", "y", "z", "num_features"])
    for fid, data in landmarks.items():
        if data['xyz']:
            writer.writerow([fid, *data['xyz'], data['count']])

print(f"Saved {len(features)} features to {feature_csv}")
print(f"Saved {sum(1 for d in landmarks.values() if d['xyz'])} landmarks to {landmark_csv}")

selected_features = []

with open(feature_csv, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if row[0] in cam0_timestamps:
            selected_features.append(row)

print(f"Selected {len(selected_features)} features that match cam0 timestamps")

# # Optional: Save to a new CSV
# with open('features_matched_with_cam0.csv', 'w', newline='') as f_out:
#     writer = csv.writer(f_out)
#     writer.writerows(selected_features)


# # the number of same timestamp
# all_timestamps = [row[0] for row in features]

# # Get unique ones using a set
# unique_timestamps = set(all_timestamps)

# # Count them
# print(f"Number of unique timestamps: {len(unique_timestamps)}")
# optimization_count = 0

# with open(log_file, 'r') as f:
#     for line in f:
#         if optimization_pattern.search(line):
#             optimization_count += 1
# print(f"Number of 'Estimator optimization start' entries: {optimization_count}")

# # load /workspaces/src/hyperion_dev/apps/pose3_features/features.csv

# vins_feature_record_name = './feature.csv'
# gt_file_name = '/datasets/euroc/raw/V1_01_easy/mav0/state_groundtruth_estimate0/data.csv'
# imu_file_name = '/datasets/euroc/raw/V1_01_easy/mav0/imu0/data.csv'
# cam0_file_name = '/datasets/euroc/raw/V1_01_easy/mav0/cam0/data.csv'

# timestamps_vins = first_timestamp(read_csv(vins_feature_record_name))
# timestamps_gt = first_timestamp(read_csv(gt_file_name))
# timestamps_imu = first_timestamp(read_csv(imu_file_name))
# timestamps_cam0 = first_timestamp(read_csv(cam0_file_name))
# #
# print("len(timestamps_vins):", len(timestamps_vins))
# print("len(timestamps_cam0):", len(timestamps_cam0))
# print("len(timestamps_gt):", len(timestamps_gt))
# print("timestamps_vins[0]:\t", timestamps_vins[0])
# print("timestamps_gt[0]:\t", timestamps_gt[0]/1e9)
# print("timestamps_imu[0]:\t", timestamps_imu[0]/1e9)
# print("timestamps_cam0[0]:\t", timestamps_cam0[0]/1e9)
# # print last timestamp
# print("-----------------------------")
# print("timestamps[-1]:\t\t", timestamps[-1])
# print("timestamps_vins[-1]:\t", timestamps_vins[-1])
# print("timestamps_gt[-1]:\t", timestamps_gt[-1]/1e9)
# print("timestamps_imu[-1]:\t", timestamps_imu[-1]/1e9)
# print("timestamps_cam0[-1]:\t", timestamps_cam0[-1]/1e9)