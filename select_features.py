import csv

# 1. Load cam0 timestamps (nanoseconds) and convert to seconds
def load_cam0_timestamps(file, precision=9):
    with open(file, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
        if 'timestamp' in data[0][0]:  # skip header if exists
            data = data[1:]
        # Convert to float seconds and round
        return set([round(int(row[0]) / 1e9, precision) for row in data])

# 2. Filter features that match any cam0 timestamp (within rounding precision)
def filter_features_by_cam0(features_file, cam0_timestamps, precision=9):
    selected_features = []
    with open(features_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            feature_ts = round(float(row[0]), precision)
            if feature_ts in cam0_timestamps:
                selected_features.append(row)
    return selected_features

# === File paths ===
features_file = './feature.csv'
cam0_file = '/datasets/euroc/raw/V1_01_easy/mav0/cam0/data.csv'
output_file = 'features_matched_with_cam0.csv'

# === Run Matching ===
cam0_ts_set = load_cam0_timestamps(cam0_file)
matched_features = filter_features_by_cam0(features_file, cam0_ts_set)

# === Write output ===
with open(output_file, 'w', newline='') as f_out:
    writer = csv.writer(f_out)
    writer.writerows(matched_features)

print(f"✅ Matched {len(matched_features)} features with {len(cam0_ts_set)} cam0 timestamps")
print(f"📄 Saved to: {output_file}")

# print how many timestamps are in the output_file
timestamps = [row[0] for row in matched_features]
unique_timestamps = set(timestamps)
print(f"Number of unique timestamps: {len(unique_timestamps)}")
