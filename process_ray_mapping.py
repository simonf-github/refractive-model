import csv
import os

RAW_FILE = "ray_mapping_raw.csv"
OUTPUT_FILE = "ray_mapping.csv"

# Load all rows from ray_mapping_raw.csv
input_path = os.path.join(os.getcwd(), RAW_FILE)
output_path = os.path.join(os.getcwd(), OUTPUT_FILE)

rows = []
distorted_xs = []
distorted_ys = []

with open(input_path, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            x = float(row['camera_x'])
            y = float(row['camera_y'])
            dx = float(row['distorted_x'])
            dy = float(row['distorted_y'])
            rows.append((x, y, dx, dy))
            distorted_xs.append(dx)
            distorted_ys.append(dy)
        except ValueError:
            continue  # Skip rows with invalid data

# Determine input size
input_width = max(row[0] for row in rows) + 1
input_height = max(row[1] for row in rows) + 1
print(f"Input mapping size: width = {input_width}, height = {input_height}")

# Report original distorted_x and distorted_y ranges
min_dx, max_dx = min(distorted_xs), max(distorted_xs)
min_dy, max_dy = min(distorted_ys), max(distorted_ys)
print(f"Original distorted_x range: min = {min_dx:.3f}, max = {max_dx:.3f}")
print(f"Original distorted_y range: min = {min_dy:.3f}, max = {max_dy:.3f}")

# Normalize distorted_x and distorted_y to input dimensions
scaled_rows = []
scaled_xs = []
scaled_ys = []
for x, y, dx, dy in rows:
    sx = (dx - min_dx) / (max_dx - min_dx) * input_width
    sy = (dy - min_dy) / (max_dy - min_dy) * input_height
    scaled_rows.append((x, y, sx, sy))
    scaled_xs.append(sx)
    scaled_ys.append(sy)

# Report scaled distorted_x and distorted_y ranges
scaled_min_dx, scaled_max_dx = min(scaled_xs), max(scaled_xs)
scaled_min_dy, scaled_max_dy = min(scaled_ys), max(scaled_ys)
print(f"Scaled distorted_x range: min = {scaled_min_dx:.3f}, max = {scaled_max_dx:.3f}")
print(f"Scaled distorted_y range: min = {scaled_min_dy:.3f}, max = {scaled_max_dy:.3f}")

# Write to ray_mapping.csv
with open(output_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['camera_x', 'camera_y', 'distorted_x', 'distorted_y'])
    writer.writerows(scaled_rows)

print(f"Saved scaled mapping to {OUTPUT_FILE}")
