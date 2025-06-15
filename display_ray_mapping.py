import os
import numpy as np
import csv
import matplotlib.pyplot as plt
import cv2
from scipy.interpolate import griddata

# === CONFIG ===
MAPPING_CSV = "ray_mapping.csv"
TARGET_RES = (1280, 800)
FLIP_INPUT_AXES = False     # Flip camera_x and camera_y
FLIP_OUTPUT_AXES = False    # Flip distorted_x and distorted_y

# === Load ray mapping ===
cam_xs, cam_ys, dist_xs, dist_ys = [], [], [], []
with open(MAPPING_CSV, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            cx_raw = float(row['camera_x'])
            cy_raw = float(row['camera_y'])
            dx_raw = float(row['distorted_x'])
            dy_raw = float(row['distorted_y'])

            if FLIP_INPUT_AXES:
                cx_raw, cy_raw = cy_raw, cx_raw
            if FLIP_OUTPUT_AXES:
                dx_raw, dy_raw = dy_raw, dx_raw

            cx = cx_raw / 31 * (TARGET_RES[0] - 1)
            cy = cy_raw / 19 * (TARGET_RES[1] - 1)
            cam_xs.append(cx)
            cam_ys.append(cy)
            dist_xs.append(dx_raw)
            dist_ys.append(dy_raw)
        except:
            continue

# === Normalize distorted coordinates based on their actual extents ===
dx_min, dx_max = min(dist_xs), max(dist_xs)
dy_min, dy_max = min(dist_ys), max(dist_ys)

# Scale distorted coordinates to canvas
scaled_dist_xs = [(x - dx_min) / (dx_max - dx_min) * (TARGET_RES[0] - 1) for x in dist_xs]
scaled_dist_ys = [(y - dy_min) / (dy_max - dy_min) * (TARGET_RES[1] - 1) for y in dist_ys]

# === Interpolate distorted coords to full grid ===
grid_x, grid_y = np.meshgrid(np.arange(TARGET_RES[0]), np.arange(TARGET_RES[1]))
map_x = griddata((cam_xs, cam_ys), scaled_dist_xs, (grid_x, grid_y), method='linear', fill_value=-1)
map_y = griddata((cam_xs, cam_ys), scaled_dist_ys, (grid_x, grid_y), method='linear', fill_value=-1)

# === Function to generate diagonal stripe pattern ===
def generate_pattern(angle_deg):
    pattern = np.zeros((TARGET_RES[1], TARGET_RES[0], 3), dtype=np.uint8)
    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    for y in range(TARGET_RES[1]):
        for x in range(TARGET_RES[0]):
            pos = (x * cos_a + y * sin_a) % 100
            color = plt.cm.hsv(pos / 100)[:3]
            rgb = tuple(int(255 * c) for c in color)
            pattern[y, x] = rgb
    return pattern

# === Prompt user for angle ===
try:
    angle = float(input("Enter line angle in degrees (0 to 180): "))
except ValueError:
    print("Invalid input. Using angle = 0.")
    angle = 0.0

pattern = generate_pattern(angle)
visual = np.zeros_like(pattern)
valid = (map_x >= 0) & (map_y >= 0)
visual[valid] = cv2.remap(pattern, map_x.astype(np.float32), map_y.astype(np.float32), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)[valid]

# === Display output ===
plt.imshow(visual)
plt.title(f"Distortion Map – Line Angle {angle:.0f}°")
plt.axis('off')
plt.show()
