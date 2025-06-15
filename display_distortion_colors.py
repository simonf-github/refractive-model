import csv
import numpy as np
import cv2
import os

MAPPING_FILE = "ray_mapping.csv"

# Load mapping data
with open(MAPPING_FILE, newline='') as f:
    reader = csv.DictReader(f)
    rows = [(int(float(r['camera_x'])), int(float(r['camera_y'])), float(r['distorted_x']), float(r['distorted_y']))
            for r in reader if r['distorted_x'] and r['distorted_y']]

if not rows:
    raise ValueError("No valid rows found in mapping CSV.")

input_width = max(x for x, _, _, _ in rows) + 1
input_height = max(y for _, y, _, _ in rows) + 1

# Normalize distorted coordinates
dxs = [dx for _, _, dx, _ in rows]
dys = [dy for _, _, _, dy in rows]
min_dx, max_dx = min(dxs), max(dxs)
min_dy, max_dy = min(dys), max(dys)

# Create output images
base_img = np.zeros((input_height, input_width, 3), dtype=np.uint8)
mapped_img = np.zeros_like(base_img)
inverse_img = np.zeros_like(base_img)

for x, y, dx, dy in rows:
    # Base image (no distortion)
    brightness = int(x / input_width * 255)
    color_value = int(y / input_height * 255)
    base_img[y, x] = (color_value, 0, brightness)

    # Distorted forward mapping
    sx = (dx - min_dx) / (max_dx - min_dx) * input_width
    sy = (dy - min_dy) / (max_dy - min_dy) * input_height
    ix, iy = int(round(sx)), int(round(sy))
    if 0 <= iy < input_height and 0 <= ix < input_width:
        mapped_img[iy, ix] = (color_value, 0, brightness)

    # Inverse mapping
    inv_brightness = int(sx / input_width * 255)
    inv_color_value = int(sy / input_height * 255)
    if 0 <= y < input_height and 0 <= x < input_width:
        inverse_img[y, x] = (inv_color_value, 0, inv_brightness)

# Display results
cv2.imshow("Base (No Distortion)", base_img)
cv2.imshow("Forward Distorted Mapping", mapped_img)
cv2.imshow("Inverse Mapped Back", inverse_img)
cv2.waitKey(0)
cv2.destroyAllWindows()