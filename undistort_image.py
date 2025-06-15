import os
import cv2
import numpy as np
import csv

# Config flags
DEBUG_PRINT = True
DOUBLE_IMAGE = True

# Constants
IMAGE_FOLDER = "image"
MAPPING_FILE = "ray_mapping.csv"

# Load image
image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
if not image_files:
    raise FileNotFoundError("No image file found in 'image' folder.")
image_path = os.path.join(IMAGE_FOLDER, image_files[0])
img = cv2.imread(image_path)
if img is None:
    raise ValueError("Failed to load image.")

# Optionally double the resolution by duplicating pixels
if DOUBLE_IMAGE:
    img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)

height, width = img.shape[:2]

# Load mapping (mapping is unaffected by DOUBLE_IMAGE)
map_x = np.zeros((height, width), dtype=np.float32)
map_y = np.zeros((height, width), dtype=np.float32)
all_dx = []
all_dy = []

with open(MAPPING_FILE, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            cx = int(float(row['camera_x']))
            cy = int(float(row['camera_y']))
            dx = float(row['distorted_x'])
            dy = float(row['distorted_y'])

            if 0 <= cx < width and 0 <= cy < height:
                map_x[cy, cx] = dx
                map_y[cy, cx] = dy
                all_dx.append(dx)
                all_dy.append(dy)
        except ValueError:
            continue

# Debug print: show image size and range of mapping
if DEBUG_PRINT:
    print(f"Image size: width={width}, height={height}")
    if all_dx and all_dy:
        print(f"Distorted_x range: min={min(all_dx):.2f}, max={max(all_dx):.2f}")
        print(f"Distorted_y range: min={min(all_dy):.2f}, max={max(all_dy):.2f}")
    else:
        print("Warning: No valid distorted_x or distorted_y values loaded.")

# Fill in missing map values by interpolation
mask = (map_x == 0) & (map_y == 0)
if np.any(mask) and DEBUG_PRINT:
    print("Warning: some remap values are zero. Consider interpolating missing data.")

# Apply remapping
undistorted = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)

# Display
cv2.imshow("Original", img)
cv2.imshow("Undistorted", undistorted)
cv2.waitKey(0)
cv2.destroyAllWindows()