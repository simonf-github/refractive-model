# display_ray_mapping.py
# Visualizes ray_mapping.csv by drawing red pixels for starting rays
# and blue pixels for final distorted hitpoints, scaled to 1280x800 display.

import csv
import matplotlib.pyplot as plt
import numpy as np

INPUT_CSV = "ray_mapping.csv"
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 800

# Load data
camera_pts = []
distorted_pts = []

with open(INPUT_CSV, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        cx = int(row['camera_x'])
        cy = int(row['camera_y'])
        camera_pts.append((cx, cy))

        if row['distorted_x'] != '' and row['distorted_y'] != '':
            dx = float(row['distorted_x'])
            dy = float(row['distorted_y'])
            distorted_pts.append((cx, cy, dx, dy))

# Normalize distorted x/y range to screen
if distorted_pts:
    xs = [pt[2] for pt in distorted_pts]
    ys = [pt[3] for pt in distorted_pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    def scale_distorted(x, y):
        sx = int((x - min_x) / (max_x - min_x) * (DISPLAY_WIDTH - 1))
        sy = int((1 - (y - min_y) / (max_y - min_y)) * (DISPLAY_HEIGHT - 1))
        return sx, sy
else:
    scale_distorted = lambda x, y: (0, 0)  # fallback

# Create blank image
img = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)

# Draw red camera frustum grid
for cx, cy in camera_pts:
    sx = int(cx / 31 * (DISPLAY_WIDTH - 1))  # scale 0–31 to 0–1279
    sy = int((1 - cy / 19) * (DISPLAY_HEIGHT - 1))  # invert y
    img[sy, sx] = [255, 0, 0]  # Red

# Draw blue distorted hitpoints
for cx, cy, dx, dy in distorted_pts:
    sx, sy = scale_distorted(dx, dy)
    img[sy, sx] = [0, 0, 255]  # Blue

# Show image
plt.imshow(img)
plt.title("Red = Camera Grid, Blue = Distorted Hitpoints")
plt.axis('off')
plt.show()