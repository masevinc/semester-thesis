import cv2
import numpy as np
import os

# === USER INPUTS ===
input_image_path = 'thesis/binary_map_test/deneme_pressuredata.png'          # Your input image
output_dir = 'thesis/binary_map_test'                       # Directory to save output images

# === SETUP OUTPUT FOLDER ===
os.makedirs(output_dir, exist_ok=True)

def save_step(img, step_name):
    path = os.path.join(output_dir, f"{step_name}.png")
    cv2.imwrite(path, img)

# === PROCESSING STEPS ===

# Load image
image = cv2.imread(input_image_path)
save_step(image, "step_0_original")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
save_step(gray, "step_1_grayscale")

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
save_step(blurred, "step_2_blurred")

# Apply binary threshold (invert if wedge is dark)
_, binary_mask = cv2.threshold(blurred, 250, 55, cv2.THRESH_BINARY_INV)
save_step(binary_mask, "step_3_binary_mask")

# Find contours
contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours by area and take the largest (likely the wedge)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
wedge_contour = contours[0]

# Draw largest contour
contour_img = image.copy()
cv2.drawContours(contour_img, [wedge_contour], -1, (0, 255, 0), 2)
save_step(contour_img, "step_4_contour")

# Approximate the contour to polygon
epsilon = 0.01 * cv2.arcLength(wedge_contour, True)
approx = cv2.approxPolyDP(wedge_contour, epsilon, True)

# Draw approximated points
approx_img = image.copy()
for pt in approx:
    cv2.circle(approx_img, tuple(pt[0]), 5, (0, 0, 255), -1)
save_step(approx_img, "step_5_approx_points")

# Output coordinates to console
print("Wedge Geometry Points:")
for point in approx.reshape(-1, 2):
    print(f"({point[0]}, {point[1]})")

# === EXPORT POINTS TO TEXT FILE ===
points_output_path = os.path.join(output_dir, "wedge_points.txt")

# Reshape and save
points = approx.reshape(-1, 2)
with open(points_output_path, 'w') as f:
    f.write("x, y\n")  # header (optional)
    for (x, y) in points:
        f.write(f"{x}, {y}\n")

print(f"\nGeometry points saved to: {points_output_path}")
