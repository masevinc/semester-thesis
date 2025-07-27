import cv2
import numpy as np
import os

'''
Basic version
- Uses png image instead of the data directly
- saves the figures for several steps, good for debugging
- saves the points in a csv file
'''

# ---------------------------
# Configuration
# ---------------------------

# Path to the CFD image
input_image_path = './pressure_data_from_rim/density_plot.png'  # Replace with your actual file path

# Output directory
output_dir = './pressure_data_from_rim/dummy_results'
os.makedirs(output_dir, exist_ok=True)

# Known physical Y distance from bottom-left to top-left of the domain
physical_domain_height = 1

# ---------------------------
# Helper Functions
# ---------------------------

def save_step(img, step_name):
    """Save intermediate image to output directory."""
    path = os.path.join(output_dir, f"{step_name}.png")
    cv2.imwrite(path, img)

# def find_extreme_points(pts):
#     """Find bottom-left and top-left points among detected points."""
#     sorted_by_x = sorted(pts, key=lambda p: (p[0], -p[1]))  # Left-most first
#     left_points = [p for p in sorted_by_x if abs(p[0] - sorted_by_x[0][0]) < 5]
#     bottom_left = min(left_points, key=lambda p: p[1])  # Smallest y
#     top_left = max(left_points, key=lambda p: p[1])     # Largest y
#     return bottom_left, top_left

# ---------------------------
# Image Processing
# ---------------------------

image = cv2.imread(input_image_path)
assert image is not None, f"Failed to load image at path: {input_image_path}"
save_step(image, "step_0_original")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
save_step(gray, "step_1_grayscale")

# blurred = cv2.GaussianBlur(gray, (5, 5), 0) # Not enough for pressure case, tried other things below
# save_step(blurred, "step_2_blurred")

# blurred_v2 = cv2.GaussianBlur(gray, (9, 9), 0) # Increase kernal size
# save_step(blurred_v2, "step_2_blurred_v2")

# blurred_v3 = cv2.GaussianBlur(gray, (9, 9), 2) # Increase kernal size and sigma
# save_step(blurred_v3, "step_2_blurred_v3")

# filtered = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75) # Bilateral filter
# save_step(filtered, "step_2_filtered")

# median = cv2.medianBlur(gray, 5) # Median filter
# save_step(median, "step_2_median")

median_v2 = cv2.medianBlur(gray, 11) # Median filter
save_step(median_v2, "step_2_median_v2")

# denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21) # Non-local means denoising
# save_step(denoised, "step_2_denoised")

# denoised_v2 = cv2.fastNlMeansDenoising(gray, h=15, templateWindowSize=7, searchWindowSize=21)
# save_step(denoised_v2, "step_2_denoised_v2")

# blurred_v4 = cv2.GaussianBlur(median, (9, 9), 2) # Median filter followed by Gaussian blur
# save_step(blurred_v4, "step_2_blurred_v4")

# blurred_v5 = cv2.GaussianBlur(denoised, (9, 9), 2) # Denoising followed by Gaussian blur
# save_step(blurred_v5, "step_2_blurred_v5")

# blurred_v6 = cv2.GaussianBlur(median_v2, (9, 9), 2) # Aggresive Median filter followed by Gaussian blur
# save_step(blurred_v6, "step_2_blurred_v6")

blurred_v7 = cv2.GaussianBlur(median_v2, (9, 9), 2) # Aggresive Denoising followed by Gaussian blur
save_step(blurred_v7, "step_2_blurred_v7")


_, binary_mask = cv2.threshold(blurred_v7, 30, 255, cv2.THRESH_BINARY_INV) # If the pixel value is greater than 30, it will be set to 0 
save_step(binary_mask, "step_3_binary_mask")                               # 255 - max value to use (255 white - 0 black)

contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
wedge_contour = contours[0]

epsilon = 0.01 * cv2.arcLength(wedge_contour, True)
approx = cv2.approxPolyDP(wedge_contour, epsilon, True)
points = approx.reshape(-1, 2)

# domain left-upper, left-lower, and right-lower points
coords = cv2.findNonZero(blurred_v7).reshape(-1, 2)

# Extract left-upper, left-lower, and right-lower points
left_upper = coords[np.argmin(coords[:, 0] + coords[:, 1])]
left_lower = coords[np.argmin(coords[:, 0] - coords[:, 1])]
right_lower = coords[np.argmax(coords[:, 0] + coords[:, 1])]

# Stack corner points with existing points
corner_points = np.array([left_upper, left_lower, right_lower])
points = np.vstack([points, corner_points])

# Visualize detected points
approx_img = image.copy()
for pt in points:
    cv2.circle(approx_img, tuple(pt), 5, (0, 0, 255), -1)
save_step(approx_img, "step_4_approx_points")

# ---------------------------
# Scaling with Y-Axis Flip
# ---------------------------

# Identify reference points
#bottom_left, top_left = find_extreme_points(points)

# Compute pixel height and scale factor
pixel_height = left_lower[1] - left_upper[1]
scale_y = physical_domain_height / pixel_height
scale_x = scale_y  # maintain proportions

# Flip y-axis based on image coordinate system
image_height = image.shape[0]

# Translate and scale all points
scaled_points = []
for pt in points:
    dx = pt[0] - left_lower[0]
    # Flip Y by subtracting from image height
    dy = (image_height - pt[1]) - (image_height - left_lower[1])
    x_phys = dx * scale_x
    y_phys = dy * scale_y
    scaled_points.append((x_phys, y_phys))

# ---------------------------
# Export Results
# ---------------------------

# CSV output
csv_path = os.path.join(output_dir, "wedge_points.csv")
with open(csv_path, 'w') as f:
    f.write("x, y\n")
    for x, y in scaled_points:
        f.write(f"{x:.6f}, {y:.6f}\n")

# Gmsh .geo-compatible point definitions
gmsh_path = os.path.join(output_dir, "wedge_points_gmsh.txt")
with open(gmsh_path, 'w') as f:
    for idx, (x, y) in enumerate(scaled_points, start=1):
        f.write(f"Point({idx}) = {{{x:.6f}, {y:.6f}, 0, 1.0}};\n")

print("Processing complete.")
print("Results saved to:", output_dir)
