import cv2
import numpy as np
import os
from point_reorder_gmsh import sort_points

def process_image(input_image_path, physical_domain_height=1.0, save_steps=False, output_dir=None):
    """
    Process CFD image to extract scaled physical coordinates of key points.

    Parameters:
        input_image_path (str): Path to the input image.
        physical_domain_height (float): Physical height of domain in real units.
        save_steps (bool): Whether to save intermediate images.
        output_dir (str): Directory to save intermediate images if save_steps is True.

    Returns:
        List[Tuple[float, float]]: Scaled (x, y) coordinates in physical units.
    """
    if save_steps:
        assert output_dir is not None, "Output directory must be specified if save_steps is True."
        os.makedirs(output_dir, exist_ok=True)

    def save_step(img, step_name):
        if save_steps:
            path = os.path.join(output_dir, f"{step_name}.png")
            cv2.imwrite(path, img)
    """
    image = cv2.imread(input_image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at: {input_image_path}")
    save_step(image, "step_0_original")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    median_v2 = cv2.medianBlur(gray, 11)
    blurred_v7 = cv2.GaussianBlur(median_v2, (9, 9), 2)
    save_step(blurred_v7, "step_2_before_binary_mask")                               
    _, binary_mask = cv2.threshold(blurred_v7, 35, 255, cv2.THRESH_BINARY_INV)
    save_step(binary_mask, "step_3_binary_mask")                               # 255 - max value to use (255 white - 0 black)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No contours found in image.")
    wedge_contour = max(contours, key=cv2.contourArea)

    epsilon = 0.01 * cv2.arcLength(wedge_contour, True)
    approx = cv2.approxPolyDP(wedge_contour, epsilon, True)
    points = approx.reshape(-1, 2)

    coords = cv2.findNonZero(blurred_v7).reshape(-1, 2)
    left_upper = coords[np.argmin(coords[:, 0] + coords[:, 1])]
    left_lower = coords[np.argmin(coords[:, 0] - coords[:, 1])]
    right_lower = coords[np.argmax(coords[:, 0] + coords[:, 1])]

    corner_points = np.array([left_upper, left_lower, right_lower])
    points = np.vstack([points, corner_points])
    """
    
    image = cv2.imread(input_image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at: {input_image_path}")
    save_step(image, "step_0_original")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 1: Smoothing
    median_filtered = cv2.medianBlur(gray, 11)
    blurred = cv2.GaussianBlur(median_filtered, (9, 9), 2)

    # Step 2: Contrast Enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    save_step(enhanced, "step_2_contrast_enhanced")
    
        # Assuming enhanced is the CLAHE output
    threshold_value = 32  # tweak this manually based on what works best
    _, binary_mask = cv2.threshold(enhanced, threshold_value, 255, cv2.THRESH_BINARY_INV)
    save_step(binary_mask, "step_3_binary_mask")  # Save the binary mask for debugging

    # # Step 3: Adaptive Thresholding
    # adaptive_thresh = cv2.adaptiveThreshold(
    #     enhanced, 255,
    #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #     cv2.THRESH_BINARY_INV, 11, 2)
    # save_step(adaptive_thresh, "step_3_adaptive_thresh")

    # # Step 4: Morphological Refinement
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # opened = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
    # save_step(closed, "step_4_morph_refined")

    # # Step 5: Gradient Filtering
    # sobelx = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=5)
    # sobely = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=5)
    # gradient_mag = cv2.magnitude(sobelx, sobely)
    # gradient_mag = cv2.convertScaleAbs(gradient_mag)
    # _, gradient_thresh = cv2.threshold(gradient_mag, 30, 255, cv2.THRESH_BINARY)
    # save_step(gradient_thresh, "step_5_gradient_mask")

    # # Step 6: Combine Masks
    # combined_mask = cv2.bitwise_or(closed, gradient_thresh)
    # save_step(combined_mask, "step_6_combined_mask")

    # # Step 7: Contour Detection (Filter Small Areas)
    # contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours = [c for c in contours if cv2.contourArea(c) > 500]  # Adjust threshold
    # if not contours:
    #     raise RuntimeError("No valid contours found in image.")

    # # Step 8: Find largest contour
    # wedge_contour = max(contours, key=cv2.contourArea)
    
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No contours found in image.")
    wedge_contour = max(contours, key=cv2.contourArea)
    
    # Step 9: Approximate shape
    epsilon = 0.01 * cv2.arcLength(wedge_contour, True)
    approx = cv2.approxPolyDP(wedge_contour, epsilon, True)
    points = approx.reshape(-1, 2)

    # Step 10: Corner detection
    coords = cv2.findNonZero(enhanced).reshape(-1, 2)
    left_upper = coords[np.argmin(coords[:, 0] + coords[:, 1])]
    left_lower = coords[np.argmin(coords[:, 0] - coords[:, 1])]
    right_lower = coords[np.argmax(coords[:, 0] + coords[:, 1])]

    corner_points = np.array([left_upper, left_lower, right_lower])
    points = np.vstack([points, corner_points])

    ### Will deleted
    # Visualize detected points
    approx_img = image.copy()
    for pt in points:
        cv2.circle(approx_img, tuple(pt), 5, (0, 0, 255), -1)
    save_step(approx_img, "step_4_approx_points")
    ### Will deleted
    
    
    # Scaling
    pixel_height = left_lower[1] - left_upper[1]
    scale_y = physical_domain_height / pixel_height
    scale_x = scale_y
    image_height = image.shape[0]

    scaled_points = []
    for pt in points:
        dx = pt[0] - left_lower[0]
        dy = (image_height - pt[1]) - (image_height - left_lower[1])
        x_phys = dx * scale_x
        y_phys = dy * scale_y
        scaled_points.append((round(x_phys, 6), round(y_phys, 6)))
        scaled_points = sort_points(scaled_points)

    return scaled_points

points = process_image(
    input_image_path="./double_ramp_configuration/dramp_auto_backwards/density_plot.png",
    physical_domain_height=1.0,
    save_steps=True,
    output_dir="./double_ramp_configuration/dramp_auto_backwards/dummy_results"
)

print(points)