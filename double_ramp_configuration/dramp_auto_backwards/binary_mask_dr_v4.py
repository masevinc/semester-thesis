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

    
    image = cv2.imread(input_image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at: {input_image_path}")
    save_step(image, "step_0_original")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 1: Smoothing
    median_filtered = cv2.medianBlur(gray, 7)
    blurred = cv2.GaussianBlur(median_filtered, (5, 5), 1)

    # Step 2: Contrast Enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    save_step(enhanced, "step_2_contrast_enhanced")
    
        # Assuming enhanced is the CLAHE output
    _, binary_mask = cv2.threshold(enhanced, 230, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    save_step(binary_mask, "step_3_binary_mask")  # Save the binary mask for debugging

    cleared_mask = binary_mask.copy()
    cv2.rectangle(cleared_mask, (0, 0), (cleared_mask.shape[1]-1, cleared_mask.shape[0]-1), 0, thickness=3)
    save_step(cleared_mask, "step_4_cleared_mask") 
    contours, _ = cv2.findContours(cleared_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise RuntimeError("No contours found in image.")
    wedge_contour = max(contours, key=cv2.contourArea)
    
    # Step 9: Approximate shape
    epsilon = 0.01 * cv2.arcLength(wedge_contour, True)
    approx = cv2.approxPolyDP(wedge_contour, epsilon, True)
    
    # Reshape to 2D array
    points = approx.reshape(-1, 2)

    # Define search tolerance
    x_tolerance = 10

    # Compute approximate max x
    max_x = np.max(points[:, 0])

    # Find all points within tolerance of max x
    candidate_mask = (points[:, 0] >= max_x - x_tolerance)
    candidate_points = points[candidate_mask]

    # Only proceed if more than one candidate exists
    if len(candidate_points) > 1:
        # Find the one with min y among candidates
        min_y = np.min(candidate_points[:, 1])
        to_remove_mask = candidate_mask & (points[:, 1] == min_y)
        
        # Remove that point
        points = points[~to_remove_mask]
        
        # Step 10: Corner detection
    coords = cv2.findNonZero(enhanced).reshape(-1, 2)
    left_upper = coords[np.argmin(coords[:, 0] + coords[:, 1])]
    left_lower = coords[np.argmin(coords[:, 0] - coords[:, 1])]
    right_lower = coords[np.argmax(coords[:, 0] + coords[:, 1])]

    corner_points = np.array([left_upper, left_lower, right_lower])
    points = np.vstack([points, corner_points])
    '''   
    
    # Safer version using only pixels within the contour
    mask = np.zeros_like(enhanced, dtype=np.uint8)
    cv2.drawContours(mask, [wedge_contour], -1, color=255, thickness=-1)  # Fill the contour
    inside_pixels = cv2.findNonZero(mask).reshape(-1, 2)

    left_upper = inside_pixels[np.argmin(inside_pixels[:, 0] + inside_pixels[:, 1])]
    left_lower = inside_pixels[np.argmin(inside_pixels[:, 0] - inside_pixels[:, 1])]
    right_lower = inside_pixels[np.argmax(inside_pixels[:, 0] + inside_pixels[:, 1])]
    '''
        
    ### Will deleted
    # Visualize detected points
    approx_img = image.copy()
    for pt in points:
        cv2.circle(approx_img, tuple(pt), 5, (0, 0, 255), -1)
    save_step(approx_img, "step_5_approx_points")
    ### Will deleted
    
    
    # Scaling
    pixel_height = left_lower[1] - left_upper[1]
    if pixel_height == 0:
        raise ValueError("Pixel height between left_upper and left_lower is zero. Check detected points.")

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