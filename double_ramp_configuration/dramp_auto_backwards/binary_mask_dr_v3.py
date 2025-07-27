import cv2
import numpy as np
import os
from point_reorder_gmsh import sort_points

def process_image(input_image_path, physical_domain_height=1.0, save_steps=False, output_dir=None):
    """
    Process CFD image to extract edge-based key points using Canny and Hough transform.

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

    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Median + Gaussian blur
    median_filtered = cv2.medianBlur(gray, 11)
    save_step(median_filtered, "step_1_median_blurred")
    blurred = cv2.GaussianBlur(median_filtered, (3, 3), 1)
    save_step(blurred, "step_2_gaussian_blurred")

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    save_step(enhanced, "step_3_clahe")

    # Normalize for better contrast in edges
    norm = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    save_step(norm, "step_4_normalized")
    # Canny edge detection
    edges = cv2.Canny(norm, 80, 200, apertureSize=3, L2gradient=True)
    save_step(edges, "step_5_canny_edges")

    # Hough line detection
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=120, maxLineGap=10)
    edge_pts = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Optional: filter by angle if ramps are expected to have steep slopes
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if 20 < abs(angle) < 75:  # You can tune this based on ramp orientation
                edge_pts.append((x1, y1))
                edge_pts.append((x2, y2))

    if not edge_pts:
        raise RuntimeError("No edge points detected.")

    # Optional: Draw detected edge points
    edge_overlay = image.copy()
    for pt in edge_pts:
        cv2.circle(edge_overlay, pt, 3, (0, 0, 255), -1)
    save_step(edge_overlay, "step_6_detected_edge_points")

    # Select reference corner points for scaling
    all_coords = np.array(edge_pts)
    left_upper = all_coords[np.argmin(all_coords[:, 0] + all_coords[:, 1])]
    left_lower = all_coords[np.argmin(all_coords[:, 0] - all_coords[:, 1])]

    # Estimate bottom-right point
    right_lower = all_coords[np.argmax(all_coords[:, 0] + all_coords[:, 1])]

    pixel_height = left_lower[1] - left_upper[1]
    if pixel_height == 0:
        raise ValueError("Invalid reference points for scaling.")

    scale_y = physical_domain_height / pixel_height
    scale_x = scale_y  # assume square pixels
    image_height = image.shape[0]

    scaled_points = []
    for pt in edge_pts:
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