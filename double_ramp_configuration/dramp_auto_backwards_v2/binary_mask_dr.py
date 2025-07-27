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

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    median_v2 = cv2.medianBlur(gray, 11)
    blurred_v7 = cv2.GaussianBlur(median_v2, (9, 9), 2)
    _, binary_mask = cv2.threshold(blurred_v7, 30, 255, cv2.THRESH_BINARY_INV)

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

# points = process_image(
#     input_image_path="./pressure_data_from_rim/density_plot.png",
#     physical_domain_height=1.0,
#     save_steps=True,
#     output_dir="./pressure_data_from_rim/dummy_results"
# )

# print(points)