"""

Step 1.1

cv_processing.py

"""

import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2
from src.point_reorder_gmsh import sort_points


def extract_metadata(filename):
    """
    Extracts metadata from a filename using a regex pattern.

    Args:
        filename (str): The filename to parse.

    Returns:
        dict or None: Dictionary with keys 'ramp1', 'ramp2', 'ma', 'pres' if matched, else None.
    """
    pattern = r"double_ramp_(\d+\.\d+)(?:_(\d+\.\d+))?_ma_(\d+\.\d+)_pres_(\d+)"
    match = re.search(pattern, filename)
    if not match:
        return None
    ramp1 = float(match.group(1))
    ramp2 = float(match.group(2)) if match.group(2) else None
    ma = float(match.group(3))
    pres = int(match.group(4))
    return {"ramp1": ramp1, "ramp2": ramp2, "ma": ma, "pres": pres}


def matches_filters(meta, filters):
    """
    Checks if metadata matches the provided filter criteria.

    Args:
        meta (dict): Metadata dictionary.
        filters (dict): Filter criteria with possible keys: 'ramp1', 'ramp2', 'min_ma', 'max_ma'.

    Returns:
        bool: True if all filters match, False otherwise.
    """
    if filters["ramp1"] is not None and meta["ramp1"] != filters["ramp1"]:
        return False
    if filters["ramp2"] is not None and meta["ramp2"] != filters["ramp2"]:
        return False
    if filters["min_ma"] is not None and meta["ma"] < filters["min_ma"]:
        return False
    if filters["max_ma"] is not None and meta["ma"] > filters["max_ma"]:
        return False
    return True


def extract_image_from_array(npz_path, data_key):
    """
    Loads a 2D array from a .npz file and converts it to an RGB image using matplotlib.

    Args:
        npz_path (str): Path to the .npz file.
        data_key (str): Key of the array to extract.

    Returns:
        np.ndarray: RGB image as a numpy array.
    """
    data = np.load(npz_path)
    if data_key not in data:
        raise KeyError(f"'{data_key}' not found. Available keys: {list(data.keys())}")
    array = data[data_key]

    height, width = array.shape
    dpi = 100
    figsize = (width / dpi, height / dpi)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(array, cmap='viridis')
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    canvas = FigureCanvas(fig)
    canvas.draw()

    # Get the real size of the canvas output - this avoids reshaping errors
    canvas_width, canvas_height = canvas.get_width_height()
    image_np = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image_np = image_np.reshape((canvas_height, canvas_width, 3))
    plt.close(fig)

    return image_np


def process_image_from_array(image_array, physical_domain_height=1.0):
    """
    Extracts and scales key geometric points from an image array.

    Args:
        image_array (np.ndarray): RGB image as a numpy array.
        physical_domain_height (float): Physical height for scaling.

    Returns:
        list of tuple: Sorted list of (x, y) points in physical coordinates.
    """
    # --- Preprocessing: Convert to grayscale and apply median blur ---
    image = image_array.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.medianBlur(gray, 11)

    # --- Thresholding to create binary mask ---
    threshold_value = 30  # Adjust as needed for your images
    _, binary_mask = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # --- Find the largest contour (assumed to be the wedge) ---
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No contours found in the image.")
    wedge_contour = max(contours, key=cv2.contourArea)

    # --- Approximate the contour to a polygon ---
    epsilon = 0.0025 * cv2.arcLength(wedge_contour, True)
    approx = cv2.approxPolyDP(wedge_contour, epsilon, True)
    points = approx.reshape(-1, 2)

    # --- Align left wall by copying y-value of leftmost point to next leftmost ---
    sorted_indices = np.argsort(points[:, 0])
    points[sorted_indices[1], 1] = points[sorted_indices[0], 1]

    # --- Remove extra right upper corner point if present ---
    x_tolerance = 10
    max_x = np.max(points[:, 0])
    candidate_mask = (points[:, 0] >= max_x - x_tolerance)
    candidate_points = points[candidate_mask]
    if len(candidate_points) > 1:
        min_y = np.min(candidate_points[:, 1])
        to_remove_mask = candidate_mask & (points[:, 1] == min_y)
        points = points[~to_remove_mask]

    # --- Detect key corners in the image for scaling ---
    coords = cv2.findNonZero(blurred).reshape(-1, 2)
    left_upper = coords[np.argmin(coords[:, 0] + coords[:, 1])]
    left_lower = coords[np.argmin(coords[:, 0] - coords[:, 1])]
    right_lower = coords[np.argmax(coords[:, 0] + coords[:, 1])]

    # --- Ensure exactly 6 points: add or truncate as needed ---
    desired_num_points = 6
    contour_points = wedge_contour.reshape(-1, 2)
    existing_points = points.tolist()

    # Add points if fewer than 6
    if len(existing_points) < desired_num_points:
        contour_set = {tuple(pt) for pt in contour_points}
        existing_set = {tuple(pt) for pt in existing_points}
        candidates = list(contour_set - existing_set)
        while len(existing_points) < desired_num_points and candidates:
            # Pick candidate farthest from existing points
            best_candidate = None
            max_min_dist = -1
            for cand in candidates:
                dists = [np.linalg.norm(np.array(cand) - np.array(ep)) for ep in existing_points]
                min_dist = min(dists)
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_candidate = cand
            if best_candidate:
                existing_points.append(list(best_candidate))
                candidates.remove(best_candidate)

    # Truncate if more than 6
    elif len(existing_points) > desired_num_points:
        # Greedy farthest point sampling (FPS) to keep the most spatially distinct 6
        selected = [existing_points[0]]
        while len(selected) < desired_num_points:
            remaining = [p for p in existing_points if p not in selected]
            farthest = max(
                remaining,
                key=lambda pt: min(np.linalg.norm(np.array(pt) - np.array(sel)) for sel in selected)
            )
            selected.append(farthest)
        existing_points = selected

    # Convert to numpy array for further processing
    points = np.array(existing_points)

    # --- Duplicate points for lower wall alignment ---
    points_lower_wall = points.copy()
    points_lower_wall[:, 1] = left_lower[1]
    points = np.vstack([points, points_lower_wall])

    # --- Scale points to physical domain ---
    pixel_height_y = left_lower[1] - left_upper[1]
    pixel_height_x = right_lower[0] - left_lower[0]
    scale_y = physical_domain_height / pixel_height_y
    scale_x = physical_domain_height / pixel_height_x
    image_height = image.shape[0]

    scaled_points = []
    for pt in points:
        dx = pt[0] - left_lower[0]
        dy = (image_height - pt[1]) - (image_height - left_lower[1])
        x_phys = dx * scale_x
        y_phys = dy * scale_y
        scaled_points.append((x_phys, y_phys))

    # --- Sort and return the scaled points
    scaled_points = sort_points(scaled_points)
    return scaled_points

