# cv_processing.py

import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2
from point_reorder_gmsh import sort_points


def extract_metadata(filename):
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
    if filters["ramp1"] is not None and meta["ramp1"] != filters["ramp1"]:
        return False
    if filters["ramp2"] is not None and meta["ramp2"] != filters["ramp2"]:
        return False
    if filters["min_ma"] is not None and meta["ma"] < filters["min_ma"]:
        return False
    if filters["max_ma"] is not None and meta["ma"] > filters["max_ma"]:
        return False
    return True

'''
# vol_1 - causing scaling problems 256x256 to 1280x960 -- broken scaling
def extract_image_from_array(npz_path, data_key):
    data = np.load(npz_path)
    if data_key not in data:
        raise KeyError(f"'{data_key}' not found. Available keys: {list(data.keys())}")
    array = data[data_key]

    fig, ax = plt.subplots()
    # TODO: Find proper cmap to capture ramp geometry more easily!
    ax.imshow(array, cmap='viridis')
    ax.axis('off')

    canvas = FigureCanvas(fig)
    canvas.draw()
    image_np = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image_np = image_np.reshape(canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return image_np
'''

# vol2 -adapted to input dimensions
def extract_image_from_array(npz_path, data_key):
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

    # Get the real size of the canvas output (this avoids reshaping errors)
    canvas_width, canvas_height = canvas.get_width_height()

    image_np = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image_np = image_np.reshape((canvas_height, canvas_width, 3))
    plt.close(fig)

    return image_np

# vol 1 - baseline

def process_image_from_array(image_array, physical_domain_height=1.0):
    image = image_array.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    median_v2 = cv2.medianBlur(gray, 11)
    # blurred_v7 = cv2.GaussianBlur(median_v2, (9, 9), 2)
    # TODO: Find proper thresholding method to capture ramp geometry more easily! - 20-40 It varies along the cases
    # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    # enhanced = clahe.apply(blurred_v7)
    
    threshold_value = 30  # tweak this manually based on what works best
    _, binary_mask = cv2.threshold(median_v2, threshold_value, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No contours found.")
    wedge_contour = max(contours, key=cv2.contourArea)

    epsilon = 0.0025 * cv2.arcLength(wedge_contour, True)
    approx = cv2.approxPolyDP(wedge_contour, epsilon, True)
    
    # edited part due to the right upper corner extra point issue
    points = approx.reshape(-1, 2)
    
    # Get indices that would sort the points by x-coordinate
    sorted_indices = np.argsort(points[:, 0])

    # Find the two smallest-x points
    first_index = sorted_indices[0]   # Smallest x
    second_index = sorted_indices[1]  # 2nd smallest x
    
    # Copy the y-value of the smallest-x point to the 2nd smallest-x point
    points[second_index, 1] = points[first_index, 1]


    # Define search tolerance
    x_tolerance = 10

    # Compute approximate max x and y
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
    coords = cv2.findNonZero(median_v2).reshape(-1, 2) # was - enhanced
    left_upper = coords[np.argmin(coords[:, 0] + coords[:, 1])]
    left_lower = coords[np.argmin(coords[:, 0] - coords[:, 1])]
    right_lower = coords[np.argmax(coords[:, 0] + coords[:, 1])]

    #corner_points = np.array([left_upper, left_lower, right_lower])
    # corner_points = np.array([left_upper])
    
    # points = np.vstack([points, corner_points])
    
        # Ensure exactly 6 points: add if less, truncate if more
    desired_num_points = 6
    contour_points = wedge_contour.reshape(-1, 2)
    existing_points = points.tolist()

    # Add points if fewer than 6
    if len(existing_points) < desired_num_points:
        # Remove duplicates
        contour_set = {tuple(pt) for pt in contour_points}
        existing_set = {tuple(pt) for pt in existing_points}
        candidates = list(contour_set - existing_set)

        while len(existing_points) < desired_num_points and candidates:
            # Pick the candidate with the **max distance to the closest existing point**
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
            farthest = max(remaining, key=lambda pt: min(np.linalg.norm(np.array(pt) - np.array(sel)) for sel in selected))
            selected.append(farthest)
        existing_points = selected

    # Final array conversion
    points = np.array(existing_points)
    
    '''
     # Ensure exactly 6 points by adding one more from the contour if needed
    if len(points) < 6:
        # Find a point in the original contour that is not already in the points
        contour_points = wedge_contour.reshape(-1, 2)

        # Convert to sets of tuples for easy comparison
        existing_points_set = set(tuple(p) for p in points)
        
        # Find candidate point that's furthest from existing points to add diversity
        max_dist = -1
        best_candidate = None
        for pt in contour_points:
            pt_tuple = tuple(pt)
            if pt_tuple not in existing_points_set:
                # Calculate distance to the closest existing point
                dists = [np.linalg.norm(pt - ep) for ep in points]
                min_dist = min(dists)
                if min_dist > max_dist:
                    max_dist = min_dist
                    best_candidate = pt

        if best_candidate is not None:
            points = np.vstack([points, best_candidate])
    '''  
    points_lower_wall = points.copy()
    points_lower_wall[:, 1] = left_lower[1]
    
    points = np.vstack([points, points_lower_wall])


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

    scaled_points = sort_points(scaled_points)
    return scaled_points

