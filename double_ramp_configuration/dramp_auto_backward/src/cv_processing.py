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

# ---------------------------------------------------------------------------
# Aggressive ramp start (2nd point) detection configuration
# These defaults can be tuned if needed; they are intentionally conservative
# except for the ramp drop threshold to aggressively capture the transition.
# ---------------------------------------------------------------------------
AGGR_ENABLE = True  # Master toggle for aggressive 2nd point capture
AGGR_LEFT_BAND_WIDTH = 12      # Pixels from left boundary to characterize plateau
AGGR_PLATEAU_Y_TOL = 2         # ± pixel tolerance to consider y still on plateau
AGGR_RAMP_DROP_THRESH = 3      # First y increase (downward in image coords) beyond this indicates ramp start
AGGR_INTERPOLATE = False       # If True, linearly interpolate ramp start (rarely necessary; contour is discrete)

def _detect_ramp_start_point(contour):
    """Detect the x-position at which the top flat (plateau) ends and the first ramp begins.

    Parameters
    ----------
    contour : np.ndarray
        Array of shape (N, 1, 2) or (N,2) with integer pixel coordinates (OpenCV contour).

    Returns
    -------
    (int, int) or None
        (x_plateau_end, plateau_y) for the SECOND point. plateau_y is the y of the left/top start.
        Returns None if a confident detection is not possible (fallback logic will be used).
    """
    try:
        pts = contour.reshape(-1, 2)
        # Identify leftmost x and gather plateau candidate y's within band
        min_x = int(np.min(pts[:, 0]))
        band_mask = pts[:, 0] <= (min_x + AGGR_LEFT_BAND_WIDTH)
        band_pts = pts[band_mask]
        if band_pts.size == 0:
            return None
        # Plateau y taken as robust central tendency (median)
        plateau_y = int(np.median(band_pts[:, 1]))

        # Sort by x to scan outward
        order = np.argsort(pts[:, 0])
        last_plateau_idx = None
        for idx in order:
            x, y = pts[idx]
            if abs(int(y) - plateau_y) <= AGGR_PLATEAU_Y_TOL:
                last_plateau_idx = idx
                continue
            # y increases (image coordinate) when geometry goes downward physically
            if last_plateau_idx is not None and (y - plateau_y) >= AGGR_RAMP_DROP_THRESH:
                # Optional interpolation between last plateau point and this first drop point
                if AGGR_INTERPOLATE:
                    x_prev, y_prev = pts[last_plateau_idx]
                    dy_total = y - y_prev
                    if dy_total <= 0:
                        # Degenerate; fallback to previous x
                        return int(x_prev), int(plateau_y)
                    # Fraction until threshold reached
                    frac = (AGGR_RAMP_DROP_THRESH) / dy_total
                    frac = max(0.0, min(1.0, frac))
                    x_interp = x_prev + frac * (x - x_prev)
                    return int(round(x_interp)), int(plateau_y)
                return int(pts[last_plateau_idx][0]), int(plateau_y)
        # If we never observed a significant drop, still provide the furthest plateau x
        if last_plateau_idx is not None:
            return int(pts[last_plateau_idx][0]), int(plateau_y)
    except Exception:
        return None
    return None


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


def extract_image_from_array(npz_path, data_key, return_raw: bool = False):
    """Load a 2D array from a .npz file and render an RGB image via matplotlib.

    Parameters
    ----------
    npz_path : str
        Path to the .npz file.
    data_key : str
        Key of the array to extract.
    return_raw : bool, default False
        If True returns a tuple (rgb_image, raw_array). Otherwise only the RGB image
        (legacy behavior maintained).

    Returns
    -------
    np.ndarray or (np.ndarray, np.ndarray)
        RGB image (H x W x 3, uint8) or tuple with raw scalar array.
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

    if return_raw:
        return image_np, array
    return image_np


def process_image_from_array(image_array, physical_domain_height=1.0, return_debug: bool = False):
    """Extract and scale key geometric points from an image array.

    Parameters
    ----------
    image_array : np.ndarray
        RGB image as a numpy array.
    physical_domain_height : float, default 1.0
        Physical height used to scale pixel coordinates into physical coordinates.
    return_debug : bool, default False
        If True returns a tuple (scaled_points, debug_dict) where debug_dict contains
        intermediate data required for visualization. If False (default) maintains
        legacy behavior returning only the list of scaled points.

    Returns
    -------
    list[(float, float)] or (list[(float,float)], dict)
        Scaled points (and optionally debug info when return_debug=True).
    """
    # --- Preprocessing: Convert to grayscale and apply median blur ---
    image = image_array.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.medianBlur(gray, 9) #was 11

    # --- Thresholding to create binary mask ---
    threshold_value = 30  # Adjust as needed for your images
    _, binary_mask = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # --- Find the largest contour (assumed to be the wedge) ---
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No contours found in the image.")
    wedge_contour = max(contours, key=cv2.contourArea)

    # --- Approximate the contour to a polygon ---
    epsilon = 0.00235 * cv2.arcLength(wedge_contour, True)
    approx = cv2.approxPolyDP(wedge_contour, epsilon, True)
    points = approx.reshape(-1, 2)

    # --- Aggressive detection for 2nd point (ramp start) BEFORE further edits ---
    detected_second_point = None
    if AGGR_ENABLE:
        detected_second_point = _detect_ramp_start_point(wedge_contour)

    # We'll insert / enforce this after ensuring we have at least 2 distinct x's

    # --- Align left wall by copying y-value of leftmost point to next leftmost ---
    sorted_indices = np.argsort(points[:, 0])
    if len(sorted_indices) >= 2:
        points[sorted_indices[1], 1] = points[sorted_indices[0], 1]

    # --- Enforce aggressively detected 2nd point (x of plateau end, y of first point) ---
    if detected_second_point is not None and len(sorted_indices) >= 2:
        first_idx = sorted_indices[0]
        # Replace second point's coordinates
        second_idx = sorted_indices[1]
        x_candidate, plateau_y = detected_second_point
        # Guarantee ordering: x_candidate should be >= first point's x
        if x_candidate < points[first_idx][0]:
            x_candidate = points[first_idx][0]
        points[second_idx][0] = x_candidate
        points[second_idx][1] = points[first_idx][1]  # same y as first (plateau)

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

        # Enforce a minimum separation so we don't add a point like (511, 0) that is
        # effectively a duplicate of an already chosen corner and degrades the logic.
        # For 512x512 domain: diag ≈ 724. Tune thresholds to balance coverage & mesh quality.
        bbox_min = np.min(contour_points, axis=0)
        bbox_max = np.max(contour_points, axis=0)
        bbox_diag = np.linalg.norm(bbox_max - bbox_min)
        # Dynamic part plus floor: raise floor for stability.
        base_min_sep = 8.0  # ~6% of 512, prevents very tight clustering
        dynamic_min_sep = 0.07 * bbox_diag  # ~50 px for 512^2 domain
        min_separation = max(base_min_sep, dynamic_min_sep)

        # Additional guard: block adding any new point whose x is within a tiny tolerance
        # of an existing point's x (we only care about x clustering for mesh quality).
        x_dup_tol = 3  # pixels tolerance in x (adjust if needed)
        def x_too_close(pt, pts):
            return any(abs(pt[0] - p[0]) <= x_dup_tol for p in pts)

        # Initial pruning of candidates too close overall OR too close in x
        def far_enough(pt, pts):
            return all(np.linalg.norm(np.array(pt) - np.array(p)) >= min_separation for p in pts)

        candidates = [c for c in candidates if far_enough(c, existing_points) and not x_too_close(c, existing_points)]

        while len(existing_points) < desired_num_points and candidates:
            best_candidate = None
            max_min_dist = -1.0
            for cand in candidates:
                if x_too_close(cand, existing_points):
                    continue
                dists = [np.linalg.norm(np.array(cand) - np.array(ep)) for ep in existing_points]
                min_dist = min(dists)
                if min_dist < min_separation:
                    continue
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_candidate = cand
            if best_candidate is None:
                # No candidate satisfies requirements; stop trying to add more
                break
            existing_points.append(list(best_candidate))
            # Remove chosen candidate and re-prune remaining list with updated constraints
            candidates.remove(best_candidate)
            candidates = [c for c in candidates if far_enough(c, existing_points) and not x_too_close(c, existing_points)]
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

    if not return_debug:
        return scaled_points

    debug = {
        "image_rgb": image_array,  # original rendered RGB image
        "scale_x": scale_x,
        "scale_y": scale_y,
        "left_lower": tuple(left_lower.tolist()),
        "left_upper": tuple(left_upper.tolist()),
        "right_lower": tuple(right_lower.tolist()),
        "image_height": image_height,
        "aggr_second_point_pixel": detected_second_point,
        "aggr_params": {
            "enabled": AGGR_ENABLE,
            "left_band_width": AGGR_LEFT_BAND_WIDTH,
            "plateau_y_tol": AGGR_PLATEAU_Y_TOL,
            "ramp_drop_thresh": AGGR_RAMP_DROP_THRESH,
            "interpolate": AGGR_INTERPOLATE,
        }
    }
    return scaled_points, debug

