"""

Step 1.1

cv_processing.py

"""

import re
import os
import json
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
    _primary_render_error = None
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


# ---------------------------------------------------------------------------
# Helper utilities (new) to generalize the workflow to arbitrary images
# ---------------------------------------------------------------------------
def load_image_as_rgb(path: str) -> np.ndarray:
    """Load an image file (png/jpg) as RGB uint8 array.

    Parameters
    ----------
    path : str
        Image filepath.

    Returns
    -------
    np.ndarray
        (H,W,3) RGB image.
    """
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image file: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def generate_binary_mask(gray: np.ndarray, threshold_value: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Generate raw and cleaned binary mask following project conventions.

    Returns (binary_mask_raw, binary_mask_clean) where clean has ramp black (0) and flow white (255).
    """
    blurred = cv2.medianBlur(gray, 9)
    _t, inv = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No contours found while generating binary mask.")
    wedge_contour = max(contours, key=cv2.contourArea)
    h, w = inv.shape
    clean = np.full((h, w), 255, dtype=np.uint8)
    cv2.drawContours(clean, [wedge_contour], -1, 0, -1)
    return inv, clean


def find_wedge_contour(binary_mask_raw: np.ndarray) -> np.ndarray:
    """Return the largest contour from a raw inverted binary mask."""
    contours, _ = cv2.findContours(binary_mask_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No contours found in provided mask")
    return max(contours, key=cv2.contourArea)


def process_image_file(
    image_path: str,
    physical_domain_height: float = 1.0,
    physical_domain_width: float | None = None,
    return_debug: bool = False,
    save_debug_dir: str | None = None,
    save_basename: str | None = None,
    save_binary_mask: bool = False,
    save_contour_overlay: bool = False,
    json_output: str | None = None,
):
    """High-level convenience wrapper to process a normal image file.

    Mirrors arguments of ``process_image_from_array``; loads image then delegates.
    Optionally writes scaled points (and debug when requested) to a JSON file.
    """
    rgb = load_image_as_rgb(image_path)
    result = process_image_from_array(
        rgb,
        physical_domain_height=physical_domain_height,
        physical_domain_width=physical_domain_width,
        return_debug=return_debug,
        save_debug_dir=save_debug_dir,
        save_basename=save_basename,
        save_binary_mask=save_binary_mask,
        save_contour_overlay=save_contour_overlay,
    )
    if json_output:
        if return_debug:
            scaled, dbg = result
            payload = {"points": scaled, "debug": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in dbg.items() if k not in {"image_rgb"}}}
        else:
            scaled = result
            payload = {"points": scaled}
        with open(json_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return result


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
    # NOTE: allow_pickle=True retained (legacy files may contain pickled python objects). We immediately
    # validate & coerce; if security is a concern and you fully control data generation, set to False
    # AFTER re-exporting all .npz with pure numeric arrays.
    data = np.load(npz_path, allow_pickle=True)
    if data_key not in data:
        raise KeyError(f"'{data_key}' not found in {os.path.basename(npz_path)}. Available keys: {list(data.keys())}")
    array = data[data_key]

    # --- Robust validation & auto-repair attempts --------------------------------------
    # Historical user error reported: "object __array__ method not producing an array".
    # This typically means an element inside an object-dtype ndarray defines __array__ but
    # returns a non-array object, OR the top-level object itself is not a pure numeric array.
    # We defensively coerce or raise with a VERY explicit diagnostic.
    import numpy as _np
    original_type = type(array)
    original_dtype = getattr(array, 'dtype', None)

    def _coerce_object_array(obj_arr):
        """Attempt to coerce an object-dtype ndarray into a numeric 2D float array.

        Strategies:
          1. If elements are scalar-like (int/float/np.number) -> vectorized float cast.
          2. If elements are 0-d ndarrays -> extract .item().
          3. If elements are lists/tuples of length 1 -> take first element.
        Returns coerced ndarray or raises ValueError.
        """
        flat = obj_arr.ravel()
        cleaned = []
        for el in flat:
            # unwrap 0-d arrays
            if isinstance(el, _np.ndarray) and el.shape == ():
                el = el.item()
            # unwrap length-1 containers
            if isinstance(el, (list, tuple)) and len(el) == 1:
                el = el[0]
            if isinstance(el, (_np.integer, _np.floating, int, float)):
                cleaned.append(float(el))
            else:
                raise ValueError(f"Encountered non-numeric element of type {type(el)} while coercing object array")
        coerced = _np.array(cleaned, dtype=float).reshape(obj_arr.shape)
        return coerced

    try:
        # Fast path: proper ndarray numeric & 2D
        if isinstance(array, _np.ndarray) and original_dtype is not None and original_dtype is not object:
            if array.ndim != 2:
                raise ValueError(f"Array for key '{data_key}' in {os.path.basename(npz_path)} must be 2D, got shape {array.shape}")
        else:
            # Try to convert to ndarray (covers python lists, nested lists, masked arrays, etc.)
            try:
                array = _np.array(array)
            except Exception as conv_err:
                raise TypeError(f"Failed raw np.array() conversion for key '{data_key}' in {os.path.basename(npz_path)}: {conv_err}") from conv_err
            if array.dtype == object:
                # Extra guard: detect elements advertising __array__ incorrectly (common with mismatched library versions)
                try:
                    array = _coerce_object_array(array)
                except Exception as coercion_err:
                    sample_types = _np.unique([type(x).__name__ for x in array.ravel()[:25]])
                    raise TypeError(
                        "Object-dtype array for key '{k}' contains non-numeric elements after macOS / dependency update. "
                        "Sample element types: {st}. Original file: {f}. Root: {r}".format(
                            k=data_key, st=list(sample_types), f=os.path.basename(npz_path), r=coercion_err
                        )
                    ) from coercion_err
            if array.ndim != 2:
                raise ValueError(f"Coerced array for key '{data_key}' not 2D (shape {array.shape}) in {os.path.basename(npz_path)}")
    except Exception as _e:
        # Provide rich context and re-raise. Build message then raise (easier to read & avoids nesting quotes issues).
        # Provide targeted remediation suggestions (esp. after OS / NumPy upgrades changing object model behavior)
        msg = (
            "Failed to obtain a clean 2D numeric array for key '{k}' in file '{f}'. "
            "Original type={t}, original dtype={d}. Root cause: {err}. Suggestions: (1) Re-create the .npz with a current "
            "NumPy version using plain numeric dtypes; (2) If you intentionally stored Python objects, refactor to store "
            "their numeric payloads only; (3) Pin numpy/scipy versions that produced the original dataset to re-export."
        ).format(k=data_key, f=os.path.basename(npz_path), t=original_type, d=original_dtype, err=_e)
        raise TypeError(msg) from _e

    height, width = array.shape
    dpi = 100
    figsize = (width / dpi, height / dpi)

    # Some environments / backend combinations (especially after downgrading matplotlib or mixing wheels)
    # can present a FigureCanvasAgg missing 'tostring_rgb'. We implement a robust, ordered fallback chain:
    #  1. Normal matplotlib render using tostring_rgb (fast path)
    #  2. If missing, try buffer_rgba() -> convert RGBA to RGB
    #  3. If still failing, use figure.canvas.print_to_buffer()
    #  4. Absolute fallback: bypass matplotlib entirely and manually map scalar field to viridis colormap.

    image_np = None
    try:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.imshow(array, cmap='viridis')
        ax.axis('off')
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        canvas = FigureCanvas(fig)
        canvas.draw()
        canvas_width, canvas_height = canvas.get_width_height()
        if hasattr(canvas, 'tostring_rgb'):
            raw = canvas.tostring_rgb()
            image_np = np.frombuffer(raw, dtype='uint8').reshape((canvas_height, canvas_width, 3))
        elif hasattr(canvas, 'buffer_rgba'):
            buf = canvas.buffer_rgba()
            rgba = np.asarray(buf, dtype=np.uint8).reshape((canvas_height, canvas_width, 4))
            image_np = rgba[:, :, :3].copy()
        else:
            # Attempt print_to_buffer
            if hasattr(canvas, 'print_to_buffer'):
                raw, (w, h) = canvas.print_to_buffer()
                image_np = np.frombuffer(raw, dtype='uint8').reshape((h, w, 4))[:, :, :3].copy()
        plt.close(fig)
    except Exception as render_err:
        # Swallow and proceed to manual fallback
        try:
            plt.close(fig)
        except Exception:
            pass
        _primary_render_error = render_err
        image_np = None

    if image_np is None:
        # Manual fallback: colormap mapping without figure.
        try:
            from matplotlib import cm
            arr = array.astype(np.float32)
            finite_mask = np.isfinite(arr)
            if not finite_mask.any():
                raise ValueError("All values are non-finite; cannot colorize.")
            vmin = float(arr[finite_mask].min())
            vmax = float(arr[finite_mask].max())
            if vmax == vmin:
                vmax = vmin + 1e-9
            normed = (arr - vmin) / (vmax - vmin)
            normed = np.clip(normed, 0.0, 1.0)
            vir = cm.get_cmap('viridis')
            rgba = vir(normed)  # (H, W, 4) float
            image_np = (rgba[:, :, :3] * 255).astype(np.uint8)
        except Exception as fallback_err:
            raise RuntimeError(
                f"Failed all rendering paths for key '{data_key}' in {os.path.basename(npz_path)}: primary={_primary_render_error}, manual={fallback_err}"
            ) from fallback_err

    if return_raw:
        return image_np, array
    return image_np


def process_image_from_array(
    image_array,
    physical_domain_height=1.0,
    physical_domain_width=None,
    return_debug: bool = False,
    save_debug_dir: str | None = None,
    save_basename: str | None = None,
    save_binary_mask: bool = False,
    save_contour_overlay: bool = False,
):
    """Extract and scale key geometric points from an image array.

    Parameters
    ----------
    image_array : np.ndarray
        RGB image as a numpy array.
    physical_domain_height : float, default 1.0
        Physical height used to scale pixel coordinates into physical coordinates (y-direction).
    physical_domain_width : float | None, default None
        Physical width for scaling x-direction. If None, uses *physical_domain_height* (legacy
        behavior assuming a square domain). Provide a value to enable rectangular scaling,
        e.g. width=0.62, height=0.40.
    return_debug : bool, default False
        If True returns a tuple (scaled_points, debug_dict) where debug_dict contains
        intermediate data required for visualization. If False (default) maintains
        legacy behavior returning only the list of scaled points.

        Additional optional save outputs (when save_debug_dir provided):
            * Binary mask PNG: ramp (solid) region black (0), flow field white (255) -> '<basename>_binary_mask.png' (now uses clean filled polygon to avoid wavy edges)
            * Contour overlay PNG: JUST the flow field (no lines/points) scaled to physical aspect ratio -> '<basename>_contour.png'

    Parameters (new)
    ----------------
    save_debug_dir : str | None
        If provided, directory where optional debug PNGs will be written.
    save_basename : str | None
        Base filename (without extension) used when saving debug PNGs. If None and saving
        requested, a ValueError is raised.
    save_binary_mask : bool
        Write binary mask PNG (ramp black, flow white) if True and save_debug_dir given.
    save_contour_overlay : bool
        Write overlay visualization if True and save_debug_dir given.

    Returns
    -------
    list[(float, float)] or (list[(float,float)], dict)
        Scaled points (and optionally debug info when return_debug=True). Debug dict now
        also contains 'binary_mask' when return_debug=True.
    """
    # --- Preprocessing: Convert to grayscale and apply median blur ---
    image = image_array.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.medianBlur(gray, 9) #was 11

    # --- Thresholding to create binary mask ---
    threshold_value = 40  # Adjust as needed for your images
    _, binary_mask_raw = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # We want ramp BLACK (0) and flow WHITE (255).
    # Create a clean polygon fill (avoid waviness from threshold noise):
    #   1) Use the largest contour (from binary_mask_raw) but draw filled onto blank mask.
    #   2) Invert to achieve ramp=0, flow=255.
    binary_mask = None  # placeholder until contour extracted

    # --- Find the largest contour (assumed to be the wedge) ---
    contours, _ = cv2.findContours(binary_mask_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No contours found in the image.")
    wedge_contour = max(contours, key=cv2.contourArea)
    # Build crisp mask
    h, w = binary_mask_raw.shape
    crisp = np.full((h, w), 255, dtype=np.uint8)  # start all white (flow)
    cv2.drawContours(crisp, [wedge_contour], -1, color=0, thickness=-1)  # filled ramp black
    binary_mask = crisp

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
    # Backward compatible: if no width provided, assume square domain.
    if physical_domain_width is None:
        physical_domain_width = physical_domain_height

    scale_y = physical_domain_height / pixel_height_y
    scale_x = physical_domain_width / pixel_height_x
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
        "binary_mask": binary_mask,  # 0 = ramp, 255 = flow
        "scale_x": scale_x,
        "scale_y": scale_y,
        "physical_domain_height": physical_domain_height,
        "physical_domain_width": physical_domain_width,
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

    # Optional saving of debug images
    if save_debug_dir and (save_binary_mask or save_contour_overlay):
        if not save_basename:
            raise ValueError("save_basename must be provided when saving debug images")
        os.makedirs(save_debug_dir, exist_ok=True)
        # Determine aspect scaling for saving so that pixel width reflects physical width ratio.
        # If physical width differs from height we rescale horizontally.
        aspect_scale = 1.0
        if physical_domain_width is not None and physical_domain_height is not None and physical_domain_height > 0:
            aspect_scale = physical_domain_width / physical_domain_height

        def _rescale_width(img):
            if aspect_scale == 1.0:
                return img
            new_w = max(1, int(round(img.shape[1] * aspect_scale)))
            return cv2.resize(img, (new_w, img.shape[0]), interpolation=cv2.INTER_NEAREST)

        if save_binary_mask:
            mask_path = os.path.join(save_debug_dir, f"{save_basename}_binary_mask.png")
            cv2.imwrite(mask_path, _rescale_width(binary_mask))
        if save_contour_overlay:
            # Plain flow field image (no points / lines) scaled to aspect
            base_field = image_array
            # Convert to BGR for writing
            plain = cv2.cvtColor(base_field, cv2.COLOR_RGB2BGR)
            plain_path = os.path.join(save_debug_dir, f"{save_basename}_contour.png")
            cv2.imwrite(plain_path, _rescale_width(plain))
    return scaled_points, debug


def _cli():  # pragma: no cover - lightweight manual utility
    import argparse
    p = argparse.ArgumentParser(description="Extract and scale wedge geometry points from image/npz.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="Path to ordinary image (png/jpg)")
    src.add_argument("--npz", help="Path to .npz file to render")
    p.add_argument("--key", help="Data key inside .npz (when using --npz)")
    p.add_argument("--height", type=float, default=1.0, help="Physical domain height")
    p.add_argument("--width", type=float, default=None, help="Physical domain width (if rectangular)")
    p.add_argument("--debug", action="store_true", help="Return debug info and save when output paths given")
    p.add_argument("--out-dir", help="Directory for debug image outputs")
    p.add_argument("--basename", help="Basename for saved debug images (no extension)")
    p.add_argument("--save-binary", action="store_true", help="Save binary mask PNG if debug dir provided")
    p.add_argument("--save-overlay", action="store_true", help="Save contour overlay PNG if debug dir provided")
    p.add_argument("--json", help="Write points (and debug) to JSON path")
    args = p.parse_args()

    if args.npz and not args.key:
        p.error("--key is required when using --npz")

    if args.image:
        result = process_image_file(
            args.image,
            physical_domain_height=args.height,
            physical_domain_width=args.width,
            return_debug=args.debug,
            save_debug_dir=args.out_dir,
            save_basename=args.basename,
            save_binary_mask=args.save_binary,
            save_contour_overlay=args.save_overlay,
            json_output=args.json,
        )
    else:  # npz path
        rgb = extract_image_from_array(args.npz, args.key, return_raw=False)
        result = process_image_from_array(
            rgb,
            physical_domain_height=args.height,
            physical_domain_width=args.width,
            return_debug=args.debug,
            save_debug_dir=args.out_dir,
            save_basename=args.basename,
            save_binary_mask=args.save_binary,
            save_contour_overlay=args.save_overlay,
        )
        if args.json:
            if args.debug:
                pts, dbg = result
                payload = {"points": pts, "debug": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in dbg.items() if k != "image_rgb"}}
            else:
                payload = {"points": result}
            with open(args.json, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
    # Simple print summary for CLI usage
    if args.debug:
        pts = result[0]
    else:
        pts = result
    print("Scaled points (sorted):")
    for i, (x, y) in enumerate(pts, 1):
        print(f"  {i}: ({x:.6f}, {y:.6f})")


if __name__ == "__main__":  # pragma: no cover
    _cli()

