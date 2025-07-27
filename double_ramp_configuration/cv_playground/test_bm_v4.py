import re
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2
from point_reorder_gmsh import sort_points  # Make sure this is available in your path

'''
Different from the v3:
- Filters added for specific .npz files
- Now instead of using the .png files, we are using the .npz files
'''


# === USER CONFIGURATION ===
DATA_DIR = './double_ramp_npz_files_clamped' 
PHYSICAL_HEIGHT = 1.0
SAVE_STEPS = False
OUTPUT_DIR = './pressure_data_from_rim/dummy_results'

# Filter files by metadata extracted from filenames
FILTERS = {
    "ramp1": 0.01,      # Use None to ignore
    "ramp2": 0.0304,    # Use None to ignore
    "min_ma": None,    # Minimum Mach number
    "max_ma": None      # Optional max Mach number
}

# Data keys to extract from .npz files
SELECTED_KEYS = ['density']  # Choose one or more: 'density', 'pressure', etc.


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


def extract_image_from_array(npz_path, data_key):
    data = np.load(npz_path)

    if data_key not in data:
        raise KeyError(f"'{data_key}' not found in {npz_path}. Available keys: {list(data.keys())}")

    array = data[data_key]

    fig, ax = plt.subplots()
    ax.imshow(array, cmap='gray')
    ax.axis('off')

    canvas = FigureCanvas(fig)
    canvas.draw()
    image_np = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image_np = image_np.reshape(canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return image_np


def process_image_from_array(image_array, physical_domain_height=1.0, save_steps=False, output_dir=None):
    if save_steps:
        assert output_dir is not None, "Output directory must be specified if save_steps is True."
        os.makedirs(output_dir, exist_ok=True)

    def save_step(img, step_name):
        if save_steps:
            path = os.path.join(output_dir, f"{step_name}.png")
            cv2.imwrite(path, img)

    image = image_array.copy()
    save_step(image, "step_0_original")

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    median_v2 = cv2.medianBlur(gray, 11)
    blurred_v7 = cv2.GaussianBlur(median_v2, (9, 9), 2)
    _, binary_mask = cv2.threshold(blurred_v7, 20, 255, cv2.THRESH_BINARY_INV)
    save_step(binary_mask, "step_1_binary")

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
        scaled_points.append((x_phys, y_phys))

    scaled_points = sort_points(scaled_points)
    return scaled_points


def process_all_matching_files():
    for fname in os.listdir(DATA_DIR):
        if not fname.endswith(".npz"):
            continue

        metadata = extract_metadata(fname)
        if metadata is None or not matches_filters(metadata, FILTERS):
            continue

        full_path = os.path.join(DATA_DIR, fname)
        print(f"\nProcessing: {fname} | Metadata: {metadata}")

        for key in SELECTED_KEYS:
            try:
                print(f"  Processing key: {key}")
                image_np = extract_image_from_array(full_path, data_key=key)
                key_output_dir = os.path.join(OUTPUT_DIR, f"{os.path.splitext(fname)[0]}/{key}")

                points = process_image_from_array(
                    image_array=image_np,
                    physical_domain_height=PHYSICAL_HEIGHT,
                    save_steps=SAVE_STEPS,
                    output_dir=key_output_dir
                )

                print(f"    Extracted {len(points)} points from '{key}':")
                for pt in points:
                    print(f"      {pt}")

            except KeyError:
                print(f"    Key '{key}' not found in {fname}. Skipping.")
            except Exception as e:
                print(f"    Failed to process '{key}' from {fname}: {e}")


if __name__ == "__main__":
    process_all_matching_files()
