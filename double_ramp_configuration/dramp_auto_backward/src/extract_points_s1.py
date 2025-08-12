"""

Step 1

extract_points.py

Batch processes .npz files in a directory, extracting geometric points from selected data arrays,
and saves the results as .npy files for further analysis.

All configuration is set via function arguments.
"""

import os
import numpy as np
import shutil
from src.cv_processing import (
    extract_metadata,
    matches_filters,
    extract_image_from_array,
    process_image_from_array
)

def clear_output_directory(directory):
    """
    Removes all files in the specified directory. Creates the directory if it does not exist.
    """
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")
    else:
        os.makedirs(directory, exist_ok=True)

def extract_points_batch(
    data_dir,
    output_dir,
    filters,
    selected_keys,
    physical_height,
    clear_output_before_run=True
):
    """
    Processes all .npz files in data_dir, extracts points, and saves as .npy files.
    """
    if clear_output_before_run:
        clear_output_directory(output_dir)
    else:
        os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(data_dir):
        if not fname.endswith('.npz'):
            continue

        meta = extract_metadata(fname)
        if not meta or not matches_filters(meta, filters):
            continue

        for key in selected_keys:
            try:
                npz_path = os.path.join(data_dir, fname)
                image_np = extract_image_from_array(npz_path, data_key=key)
                points = process_image_from_array(image_np, physical_domain_height=physical_height)

                name_no_ext = os.path.splitext(fname)[0]
                out_path = os.path.join(output_dir, f"{name_no_ext}_{key}.npy")
                np.save(out_path, np.array(points))
                print(f"Saved points to: {out_path}")

            except Exception as e:
                print(f"Failed processing {fname} [{key}]: {e}")

# CLI usage for backwards compatibility
if __name__ == "__main__":
    # Default configuration (can be overridden by importing and calling extract_points_batch)
    DATA_DIR = './double_ramp_configuration/inputs/double_ramp_npz_files_clamped'
    OUTPUT_DIR = './double_ramp_configuration/outputs/backward/extracted_points'
    FILTERS = {
        "ramp1": 0.046,
        "ramp2": None,
        "min_ma": None,
        "max_ma": None
    }
    SELECTED_KEYS = ['density']
    PHYSICAL_HEIGHT = 256
    CLEAR_OUTPUT_BEFORE_RUN = True

    extract_points_batch(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        filters=FILTERS,
        selected_keys=SELECTED_KEYS,
        physical_height=PHYSICAL_HEIGHT,
        clear_output_before_run=CLEAR_OUTPUT_BEFORE_RUN
    )
