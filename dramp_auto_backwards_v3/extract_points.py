# extract_points.py

import os
import numpy as np
from cv_processing import (
    extract_metadata,
    matches_filters,
    extract_image_from_array,
    process_image_from_array
)

# === CONFIGURATION ===
DATA_DIR = './double_ramp_configuration/double_ramp_npz_files_clamped' 
OUTPUT_DIR = './dramp_auto_backwards_v3/dummy_results' 
FILTERS = {
    "ramp1": None, #0.046
    "ramp2": None, #0.0112
    "min_ma": None,
    "max_ma": None
}
SELECTED_KEYS = ['density']
PHYSICAL_HEIGHT = 256

os.makedirs(OUTPUT_DIR, exist_ok=True)

for fname in os.listdir(DATA_DIR):
    if not fname.endswith('.npz'):
        continue

    meta = extract_metadata(fname)
    if not meta or not matches_filters(meta, FILTERS):
        continue

    for key in SELECTED_KEYS:
        try:
            npz_path = os.path.join(DATA_DIR, fname)
            image_np = extract_image_from_array(npz_path, data_key=key)
            points = process_image_from_array(image_np, physical_domain_height=PHYSICAL_HEIGHT)

            name_no_ext = os.path.splitext(fname)[0]
            out_path = os.path.join(OUTPUT_DIR, f"{name_no_ext}_{key}.npy")
            np.save(out_path, np.array(points))
            print(f"Saved points to: {out_path}")

        except Exception as e:
            print(f"Failed processing {fname} [{key}]: {e}")
