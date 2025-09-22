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
from datetime import datetime
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
    physical_width=None,
    clear_output_before_run=True,
    # Sampling controls
    enable_random_sampling=False,
    sample_size=None,
    random_seed=None,
    # Manifest controls
    manifest_filename="selection_manifest.csv",
    manifest_in_parent=False
):
    """
    Processes .npz files in data_dir, extracts points, and saves as .npy files.

    Switching sampling:
      - Set enable_random_sampling=True and give sample_size to activate random subset.
      - Leave enable_random_sampling=False (or sample_size=None) to process ALL filtered files.
      - Backward compatibility: if sample_size is provided but enable_random_sampling=False, sampling WILL still occur (legacy behavior) so existing code keeps working. To strictly disable sampling set sample_size=None.
    """
    if clear_output_before_run:
        clear_output_directory(output_dir)
    else:
        os.makedirs(output_dir, exist_ok=True)

    # --- Gather and filter candidate files first (no processing yet) ---
    candidate_files = []
    metadata_list = []
    for fname in os.listdir(data_dir):
        if not fname.endswith('.npz'):
            continue
        meta = extract_metadata(fname)
        if not meta or not matches_filters(meta, filters):
            continue
        candidate_files.append(fname)
        metadata_list.append(meta)

    total_matches = len(candidate_files)
    if total_matches == 0:
        print("No matching .npz files found after applying filters.")
        return

    # Decide whether to sample (backward compatible logic)
    do_sampling = (enable_random_sampling and sample_size is not None) or (not enable_random_sampling and sample_size not in (None, 0))

    if do_sampling:
        if sample_size <= 0:
            print("sample_size <= 0; nothing to process.")
            return
        rng = np.random.default_rng(random_seed) if random_seed is not None else np.random.default_rng()
        if sample_size < total_matches:
            indices = rng.choice(total_matches, size=sample_size, replace=False)
            selected_files = [candidate_files[i] for i in indices]
            selected_meta = [metadata_list[i] for i in indices]
            print(f"Randomly selected {len(selected_files)} of {total_matches} matching cases.")
        else:
            selected_files = candidate_files
            selected_meta = metadata_list
            print(f"Requested sample_size >= total matches ({total_matches}); processing all.")
    else:
        selected_files = candidate_files
        selected_meta = metadata_list
        print(f"Sampling disabled. Processing all {total_matches} matching cases.")

    # --- Determine manifest directory (normalize path first) ---
    norm_out_dir = os.path.abspath(output_dir.rstrip(os.sep))
    if manifest_in_parent:
        manifest_dir = os.path.dirname(norm_out_dir)
    else:
        manifest_dir = norm_out_dir
    os.makedirs(manifest_dir, exist_ok=True)

    # --- Write manifest for reproducibility ---
    try:
        manifest_path = os.path.join(manifest_dir, manifest_filename)
        with open(manifest_path, 'w') as mf:
            mf.write("# Selection manifest generated at UTC: " + datetime.utcnow().isoformat() + "\n")
            mf.write(f"# enable_random_sampling={enable_random_sampling}, sample_size={sample_size}, random_seed={random_seed}, total_matches={total_matches}, final_count={len(selected_files)}\n")
            mf.write("# Columns: filename,ramp1,ramp2,ma,pres\n")
            for fname, meta in zip(selected_files, selected_meta):
                mf.write(f"{fname},{meta.get('ramp1')},{meta.get('ramp2')},{meta.get('ma')},{meta.get('pres')}\n")
        print(f"Selection manifest written: {manifest_path}")
    except Exception as e:
        print(f"Failed to write manifest: {e}")

    # --- Process the selected files ---
    for fname in selected_files:
        for key in selected_keys:
            try:
                npz_path = os.path.join(data_dir, fname)
                image_np = extract_image_from_array(npz_path, data_key=key)
                points = process_image_from_array(
                    image_np,
                    physical_domain_height=physical_height,
                    physical_domain_width=physical_width
                )
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
    ENABLE_RANDOM_SAMPLING = True
    SAMPLE_SIZE = 100
    RANDOM_SEED = 42

    extract_points_batch(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        filters=FILTERS,
        selected_keys=SELECTED_KEYS,
    physical_height=PHYSICAL_HEIGHT,
    physical_width=None,
        clear_output_before_run=CLEAR_OUTPUT_BEFORE_RUN,
        enable_random_sampling=ENABLE_RANDOM_SAMPLING,
        sample_size=SAMPLE_SIZE,
        random_seed=RANDOM_SEED,
        manifest_in_parent=True
    )
