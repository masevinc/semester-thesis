"""Utility script to inspect and plot a field from a .npz file.

Fixes MacOSX backend ValueError: "output array must be a NumPy array" by
forcing the non-interactive 'Agg' backend (issue observed with Python 3.12 +
matplotlib 3.7.x on macOS).

Simplified version without CLI control (as requested). Adjust parameters in
the CONFIG block below directly in the script.
"""

import os
import sys
import numpy as np

# Force a stable, non-interactive backend before importing pyplot
import matplotlib
matplotlib.use("Agg")  # Avoid MacOSX backend bug / headless safe
import matplotlib.pyplot as plt

# -------------------------------------------------
# User-editable configuration block
# -------------------------------------------------
CONFIG = {
    "NPZ_PATH": "./double_ramp_configuration/inputs/denorm/DDPM_fully/double_ramp_0.0_0.0_ma_3.298_pres_109837_interpolated_arrays.npz",
    "KEY": "density",              # Array key inside NPZ
    "OUT_PATH": "./double_ramp_configuration/X_dummy_code_playground/png_input/density_plot.png",
    "DPI": 100,                     # Output DPI
    "CMAP": "viridis",             # Matplotlib colormap name
    "NORMALIZE": True,              # Normalize to 0..1 if outside range
}

# data = np.load('./double_ramp_configuration/double_ramp_npz_files_clamped/double_ramp_0.049_0.0636_ma_2.511_pres_193365_interpolated_arrays.npz')
# # print(data.files)

# for key in data.files:
#     array = data[key]
#     print(f"\nKey: {key}")
#     print(f"Shape: {array.shape}")
#     print(f"Data type: {array.dtype}")
#     print(f"First few elements:\n{array[:5]}")

# density = data['density']

# plt.imshow(density, cmap='viridis')  # or 'gray' for traditional grayscale - origin='lower' -
# # plt.colorbar(label='Density')
# # plt.title('Density Map')
# # plt.xlabel('X')
# # plt.ylabel('Y')
# plt.axis('off')  # Hide axes
# plt.savefig('./double_ramp_configuration/cv_playground/density_plot.png', dpi=100, bbox_inches='tight', pad_inches=0)  # Save before showing 
# plt.show()
def load_array(npz_path: str, key: str) -> np.ndarray:
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    with np.load(npz_path) as data:
        if key not in data.files:
            raise KeyError(f"Key '{key}' not in file. Available: {data.files}")
        arr = data[key]
    # Ensure ndarray, float64 contiguous
    arr = np.ascontiguousarray(arr)
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float64)
    return arr


def maybe_normalize(arr: np.ndarray) -> np.ndarray:
    a_min = float(np.nanmin(arr))
    a_max = float(np.nanmax(arr))
    if a_max == a_min:
        return arr  # constant field
    if a_min < 0.0 or a_max > 1.0:
        # Normalize to 0..1 range for visualization
        return (arr - a_min) / (a_max - a_min)
    return arr


def main():
    npz_path = CONFIG["NPZ_PATH"]
    key = CONFIG["KEY"]
    out_path = CONFIG["OUT_PATH"]
    dpi = CONFIG["DPI"]
    cmap = CONFIG["CMAP"]
    normalize_flag = CONFIG["NORMALIZE"]

    arr = load_array(npz_path, key)
    original_min, original_max = float(np.nanmin(arr)), float(np.nanmax(arr))
    if normalize_flag:
        arr_vis = maybe_normalize(arr)
    else:
        arr_vis = arr

    h, w = arr_vis.shape[:2]
    figsize = (w / dpi, h / dpi)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(arr_vis, cmap=cmap, aspect='auto')

    # Optional colorbar (commented out to keep pure image)
    # fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Saved image: {out_path}")
    print(f"Array shape: {arr.shape}, dtype: {arr.dtype}, range: [{original_min:.4g}, {original_max:.4g}]")
    if arr_vis is not arr:
        print("(Normalized for visualization)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

