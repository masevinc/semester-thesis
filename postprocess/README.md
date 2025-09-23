# VTU Post-processing

A small utility to extract and plot fields from SU2 `.vtu` outputs.

What it does:
- Lists available point and cell data arrays in a `.vtu` file
- Extracts a selected field (e.g., density / rho, Temperature, Pressure)
- Saves a CSV with columns: x, y, value
- Produces quick plots: histogram and 2D field plot

## Quick start

1) Pick a VTU file, e.g.:
   `single_ramp_configuration/youtube_case/ramp_analysis_su2v8_v1/flow.vtu`

2) Run listing mode (optional):

```bash
python postprocess/vtu_postprocess.py --vtu single_ramp_configuration/youtube_case/ramp_analysis_su2v8_v1/flow.vtu --list-fields --out postprocess_outputs/listing
```

3) Extract and plot density (robust to common names like `rho`, `Density`):

```bash
python postprocess/vtu_postprocess.py --vtu single_ramp_configuration/youtube_case/ramp_analysis_su2v8_v1/flow.vtu --field density --out postprocess_outputs/density
```

Outputs are written to the `--out` directory:
- `available_fields.txt`
- `<chosen_key>_<location>.csv` with x, y, value
- `<chosen_key>_<location>_field.png` and `_hist.png`
- `stats.txt`

Notes:
- The script uses `meshio`, `numpy`, and `matplotlib`.
- For nicer 2D plots, point data will use an internal triangulation; cell data falls back to scatter.
- If your field is only present in cell data, the CSV will contain cell centers.

## NPZ Post-processing

Use `postprocess/npz_postprocess.py` to list keys in a `.npz` dataset and export a selected 2D field.

Examples:

```bash
# List keys and write available_keys.txt
python postprocess/npz_postprocess.py --npz double_ramp_configuration/inputs/double_ramp_npz_files_clamped/your_file.npz --list-keys --out postprocess_outputs/npz_listing

# Export and plot a field (uses x/y if present; else uniform grid)
python postprocess/npz_postprocess.py --npz double_ramp_configuration/inputs/double_ramp_npz_files_clamped/your_file.npz --field temperature --out postprocess_outputs/npz_temperature

# Or run with defaults (edit constants at the top of the script)
python postprocess/npz_postprocess.py

# Vector fields (e.g., velocity stored as (ny,nx,3))
python postprocess/npz_postprocess.py --npz your_vector_file.npz --field velocity --direction x --out postprocess_outputs/npz_velocity_x
python postprocess/npz_postprocess.py --npz your_vector_file.npz --field velocity --direction mag --out postprocess_outputs/npz_velocity_mag
```

Outputs:
- `available_keys.txt`
- `<field>.csv` with x, y, value (flattened grid)
- `<field>_field.png` and `<field>_hist.png`
- `stats.txt`

## VTU vs NPZ comparison

Use `postprocess/compare_vtu_npz.py` to interpolate a VTU field onto an NPZ grid, plot NPZ | VTU→grid | Diff, compute MSE, and run in batch across a sweep folder.

What it does
- Loads VTU (point or cell field) robustly (meshio with VTK fallback).
- Builds inside-domain mask from VTU connectivity (triangles/quads only); outside grid values set to a chosen constant (default 0).
- Aligns NPZ grid to VTU bounds (auto) or uses NPZ coordinates directly.
- Uses the same color scale for NPZ and VTU plots; Diff uses a symmetric diverging scale.
- Single run or batch mode over case folders; batch writes a summary CSV with MSE and a FAILED CASES section.
- Crash detection: any case folder without `flow.vtu` is marked as crashed and reported.

Key flags
- `--mode auto|single|batch`: choose single run, batch over roots, or auto-detect (default from DEFAULT_MODE).
- `--vtu`, `--vtu-field`: single-case VTU and field (e.g., Density, Pressure).
- `--npz`, `--npz-field`: single-case NPZ and field key (e.g., density, temperature).
- `--mapping auto|npz|vtu`: NPZ grid mapping; auto uses VTU bounds when NPZ coords are missing/normalized.
- `--npz-y-origin top|bottom`: image-style (top) vs math-style (bottom) orientation; ‘top’ mirrors your visualization pipeline.
- `--outside-value <float>`: value assigned outside the VTU mesh domain on the rectangular grid (default 0; use `nan` to exclude from color scaling).
- `--no-clean`: by default outputs are cleaned before run; use this to keep previous outputs.
- Batch-only: `--vtu-root`, `--npz-root`, `--out-root`, `--summary-csv`.

Defaults
- Edit the constants at the top of the script: `DEFAULT_VTU`, `DEFAULT_NPZ`, `DEFAULT_VTU_FIELD`, `DEFAULT_NPZ_FIELD`, `DEFAULT_VTU_ROOT`, `DEFAULT_NPZ_ROOT`, `DEFAULT_MODE`, `DEFAULT_CLEAN`, etc.

Outputs (single case)
- `<out>/npz.png`, `<out>/vtu_on_grid.png`, `<out>/diff.png`.
- `<out>/comparison_subplot.png` (NPZ | VTU→grid | Diff) with colorbars.
- `<out>/report.txt` containing the MSE.

Outputs (batch)
- Per-case folders under `<out-root>/<case_key>/` with the same images and `report.txt`.
- Summary CSV at `--summary-csv` (default `postprocess_outputs/compare_batch_summary.csv`) with columns:
   - `case_key, vtu_path, npz_path, npz_field, vtu_field, mapping, npz_y_origin, outside_value, mse`.
   - A `SUMMARY` section listing `total_case_folders`, `successes`, `failures`.
   - A `FAILED CASES` section with `case_key, vtu_path, npz_path, reason`.
      - Crashed cases: `reason = "crashed: no flow.vtu in case folder"`.
      - No NPZ match: `reason = "no NPZ match for key '<key>'"`.

Examples

Single-case run with explicit files:
```bash
python postprocess/compare_vtu_npz.py \
   --mode single \
   --vtu single_ramp_configuration/youtube_case/ramp_analysis_su2v8_v1/flow.vtu \
   --vtu-field Density \
   --npz double_ramp_configuration/inputs/double_ramp_npz_files_clamped/double_ramp_0.011_0.0488_ma_2.892_pres_199070_interpolated_arrays.npz \
   --npz-field density \
   --out postprocess_outputs/compare_example
```

Batch across a sweep (uses defaults for roots if not provided):
```bash
python postprocess/compare_vtu_npz.py --mode batch \
   --vtu-root double_ramp_configuration/outputs/backward/sweep \
   --npz-root double_ramp_configuration/inputs/double_ramp_npz_files_clamped \
   --out-root postprocess_outputs/compare_batch \
   --summary-csv postprocess_outputs/compare_batch_summary.csv
```

Notes
- Warnings like “VTU file corrupt… Velocity … doesn’t fit components 3” come from meshio when vector arrays are inconsistent. We skip those arrays; density/pressure extraction still works.
- To keep outside-domain areas from influencing the color scale, run with `--outside-value nan`.
- Case-key pairing normalizes names (e.g., converts `0p011` → `0.011` and trims `_interpolated_arrays` suffix) to match NPZ files.

### NEW: Vertical Line Density (or Field) Profile Extraction

You can now extract a vertical line (constant y) profile from the comparison grid for both the NPZ and the VTU-interpolated field.

Flags:
```bash
   --line-y-index <int>    # Row index (0 = lowest y after orientation handling)
   --line-y-coord <float>  # Physical y coordinate (linear interpolation between rows)
   --line-name <str>       # Base name for output files (default: vertical_profile)
```

Usage examples:
```bash
# Using a physical coordinate (preferred when you know the geometry y)
python postprocess/main_compare_vtu_npz.py \
   --mode single \
   --vtu <path>/flow.vtu \
   --npz <path>/your_arrays.npz \
   --vtu-field Density \
   --npz-field density \
   --line-y-coord 0.015   \
   --line-name density_y0015 \
   --out postprocess_outputs/compare_single

# Using a row index
python postprocess/main_compare_vtu_npz.py --mode single --line-y-index 10 --line-name test_line
```

Outputs (single case or per batch case directory):
- `<line-name>.csv` with columns:
   - `x, npz_value, vtu_value, y_used, idx_used`
   - `y_used` is the interpolated physical y actually sampled.
   - `idx_used` is the lower row index used (for interpolation reference).
- `<line-name>.png` line plot comparing NPZ vs VTU→grid along x.

Behavior:
- If both `--line-y-index` and `--line-y-coord` are supplied, the coordinate takes precedence.
- Coordinate-based extraction performs linear interpolation between the two bracketing rows (both for NPZ and VTU-on-grid arrays) to ensure aligned sampling.
- If the requested coordinate lies outside the grid bounds, an error file `<line-name>_error.txt` is written instead.

Tips:
- Use `--npz-y-origin top` (default) or `bottom` consistently when interpreting indices. Changing origin flips the vertical orientation before extraction.
- For multiple profiles, you can run the script repeatedly or extend it to accept a comma-separated list (future enhancement).