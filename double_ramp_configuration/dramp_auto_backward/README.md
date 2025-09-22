Visualization Feature
=====================

This configuration now supports a quick visualization to verify that the extracted corner / wall points correspond to the underlying geometry in the interpolated field arrays.

Fast Usage (no mesh generation required):

  python pipeline.py --visualize-points --extract-only --no-random

What it does:
1. Loads the first (or randomly sampled first) `.npz` case selected by filters.
2. Renders the chosen data key (currently `density`) into an RGB image.
3. Runs the point extraction; captures scaled points plus debug info.
4. Overlays the extracted points as red markers with indices on the image.
5. Saves:
   - Overlay PNG: `./double_ramp_configuration/outputs/backward/visualizations/first_case.png`
   - CSV of scaled points: `first_case_points.csv` in the same folder.

Flags:
--visualize-points   Enable first-case overlay generation.
--extract-only       Skip mesh & sweep steps (avoids gmsh dependency).
--no-random          Process all filtered files (visualization still only for the first processed one).
--sample-size N      Random sample size if random sampling active (default 1).

If gmsh is not installed you can still use visualization via --extract-only.

Examples:
  # Visualize first case, keep meshes off
  python pipeline.py --visualize-points --extract-only

  # Visualize with a random sample of 5 (points for first of the 5), no sweep
  python pipeline.py --visualize-points --sample-size 5 --skip-sweep

Troubleshooting:
- If the overlay shows points misaligned, verify `physical_height` passed to extraction.
- To debug thresholding logic, temporarily lower the threshold in `cv_processing.py`.

Rectangular Physical Scaling (New Option)
----------------------------------------
By default the pipeline assumed a square physical domain (width == height). A new optional
switch allows you to map pixel coordinates into a rectangular physical space, e.g. width = 0.62
and height = 0.40 (your updated physical geometry dimensions).

How to enable:
1. Open `main.py` in this folder.
2. Set `USE_RECTANGULAR_PHYSICAL_SCALE = True` near the top.
3. Adjust `RECT_PHYSICAL_WIDTH` and `RECT_PHYSICAL_HEIGHT` if needed (defaults: 0.62, 0.40).

What happens internally:
* The extraction step now passes both `physical_height` and `physical_width`.
* If `physical_width` is provided, x coordinates scale with width while y scale with height.
* Downstream mesh generation receives the already-scaled point set; no other changes required.

Backward compatibility:
* Leaving the flag False preserves legacy square scaling behavior exactly.
* No existing functions were modified destructively; a new optional argument `physical_width` was
  added (default None) to `process_image_from_array` and `extract_points_batch`.

Verification tip:
After toggling, regenerate a few point sets and open one of the visualization overlays—the aspect
ratio should now reflect the rectangular scaling when you interpret the axes numerically (e.g., x
limits extend to ~0.62 while y to ~0.40).

Aggressive 2nd Point (Ramp Start) Detection
-------------------------------------------
The extraction pipeline now includes an aggressive strategy to precisely capture the second geometric point: the end of the initial flat (plateau) before the first ramp begins. This point is critical for correctly defining the ramp geometry.

How it works:
1. Scans the leftmost band (configurable width) of the wedge contour to estimate the plateau y (median for robustness).
2. Walks points in increasing x; the last point whose y remains within a small tolerance of the plateau is tracked.
3. The first point whose y drops (in image coordinates this is an increase) beyond a configured threshold signals the ramp start; the x of the last plateau point is used as the 2nd point.
4. The y of the 2nd point is forcibly set equal to the 1st point's y to ensure a perfectly horizontal initial segment.

Parameters (edit near top of `cv_processing.py`):
  AGGR_ENABLE            Master toggle (default: True)
  AGGR_LEFT_BAND_WIDTH   Width in pixels used to characterize plateau (default: 12)
  AGGR_PLATEAU_Y_TOL     Tolerance for remaining "on plateau" (default: 2 px)
  AGGR_RAMP_DROP_THRESH  Minimum y drop (image coords) to accept ramp onset (default: 3 px)
  AGGR_INTERPOLATE       Optional sub-pixel interpolation between last plateau and first drop (default: False)

Debugging:
If `process_image_from_array(..., return_debug=True)` is used, the debug dict now includes:
  aggr_second_point_pixel : (x, y) pixel coords detected (or None)
  aggr_params             : The active parameter values

Adjustment Tips:
- If the 2nd point is too far left (early), decrease AGGR_RAMP_DROP_THRESH.
- If it is too far right (late), increase AGGR_PLATEAU_Y_TOL (allows more y drift to still count as plateau) or decrease the drop threshold.
- If noisy early fluctuations trigger a false ramp, reduce AGGR_PLATEAU_Y_TOL or increase AGGR_RAMP_DROP_THRESH slightly.

Advanced Profile-Based Detection (Hybrid)
----------------------------------------
In some cases the contour approximation can skip subtle early descent pixels. A secondary profile analysis samples the top boundary column-by-column, smooths it, and looks for a sustained positive slope (downward physical ramp) to robustly locate the ramp onset. Parameters (in `cv_processing.py`):

  AGGR_PROFILE_USE             Enable hybrid profile method (default True)
  AGGR_PROFILE_SAMPLE_STEP     X sampling stride in pixels (1 = every column)
  AGGR_PROFILE_SMOOTH_WINDOW   Moving average window (odd). Set 1 to disable smoothing
  AGGR_PROFILE_MIN_SLOPE       Minimum dy/dx (pixels per pixel) to treat as ramp onset
  AGGR_PROFILE_CONFIRM_SPAN    Required consecutive slope samples above threshold
  AGGR_PROFILE_ALLOW_EARLIEST  True: choose earliest qualifying onset; False: last plateau
  AGGR_PROFILE_MAX_SEARCH_X    Limit (absolute) search window from left to avoid second ramp interference
  AGGR_PROFILE_PLATEAU_RELOCK_Y Re-compute plateau y from smoothed profile for cleaner horizontal alignment

Tuning strategy:
  Missed early ramp -> Lower AGGR_PROFILE_MIN_SLOPE or AGGR_PROFILE_CONFIRM_SPAN.
  False early trigger -> Increase MIN_SLOPE or CONFIRM_SPAN; maybe widen smoothing window.
  Over-smoothing -> Reduce SMOOTH_WINDOW.

If profile detection does not produce a candidate, the algorithm gracefully falls back to the raw point drop logic so all cases remain supported.

Fallback Behavior:
If detection fails (rare), the previous heuristic ordering remains in effect; no exception is raised.

This enhancement ensures higher consistency in mesh generation and downstream CFD setup where the first break point is critical.
