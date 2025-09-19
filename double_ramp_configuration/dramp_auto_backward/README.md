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
