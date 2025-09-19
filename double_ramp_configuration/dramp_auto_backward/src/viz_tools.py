"""
viz_tools.py

Lightweight visualization utilities for inspecting the first geometry
extracted in the pipeline. Produces a PNG overlay of the original
rendered field (RGB image generated from the selected array) together
with the extracted (scaled) corner / wall points.

This is intended as a quick evaluation aid to confirm that the corner
extraction logic is returning points that correspond to the geometry
features in the underlying array.
"""

from __future__ import annotations

import os
from typing import Sequence, Tuple, Dict, Any
import numpy as np
import matplotlib.pyplot as plt


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_geometry_with_points(
    debug: Dict[str, Any],
    scaled_points: Sequence[Tuple[float, float]],
    output_dir: str,
    output_name: str = "geometry_points_overlay",
    title: str | None = None,
    dpi: int = 150,
):
    """Create and save an overlay plot for the first processed geometry.

    Parameters
    ----------
    debug : dict
        Debug dictionary produced by process_image_from_array(return_debug=True)
        Expected keys: 'image_rgb', 'scale_x', 'scale_y', 'left_lower', 'image_height'
    scaled_points : Sequence[(float, float)]
        The scaled (physical) points returned by the extraction logic.
    output_dir : str
        Directory where the figure will be stored.
    output_name : str, default 'geometry_points_overlay'
        Base name (without extension) for the saved PNG.
    title : str, optional
        Title for the plot.
    dpi : int, default 150
        Resolution for the saved figure.
    """
    if not debug:
        raise ValueError("Debug information is empty; cannot produce visualization.")

    image = debug.get("image_rgb")
    if image is None:
        raise KeyError("debug dict missing 'image_rgb'")

    scale_x = debug.get("scale_x")
    scale_y = debug.get("scale_y")
    left_lower = debug.get("left_lower")  # pixel coords (x, y)
    image_height = debug.get("image_height")

    if None in (scale_x, scale_y, left_lower, image_height):
        raise KeyError("debug dict missing one of required keys: scale_x, scale_y, left_lower, image_height")

    # Convert scaled physical points back to pixel coordinates for overlay.
    # Mapping inverse of logic in cv_processing.process_image_from_array
    llx, lly = left_lower
    px_points = []
    for x_phys, y_phys in scaled_points:
        dx_px = x_phys / scale_x
        dy_px = y_phys / scale_y
        x_px = llx + dx_px
        y_px = lly - dy_px
        px_points.append((x_px, y_px))
    px_points = np.array(px_points)

    _ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(image.shape[1] / dpi, image.shape[0] / dpi), dpi=dpi)
    ax.imshow(image)
    if len(px_points):
        ax.scatter(px_points[:, 0], px_points[:, 1], c='red', s=30, marker='o', edgecolors='white', linewidths=0.5, label='Extracted Points')
        for i, (xp, yp) in enumerate(px_points):
            ax.text(xp + 3, yp + 3, str(i + 1), color='white', fontsize=6, ha='left', va='bottom')
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=8)
    if len(px_points):
        ax.legend(loc='lower right', fontsize=6, frameon=True)
    fig.tight_layout(pad=0)
    out_path = os.path.join(output_dir, f"{output_name}.png")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Visualization saved: {out_path}")

    # Also save a simple CSV of scaled points for quick reference next to image
    csv_path = os.path.join(output_dir, f"{output_name}_points.csv")
    try:
        np.savetxt(csv_path, np.asarray(scaled_points), delimiter=",", header="x_phys,y_phys", comments="")
    except Exception as e:
        print(f"Failed writing points CSV ({csv_path}): {e}")
