"""
evaluation_viz.py
---------------------------------
Lightweight, optional evaluation visualization.

Purpose:
  After running the extraction step, produce a single figure that helps
  verify that the extracted (scaled) corner/wall points correspond to
  the underlying geometry in the dataset. It simply renders the raw
  scalar field (the array inside the .npz file) as an image background
  and overlays the extracted points.

Design choices / minimal intrusion:
  * Does NOT modify existing extraction logic.
  * Does NOT change CLI / argparse usage (invoked manually from main.py).
  * Works purely from already written outputs (.npy point files) and
    the original .npz source files.
  * Uses matplotlib only (already a dependency in the project).

Usage (from main.py after extraction):
  from src.evaluation_viz import visualize_first_extracted_case
  visualize_first_extracted_case(data_dir, points_dir, eval_output_dir, physical_height)

Output:
  A PNG (and companion CSV copy of the points) saved into the chosen
  evaluation output directory.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
import sys

# Ensure parent directory (containing src) is on path when imported standalone
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

try:
    from src.cv_processing import extract_metadata  # type: ignore
except ModuleNotFoundError:
    from cv_processing import extract_metadata  # type: ignore


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _infer_original_npz_filename(points_filename: str, data_key: str) -> str:
    """Given a saved points .npy filename and the data key, recover original .npz base name.

    Points files are saved as:  <original_npz_name_without_ext>_<data_key>.npy
    Example:  double_ramp_0.011_0.0488_ma_2.892_pres_199070_density.npy
              -> original npz: double_ramp_0.011_0.0488_ma_2.892_pres_199070.npz
    """
    if not points_filename.endswith('.npy'):
        raise ValueError(f"Expected .npy file, got: {points_filename}")
    stem = points_filename[:-4]  # drop .npy
    suffix = f"_{data_key}"
    if not stem.endswith(suffix):
        # Fall back: strip last underscore segment (more generic)
        orig = stem.rsplit('_', 1)[0]
    else:
        orig = stem[: -len(suffix)]
    return orig + '.npz'


def visualize_first_extracted_case(
    data_dir: str,
    points_dir: str,
    output_dir: str,
    physical_height: float,
    physical_width: float | None = None,
    data_key: str = 'density',
    figure_name: str = 'evaluation_overlay',
    cmap: str = 'viridis',
    flip_vertical: bool = True,
    margin_frac: float = 0.035,
    point_size: float = 45.0,
    save_background: bool = True,
    background_suffix: str = '_stage'
):
    """Generate a single evaluation visualization for the first available point file.

    Parameters
    ----------
    data_dir : str
        Directory containing the original source .npz files.
    points_dir : str
        Directory containing extracted points (.npy) produced by extraction step.
    output_dir : str
        Target directory where the visualization PNG and CSV copy will be stored.
    physical_height : float
        Physical height scaling (y-direction) used during extraction.
    physical_width : float | None, default None
        Physical width scaling (x-direction). If None assumes square domain (width == height).
    data_key : str, default 'density'
        The array key used during extraction (needed to reconstruct original filename).
    figure_name : str, default 'evaluation_overlay'
        Base name (without extension) for saved outputs.
    cmap : str, default 'viridis'
        Colormap for background field.
    """
    if not os.path.isdir(points_dir):
        print(f"[evaluation_viz] Points directory does not exist: {points_dir}")
        return

    point_files = sorted([f for f in os.listdir(points_dir) if f.endswith('.npy') and f.endswith(f'_{data_key}.npy')])
    if not point_files:
        print(f"[evaluation_viz] No point files matching '*_{data_key}.npy' in {points_dir}")
        return

    first_points_file = point_files[0]
    npz_name = _infer_original_npz_filename(first_points_file, data_key=data_key)
    npz_path = os.path.join(data_dir, npz_name)
    points_path = os.path.join(points_dir, first_points_file)

    if not os.path.isfile(npz_path):
        print(f"[evaluation_viz] Original npz not found: {npz_path}")
        return

    try:
        scaled_points = np.load(points_path)
    except Exception as e:
        print(f"[evaluation_viz] Failed loading points '{points_path}': {e}")
        return

    try:
        with np.load(npz_path) as data:
            if data_key not in data:
                print(f"[evaluation_viz] Key '{data_key}' not in {npz_path}. Available: {list(data.keys())}")
                return
            field = data[data_key]
    except Exception as e:
        print(f"[evaluation_viz] Failed loading npz '{npz_path}': {e}")
        return

    if field.ndim != 2:
        print(f"[evaluation_viz] Expected 2D array for '{data_key}', got shape {field.shape}")
        return

    # Optionally flip vertically (user reported orientation issue: "flip through x axis").
    if flip_vertical:
        field = np.flipud(field)

    # Determine width (square legacy behavior if not specified)
    if physical_width is None:
        physical_width = physical_height

    # Background extent reflects rectangular domain if provided.
    extent = [0.0, physical_width, 0.0, physical_height]

    _ensure_dir(output_dir)
    # Adjust figure aspect: keep ~6.2 units for the longer side for consistency
    base = 6.2
    aspect_ratio = physical_width / physical_height if physical_height > 0 else 1.0
    if aspect_ratio >= 1:  # wider than tall
        figsize = (base, base / aspect_ratio)
    else:  # taller than wide (unlikely here)
        figsize = (base * aspect_ratio, base)
    fig, ax = plt.subplots(figsize=figsize, dpi=120)
    im = ax.imshow(field, cmap=cmap, origin='lower', extent=extent, aspect='equal')
    # Colorbar can also trigger the same ABI ImportError; guard it.
    cb = None
    try:
        cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cb.ax.tick_params(labelsize=8)
    except ImportError as e:
        print(f"[evaluation_viz][warn] colorbar ImportError (continuing): {e}")
    except Exception as e:
        if os.environ.get('STRICT_VIZ_DEBUG'):
            raise
        print(f"[evaluation_viz][warn] colorbar unexpected error: {e.__class__.__name__}: {e}")

    # Add small margin to make edge points easier to see
    margin = margin_frac * max(physical_height, physical_width)
    ax.set_xlim(-margin, physical_width + margin)
    ax.set_ylim(-margin, physical_height + margin)
    ax.set_xlabel('x (physical)')
    ax.set_ylabel('y (physical)')
    # Concise title with ramps, Mach, pressure
    meta = extract_metadata(npz_name)
    if meta:
        r1 = meta.get('ramp1')
        r2 = meta.get('ramp2')
        ma = meta.get('ma')
        pres = meta.get('pres')
        if r2 is not None:
            title_str = f"r1={r1:.3f}  r2={r2:.3f}  M={ma:.3f}  p={pres}"
        else:
            title_str = f"r1={r1:.3f}  M={ma:.3f}  p={pres}"
    else:
        title_str = npz_name.replace('.npz','')
    ax.set_title(title_str, fontsize=10)

    # Optionally save a background-only image (no points/labels)
    if save_background:
        try:
            fig.tight_layout()
        except ImportError as e:
            print(f"[evaluation_viz][warn] tight_layout ImportError (background, continuing): {e}")
            try:
                fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.12, wspace=0.25, hspace=0.30)
            except Exception as e2:
                print(f"[evaluation_viz][warn] subplots_adjust fallback failed (background): {e2.__class__.__name__}: {e2}")
        except Exception as e:
            if os.environ.get('STRICT_VIZ_DEBUG'):
                raise
            print(f"[evaluation_viz][warn] tight_layout unexpected error (background): {e.__class__.__name__}: {e}")
            try:
                fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.12, wspace=0.25, hspace=0.30)
            except Exception as e2:
                print(f"[evaluation_viz][warn] subplots_adjust (unexpected path) failed (background): {e2.__class__.__name__}: {e2}")
        bg_png_path = os.path.join(output_dir, f"{figure_name}{background_suffix}.png")
        try:
            fig.savefig(bg_png_path, dpi=150)
        except ImportError as e:
            print(f"[evaluation_viz][warn] savefig ImportError (background, continuing): {e}")
        except Exception as e:
            if os.environ.get('STRICT_VIZ_DEBUG'):
                raise
            print(f"[evaluation_viz][warn] savefig unexpected error (background): {e.__class__.__name__}: {e}")
        else:
            print(f"[evaluation_viz] Saved background (no points): {bg_png_path}")

    # Now overlay points and labels
    if scaled_points.size > 0:
        pts = np.asarray(scaled_points)
        if pts.ndim == 2 and pts.shape[1] == 2:
            ax.scatter(pts[:, 0], pts[:, 1], c='red', s=point_size, edgecolors='white', linewidths=0.7, label='Extracted Points')
            for i, (x, y) in enumerate(pts):
                # offset scaled relative to smaller dimension for readability
                offset = 0.012 * min(physical_height, physical_width)
                ax.text(x + offset, y + offset, str(i + 1), color='white', fontsize=7, ha='left', va='bottom')
        else:
            print(f"[evaluation_viz] Unexpected points array shape: {pts.shape}")

    ax.legend(loc='upper right', fontsize=7, frameon=True)
    # --- Robust layout handling -------------------------------------------------
    # Some environments (e.g., mismatched Matplotlib/NumPy builds) trigger ImportError
    # deep inside tight_layout due to missing symbols (e.g., ERR_IGNORE). Guard it so
    # visualization still proceeds and warn the user.
    try:
        fig.tight_layout()
    except ImportError as e:
        print(f"[evaluation_viz][warn] tight_layout ImportError (continuing): {e}")
        try:
            fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.12, wspace=0.25, hspace=0.30)
        except Exception as e2:
            print(f"[evaluation_viz][warn] subplots_adjust fallback failed: {e2.__class__.__name__}: {e2}")
    except Exception as e:
        if os.environ.get('STRICT_VIZ_DEBUG'):
            raise
        print(f"[evaluation_viz][warn] tight_layout unexpected error: {e.__class__.__name__}: {e}")
        try:
            fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.12, wspace=0.25, hspace=0.30)
        except Exception as e2:
            print(f"[evaluation_viz][warn] subplots_adjust (unexpected path) failed: {e2.__class__.__name__}: {e2}")

    png_path = os.path.join(output_dir, f"{figure_name}.png")
    # Saving may still trigger the ImportError; guard it too.
    try:
        fig.savefig(png_path, dpi=150)
        print(f"[evaluation_viz] Saved evaluation figure: {png_path}")
    except ImportError as e:
        print(f"[evaluation_viz][warn] savefig ImportError (continuing without figure): {e}")
    except Exception as e:
        if os.environ.get('STRICT_VIZ_DEBUG'):
            raise
        print(f"[evaluation_viz][warn] savefig unexpected error: {e.__class__.__name__}: {e}")
    finally:
        plt.close(fig)

    # Also copy points (again) for convenient side-by-side reference.
    csv_path = os.path.join(output_dir, f"{figure_name}_points.csv")
    try:
        np.savetxt(csv_path, scaled_points, delimiter=',', header='x_phys,y_phys', comments='')
    except Exception as e:
        print(f"[evaluation_viz] Failed writing CSV: {e}")
    else:
        print(f"[evaluation_viz] Saved points CSV: {csv_path}")


def visualize_all_extracted_cases(
    data_dir: str,
    points_dir: str,
    output_dir: str,
    physical_height: float,
    physical_width: float | None = None,
    data_key: str = 'density',
    cmap: str = 'viridis',
    flip_vertical: bool = True,
    margin_frac: float = 0.035,
    point_size: float = 45.0,
    max_cases: int | None = None,
    progress_every: int = 25,
    save_background: bool = True,
    background_suffix: str = '_stage'
):
    """Generate evaluation overlays for ALL extracted cases (or a limited subset).

    Parameters mirror visualize_first_extracted_case except this loops over every
    *_{data_key}.npy file in points_dir. Each output image is named based on the
    original npz base name.
    """
    # --- Early binary compatibility guard -------------------------------------
    # Detect classic NumPy/Matplotlib mismatch (ImportError: ERR_IGNORE) *before*
    # iterating all cases, so we fail fast with guidance rather than spamming.
    if os.environ.get('SKIP_MPL_COMPAT_CHECK') != '1':
        try:
            # Matplotlib <-> NumPy ABI mismatch shows up when importing from umath
            from numpy.core.umath import ERR_IGNORE  # type: ignore  # noqa: F401
        except Exception as e:  # broad on purpose: could be ImportError or AttributeError
            print("[evaluation_viz][fatal] Detected NumPy/Matplotlib binary mismatch (e.g., 'ERR_IGNORE' missing).")
            print(f"[evaluation_viz][fatal] Underlying exception: {e.__class__.__name__}: {e}")
            print("[evaluation_viz][hint] Activate the fixed environment (e.g., 'conda activate dramp') or align versions:")
            print("[evaluation_viz][hint]   Option 1: conda install numpy=1.26.4 matplotlib=3.8.4")
            print("[evaluation_viz][hint]   Option 2: conda install matplotlib>=3.9 (with NumPy 2.x)")
            print("[evaluation_viz][hint] To bypass this check temporarily (not recommended), set SKIP_MPL_COMPAT_CHECK=1")
            return

    if not os.path.isdir(points_dir):
        print(f"[evaluation_viz] Points directory does not exist: {points_dir}")
        return

    point_files = sorted([f for f in os.listdir(points_dir) if f.endswith(f'_{data_key}.npy')])
    if not point_files:
        print(f"[evaluation_viz] No point files matching '*_{data_key}.npy' in {points_dir}")
        return

    if max_cases is not None:
        point_files = point_files[:max_cases]

    _ensure_dir(output_dir)
    total = len(point_files)
    print(f"[evaluation_viz] Generating overlays for {total} case(s)...")

    for idx, pf in enumerate(point_files, start=1):
        npz_name = _infer_original_npz_filename(pf, data_key=data_key)
        npz_path = os.path.join(data_dir, npz_name)
        points_path = os.path.join(points_dir, pf)
        if not os.path.isfile(npz_path):
            print(f"[evaluation_viz] Skipping (missing npz): {npz_name}")
            continue
        try:
            scaled_points = np.load(points_path)
        except Exception as e:
            print(f"[evaluation_viz] Skip '{pf}' (failed loading points): {e}")
            continue
        try:
            with np.load(npz_path) as data:
                if data_key not in data:
                    print(f"[evaluation_viz] Skip '{npz_name}' (missing key '{data_key}')")
                    continue
                field = data[data_key]
        except Exception as e:
            print(f"[evaluation_viz] Skip '{npz_name}' (failed loading npz): {e}")
            continue
        if field.ndim != 2:
            print(f"[evaluation_viz] Skip '{npz_name}' (non-2D array)")
            continue
        if flip_vertical:
            field = np.flipud(field)

        # Determine width and extent regardless of flip
        if physical_width is None:
            pw = physical_height
        else:
            pw = physical_width
        extent = [0.0, pw, 0.0, physical_height]

        aspect_ratio = (pw / physical_height) if physical_height > 0 else 1.0
        base = 6.2
        if aspect_ratio >= 1:
            figsize = (base, base / aspect_ratio)
        else:
            figsize = (base * aspect_ratio, base)
        fig, ax = plt.subplots(figsize=figsize, dpi=120)
        ax.imshow(field, cmap=cmap, origin='lower', extent=extent, aspect='equal')
        try:
            cb = fig.colorbar(ax.images[0], ax=ax, shrink=0.85, pad=0.02)
            cb.ax.tick_params(labelsize=8)
        except ImportError as e:
            print(f"[evaluation_viz][warn] colorbar ImportError for '{npz_name}' (continuing): {e}")
        except Exception as e:
            if os.environ.get('STRICT_VIZ_DEBUG'):
                raise
            print(f"[evaluation_viz][warn] colorbar unexpected error for '{npz_name}': {e.__class__.__name__}: {e}")
        # Save an optional background-only image before overlaying points
        if save_background:
            try:
                fig.tight_layout()
            except ImportError as e:
                print(f"[evaluation_viz][warn] tight_layout ImportError (background '{npz_name}', continuing): {e}")
                try:
                    fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.12, wspace=0.25, hspace=0.30)
                except Exception as e2:
                    print(f"[evaluation_viz][warn] subplots_adjust fallback failed (background '{npz_name}'): {e2.__class__.__name__}: {e2}")
            except Exception as e:
                if os.environ.get('STRICT_VIZ_DEBUG'):
                    raise
                print(f"[evaluation_viz][warn] tight_layout unexpected error (background '{npz_name}'): {e.__class__.__name__}: {e}")
                try:
                    fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.12, wspace=0.25, hspace=0.30)
                except Exception as e2:
                    print(f"[evaluation_viz][warn] subplots_adjust (unexpected path) failed (background '{npz_name}'): {e2.__class__.__name__}: {e2}")
            bg_base = os.path.splitext(pf)[0]
            bg_png_path = os.path.join(output_dir, f"{bg_base}{background_suffix}.png")
            try:
                fig.savefig(bg_png_path, dpi=150)
            except ImportError as e:
                print(f"[evaluation_viz][warn] savefig ImportError (background '{npz_name}', continuing): {e}")
            except Exception as e:
                if os.environ.get('STRICT_VIZ_DEBUG'):
                    raise
                print(f"[evaluation_viz][warn] savefig unexpected error (background '{npz_name}'): {e.__class__.__name__}: {e}")

        if scaled_points.size > 0:
            pts = np.asarray(scaled_points)
            if pts.ndim == 2 and pts.shape[1] == 2:
                ax.scatter(pts[:, 0], pts[:, 1], c='red', s=point_size, edgecolors='white', linewidths=0.7)
                for i, (x, y) in enumerate(pts):
                    offset = 0.012 * min(physical_height, pw)
                    ax.text(x + offset, y + offset, str(i + 1), color='white', fontsize=7, ha='left', va='bottom')
        margin = margin_frac * max(physical_height, pw)
        ax.set_xlim(-margin, pw + margin)
        ax.set_ylim(-margin, physical_height + margin)
        ax.set_xlabel('x (physical)')
        ax.set_ylabel('y (physical)')
        meta = extract_metadata(npz_name)
        if meta:
            r1 = meta.get('ramp1')
            r2 = meta.get('ramp2')
            ma = meta.get('ma')
            pres = meta.get('pres')
            if r2 is not None:
                title_str = f"r1={r1:.3f}  r2={r2:.3f}  M={ma:.3f}  p={pres}"
            else:
                title_str = f"r1={r1:.3f}  M={ma:.3f}  p={pres}"
        else:
            title_str = npz_name.replace('.npz','')
        ax.set_title(title_str, fontsize=10)
        # Layout robustness (see note above for single-case function)
        try:
            fig.tight_layout()
        except ImportError as e:
            print(f"[evaluation_viz][warn] tight_layout ImportError for '{npz_name}' (continuing): {e}")
            try:
                fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.12, wspace=0.25, hspace=0.30)
            except Exception as e2:
                print(f"[evaluation_viz][warn] subplots_adjust fallback failed for '{npz_name}': {e2.__class__.__name__}: {e2}")
        except Exception as e:
            if os.environ.get('STRICT_VIZ_DEBUG'):
                raise
            print(f"[evaluation_viz][warn] tight_layout unexpected error for '{npz_name}': {e.__class__.__name__}: {e}")
            try:
                fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.12, wspace=0.25, hspace=0.30)
            except Exception as e2:
                print(f"[evaluation_viz][warn] subplots_adjust (unexpected path) failed for '{npz_name}': {e2.__class__.__name__}: {e2}")
        base = os.path.splitext(pf)[0]
        png_path = os.path.join(output_dir, f"{base}.png")
        try:
            if os.environ.get('STRICT_VIZ_DEBUG'):
                print(f"[evaluation_viz][debug] Saving figure to {png_path}")
            fig.savefig(png_path, dpi=150)
        except ImportError as e:
            print(f"[evaluation_viz][warn] savefig ImportError for '{npz_name}' (continuing): {e}")
        except Exception as e:
            if os.environ.get('STRICT_VIZ_DEBUG'):
                raise
            print(f"[evaluation_viz][warn] savefig unexpected error for '{npz_name}': {e.__class__.__name__}: {e}")
        else:
            if os.path.isfile(png_path):
                print(f"[evaluation_viz] Saved overlay: {os.path.basename(png_path)}")
            else:
                print(f"[evaluation_viz][warn] savefig reported success but file missing: {png_path}")
        finally:
            plt.close(fig)
        # Points CSV copy
        csv_path = os.path.join(output_dir, f"{base}_points.csv")
        try:
            np.savetxt(csv_path, scaled_points, delimiter=',', header='x_phys,y_phys', comments='')
        except Exception as e:
            print(f"[evaluation_viz] Failed writing CSV for {pf}: {e}")
        if progress_every and (idx % progress_every == 0 or idx == total):
            print(f"[evaluation_viz] Done {idx}/{total}")

    print("[evaluation_viz] Batch visualization complete.")
