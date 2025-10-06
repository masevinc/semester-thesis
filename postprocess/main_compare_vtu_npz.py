#!/usr/bin/env python3
"""
Compare a VTU field against an NPZ grid field.

Steps:
- Load NPZ field (2D) and grid (x,y if present)
- Load VTU field (point/cell), robustly
- Interpolate VTU onto NPZ grid
- Compute difference and MSE
- Save side-by-side and difference plots, plus stats

Defaults are set below so you can run without args.
"""

from __future__ import annotations

import argparse
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import meshio
from typing import List, Optional, Tuple, Dict
import csv
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.tri import Triangulation
from pathlib import Path
import random

# Try optional umap import later inside function (lazy) to avoid hard dependency


# Defaults
DEFAULT_MAPPING_MODE = "auto"  # 'auto' | 'npz' | 'vtu'
DEFAULT_NPZ_Y_ORIGIN = "top"    # 'top' (image-style, like your main code) or 'bottom' (math-style)
# for batch mode
DEFAULT_VTU_ROOT = 'double_ramp_configuration/outputs/backward/sweep'#"/Users/alperensevinc/Desktop/su2/su2_alp/DDPM_pipeline_results/fullyDDPM/backward/sweep"
DEFAULT_NPZ_ROOT = 'double_ramp_configuration/inputs/double_ramp_npz_files_clamped'#"double_ramp_configuration/inputs/denorm/DDPM_fully"
# for single mode
DEFAULT_VTU = "double_ramp_configuration/outputs/backward/sweep/double_ramp_0p011_0p0488_ma_2p892_pres_199070_interpolated_arrays_density_M2p892_T300p0_P199070p0/flow.vtu"
DEFAULT_NPZ = "double_ramp_configuration/inputs/double_ramp_npz_files_clamped/double_ramp_0.011_0.0488_ma_2.892_pres_199070_interpolated_arrays.npz"
# Field names
DEFAULT_VTU_FIELD = "Density"  # e.g., Density, Pressure, Temperature
DEFAULT_NPZ_FIELD = "density"
# Output paths
DEFAULT_OUT = "postprocess_outputs/compare_auto"
# Single vs Batch analysis
DEFAULT_MODE = "batch"  # 'auto' | 'single' | 'batch'
DEFAULT_CLEAN = True    # remove target output folder(s) before writing

# Embedding (density field) defaults (three legend groups: fully-DDPM, Ground Truth (merged), semi-DDPM)
DEFAULT_FULLY_GT_ROOT = "/Users/alperensevinc/Desktop/su2/su2_alp/DDPM_pipeline_results/fullyDDPM/backward/sweep"
DEFAULT_SEMI_GT_ROOT = "/Users/alperensevinc/Desktop/su2/su2_alp/DDPM_pipeline_results/semiDDPM/backward/sweep"
DEFAULT_FULLY_GEN_ROOT = "double_ramp_configuration/inputs/denorm/DDPM_fully"
DEFAULT_SEMI_GEN_ROOT = "double_ramp_configuration/inputs/denorm/DDPM_semi"
DEFAULT_EMBED_OUT = "postprocess_outputs/density_embedding"
DEFAULT_EMBED_MAX_PER_GROUP = 1000  # cap samples per logical group (after merge of GT)
DEFAULT_EMBED_STRIDE = 1            # spatial downsample stride ( >1 reduces resolution )
DEFAULT_EMBED_STANDARDIZE = True
DEFAULT_EMBED_USE_UMAP = False       # if False or UMAP missing -> PCA fallback
DEFAULT_UMAP_NEIGHBORS = 15
DEFAULT_UMAP_MIN_DIST = 0.1
DEFAULT_EMBED_SEED = 42

# ---------------------------------------------------------------------
# INTERNAL CONFIG BLOCK
# Set USE_INTERNAL_CONFIG = True to bypass CLI and use CONFIG below.
# ---------------------------------------------------------------------
USE_INTERNAL_CONFIG = True  # Set True to run with CONFIG block (no CLI needed)

CONFIG = dict(
    mode= DEFAULT_MODE,              # 'single' | 'batch' | 'auto' | 'embed' | DEFAULT_MODE
    # Single case
    vtu=DEFAULT_VTU,
    npz=DEFAULT_NPZ,
    vtu_field=DEFAULT_VTU_FIELD,
    npz_field=DEFAULT_NPZ_FIELD,
    out=DEFAULT_OUT,
    # Batch roots
    vtu_root=DEFAULT_VTU_ROOT,
    npz_root=DEFAULT_NPZ_ROOT,
    out_root="postprocess_outputs/compare_batch",
    summary_csv="postprocess_outputs/compare_batch_summary.csv",
    # Mapping/grid
    mapping=DEFAULT_MAPPING_MODE,
    npz_y_origin=DEFAULT_NPZ_Y_ORIGIN,
    outside_value=0.0,
    clean=True,
    # Line extraction (horizontal)
    line_row=40,          # e.g., 80 (0-based)
    line_y=None,            # physical y value (float) alternative to line_row
    line_normalize=False,   # normalize by max NPZ line
    # Global density normalization for plots (divide by reference value)
    density_normalize=False,        # if True and field is density/rho -> plot normalized
    density_normalize_mode="mean", # 'max' | 'mean' (ignored if density_normalize_ref given)
    density_normalize_ref=None,    # if given (float), overrides mode; else use selected mode
    # Error metric options
    mre_ref_eps=1e-12,             # exclude |y_true| <= eps from MRE denominator
    # Embedding section (set mode='embed' to invoke)
    fully_gen_root=DEFAULT_FULLY_GEN_ROOT,
    semi_gen_root=DEFAULT_SEMI_GEN_ROOT,
    fully_gt_root=DEFAULT_FULLY_GT_ROOT,
    semi_gt_root=DEFAULT_SEMI_GT_ROOT,
    embed_out=DEFAULT_EMBED_OUT,
    embed_max_per_group=DEFAULT_EMBED_MAX_PER_GROUP,
    embed_stride=DEFAULT_EMBED_STRIDE,
    embed_standardize=DEFAULT_EMBED_STANDARDIZE,
    embed_use_umap=DEFAULT_EMBED_USE_UMAP,
    umap_neighbors=DEFAULT_UMAP_NEIGHBORS,
    umap_min_dist=DEFAULT_UMAP_MIN_DIST,
    embed_seed=DEFAULT_EMBED_SEED,
)


def _sanitize_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _extract_mach_from_text(*texts: str) -> Optional[float]:
    """Try to parse a Mach number from given text fragments.

    Supports patterns like: 'M2p892', 'ma_2p892', 'mach_2.892', 'Mach2.9'.
    Returns float if found, else None.
    """
    if not texts:
        return None
    patterns = [
        r"(?i)\bM(?:ach)?[_-]?(\d+p\d+|\d+(?:\.\d+)?)\b",
        r"(?i)\bma[_-]?(\d+p\d+|\d+(?:\.\d+)?)\b",
    ]
    for t in texts:
        if not t:
            continue
        for pat in patterns:
            m = re.search(pat, t)
            if m:
                g = m.group(1)
                try:
                    g = g.replace('p', '.')
                    return float(g)
                except Exception:
                    pass
    return None


def candidate_keys(keys: List[str], preferred: str, tokens: Optional[List[str]] = None) -> List[str]:
    keys = list(keys)
    exact, sanitized, token_like, others = [], [], [], []
    pref_s = _sanitize_key(preferred)
    tokens = [t.lower() for t in (tokens or [])]
    for k in keys:
        if k.lower() == preferred.lower():
            exact.append(k)
        elif _sanitize_key(k) == pref_s:
            sanitized.append(k)
        elif tokens and any(t in k.lower() for t in tokens):
            token_like.append(k)
        else:
            others.append(k)
    out, seen = [], set()
    for group in (exact, sanitized, token_like, others):
        for k in group:
            if k not in seen:
                seen.add(k)
                out.append(k)
    return out


def load_npz_grid(
    npz_path: str,
    field_key: str,
    mapping_mode: str = "auto",
    vtu_bounds: Optional[Tuple[float, float, float, float]] = None,
    npz_y_origin: str = "top",
):
    data = np.load(npz_path, allow_pickle=True)
    if field_key not in data:
        raise KeyError(f"Field '{field_key}' not in NPZ. Keys: {list(data.keys())}")
    Z = np.asarray(data[field_key]).astype(float)
    if Z.ndim != 2:
        raise ValueError(f"NPZ field must be 2D; got {Z.shape}")
    ny, nx = Z.shape
    x = data.get('x', data.get('x_coords', None))
    y = data.get('y', data.get('y_coords', None))
    # Decide mapping strategy
    def looks_unity(arr: np.ndarray) -> bool:
        if arr is None:
            return True
        arr = np.asarray(arr).ravel()
        if arr.size < 2:
            return True
        amin, amax = float(np.nanmin(arr)), float(np.nanmax(arr))
        return (amin >= -1e-3) and (amax <= 1.0 + 1e-3)

    use_vtu_span = False
    if mapping_mode == "vtu":
        use_vtu_span = True
    elif mapping_mode == "auto":
        # If coords missing or look normalized 0..1, map to VTU bounds when provided
        if vtu_bounds is not None and (x is None or y is None or (looks_unity(x) and looks_unity(y))):
            use_vtu_span = True

    if use_vtu_span and vtu_bounds is not None:
        xmin, xmax, ymin, ymax = vtu_bounds
        x = np.linspace(xmin, xmax, nx)
        y = np.linspace(ymin, ymax, ny)
    else:
        if x is None or y is None:
            # Default to 0..1 if nothing else
            x = np.linspace(0.5 / nx, 1 - 0.5 / nx, nx)
            y = np.linspace(0.5 / ny, 1 - 0.5 / ny, ny)
        else:
            x = np.asarray(x).squeeze()
            y = np.asarray(y).squeeze()
            if x.ndim > 1:
                x = x.ravel()
            if y.ndim > 1:
                y = y.ravel()
            if x.size != nx:
                x = np.linspace(0.5 / nx, 1 - 0.5 / nx, nx)
            if y.size != ny:
                y = np.linspace(0.5 / ny, 1 - 0.5 / ny, ny)
    # For consistent math-style coordinates (y increasing upward), flip NPZ if its index 0 is at the top
    if npz_y_origin.lower() == "top":
        Z = np.flipud(Z)

    # sort axes
    ix = np.argsort(x)
    iy = np.argsort(y)
    if not np.all(ix == np.arange(nx)):
        Z = Z[:, ix]
        x = x[ix]
    if not np.all(iy == np.arange(ny)):
        Z = Z[iy, :]
        y = y[iy]
    X, Y = np.meshgrid(x, y)
    return Z, X, Y


def is_nonconstant(a: np.ndarray, eps: float = 1e-12) -> bool:
    return a.size > 0 and float(np.nanstd(a)) > eps


def extract_vtu_field(vtu_path: str, field_preference: str) -> Tuple[np.ndarray, np.ndarray, str]:
    """Return (pts_xy, values, chosen_key) using meshio then VTK fallback."""
    # meshio path
    try:
        m = meshio.read(vtu_path)
        pts = m.points[:, :2]
        pd_keys = list(m.point_data.keys())
        for k in candidate_keys(pd_keys, field_preference, tokens=['rho','density','temp','pressure']):
            vals = np.asarray(m.point_data[k]).astype(float).ravel()
            if is_nonconstant(vals):
                return pts, vals, k
        cd_keys = list(m.cell_data.keys())
        for k in candidate_keys(cd_keys, field_preference, tokens=['rho','density','temp','pressure']):
            per_block = m.cell_data[k]
            centers, values = [], []
            for cells_block, arr in zip(m.cells, per_block):
                arr = np.asarray(arr).astype(float).ravel()
                for conn in cells_block.data:
                    centers.append(m.points[conn, :2].mean(axis=0))
                values.append(arr)
            if values:
                centers = np.asarray(centers)
                vals = np.concatenate(values)
                if is_nonconstant(vals):
                    return centers, vals, k
    except Exception:
        pass
    # VTK fallback
    import vtk  # type: ignore
    from vtk.util.numpy_support import vtk_to_numpy  # type: ignore
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(vtu_path)
    reader.Update()
    ug = reader.GetOutput()
    pts3 = vtk_to_numpy(ug.GetPoints().GetData())
    pts = pts3[:, :2]
    pd = ug.GetPointData()
    cd = ug.GetCellData()
    pd_keys = [pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]
    for k in candidate_keys(pd_keys, field_preference, tokens=['rho','density','temp','pressure']):
        arr = pd.GetArray(k)
        if arr is None:
            continue
        vals = vtk_to_numpy(arr).astype(float).ravel()
        if is_nonconstant(vals):
            return pts, vals, k
    cd_keys = [cd.GetArrayName(i) for i in range(cd.GetNumberOfArrays())]
    for k in candidate_keys(cd_keys, field_preference, tokens=['rho','density','temp','pressure']):
        arr = cd.GetArray(k)
        if arr is None:
            continue
        vals = vtk_to_numpy(arr).astype(float).ravel()
        centers = []
        for i in range(ug.GetNumberOfCells()):
            cell = ug.GetCell(i)
            ids = [cell.GetPointId(j) for j in range(cell.GetNumberOfPoints())]
            centers.append(pts[ids].mean(axis=0))
        centers = np.asarray(centers)
        if is_nonconstant(vals):
            return centers, vals, k
    raise KeyError(f"Field '{field_preference}' not found in VTU: {vtu_path}")

def _triangulation_from_meshio(m: meshio.Mesh) -> Optional[Triangulation]:
    """Build a Triangulation from meshio mesh using triangles/quads only."""
    tris = []
    for block in m.cells:
        ctype = block.type
        conn = block.data
        if ctype == "triangle":
            tris.extend([tuple(t) for t in conn])
        elif ctype == "quad":
            for q in conn:
                i0, i1, i2, i3 = q
                tris.append((i0, i1, i2))
                tris.append((i0, i2, i3))
    if not tris:
        return None
    pts = m.points[:, :2]
    return Triangulation(pts[:, 0], pts[:, 1], np.asarray(tris))


def _triangulation_from_vtk(ug) -> Optional[Triangulation]:
    """Build a Triangulation from a VTK unstructured grid for triangles/quads."""
    try:
        from vtk.util.numpy_support import vtk_to_numpy  # type: ignore
    except Exception:
        return None
    pts3 = vtk_to_numpy(ug.GetPoints().GetData())
    pts = pts3[:, :2]
    tris = []
    VTK_TRIANGLE = 5
    VTK_QUAD = 9
    n_cells = ug.GetNumberOfCells()
    for i in range(n_cells):
        ctype = ug.GetCellType(i)
        cell = ug.GetCell(i)
        ids = [cell.GetPointId(j) for j in range(cell.GetNumberOfPoints())]
        if ctype == VTK_TRIANGLE and len(ids) == 3:
            tris.append((ids[0], ids[1], ids[2]))
        elif ctype == VTK_QUAD and len(ids) == 4:
            i0, i1, i2, i3 = ids
            tris.append((i0, i1, i2))
            tris.append((i0, i2, i3))
    if not tris:
        return None
    return Triangulation(pts[:, 0], pts[:, 1], np.asarray(tris))


def get_vtu_triangulation(vtu_path: str) -> Optional[Triangulation]:
    """Try to construct a mesh connectivity-based triangulation for domain masking."""
    # Try meshio first
    try:
        m = meshio.read(vtu_path)
        tri = _triangulation_from_meshio(m)
        if tri is not None:
            return tri
    except Exception:
        pass
    # Fallback to VTK
    try:
        import vtk  # type: ignore
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(vtu_path)
        reader.Update()
        ug = reader.GetOutput()
        return _triangulation_from_vtk(ug)
    except Exception:
        return None


def interpolate_to_grid(
    pts_xy: np.ndarray,
    vals: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    inside_mask: Optional[np.ndarray] = None,
    outside_value: Optional[float] = 0.0,
) -> np.ndarray:
    """Interpolate with linear then nearest fill, but only inside the mesh; set outside to outside_value.

    - inside_mask: boolean array same shape as X/Y indicating points inside mesh connectivity.
    - outside_value: value to assign outside (use None to keep NaN outside).
    """
    Zi = griddata(pts_xy, vals, (X, Y), method='linear')

    if inside_mask is not None:
        # Force outside region to desired value/NaN before nearest fill
        if outside_value is not None:
            Zi[~inside_mask] = float(outside_value)
        else:
            Zi[~inside_mask] = np.nan
        # Fill only inside region where still NaN
        fill_mask = np.isnan(Zi) & inside_mask
    else:
        fill_mask = np.isnan(Zi)

    if np.any(fill_mask):
        Zi_near = griddata(pts_xy, vals, (X, Y), method='nearest')
        Zi[fill_mask] = Zi_near[fill_mask]

    # Ensure any stray values outside are set properly (in case linear/nearest produced something)
    if inside_mask is not None:
        if outside_value is not None:
            Zi[~inside_mask] = float(outside_value)
        else:
            Zi[~inside_mask] = np.nan
    return Zi


def render_array_to_rgb(array: np.ndarray, cmap: str = 'viridis', dpi: int = 100) -> np.ndarray:
    """Render a 2D array to an RGB numpy image via Matplotlib canvas (compatibility path)."""
    height, width = array.shape
    figsize = (width / dpi, height / dpi)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(array, cmap=cmap)
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    canvas = FigureCanvas(fig)
    canvas.draw()
    canvas_width, canvas_height = canvas.get_width_height()
    image_np = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image_np = image_np.reshape((canvas_height, canvas_width, 3))
    plt.close(fig)
    return image_np


def plots(outdir: str, X: np.ndarray, Y: np.ndarray, Z_npz: np.ndarray, Z_vtu: np.ndarray, field_npz: str, field_vtu: str) -> Tuple[float, float, float]:
    os.makedirs(outdir, exist_ok=True)
    diff = Z_vtu - Z_npz
    mse = float(np.nanmean((diff) ** 2))
    rmse = float(np.sqrt(mse)) if np.isfinite(mse) else np.nan
    # Relative metrics
    denom_energy = float(np.nanmean(Z_npz ** 2)) if np.isfinite(Z_npz).any() else np.nan
    rel_mse = float(mse / denom_energy) if (denom_energy is not None and np.isfinite(denom_energy) and denom_energy > 0) else np.nan
    npz_range = float(np.nanmax(Z_npz) - np.nanmin(Z_npz)) if np.isfinite(Z_npz).any() else np.nan
    rmse_pct = float(100.0 * rmse / npz_range) if (np.isfinite(npz_range) and npz_range > 0) else np.nan
    extent = [float(X.min()), float(X.max()), float(Y.min()), float(Y.max())]

    # Shared color scale for NPZ and VTU-on-grid
    vmin_shared = float(np.nanmin(np.stack([Z_npz, Z_vtu])))
    vmax_shared = float(np.nanmax(np.stack([Z_npz, Z_vtu])))

    def save_im(arr, title, fname, vmin=None, vmax=None, cmap='viridis'):
        plt.figure(figsize=(6, 5))
        im = plt.imshow(arr, origin='lower', aspect='equal', extent=extent, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.title(title)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, fname), dpi=220)
        plt.close()

    # Use the same vmin/vmax for both field plots
    save_im(Z_npz, f"Prior CFD: {field_npz}", "GT_npz.png", vmin=vmin_shared, vmax=vmax_shared, cmap='viridis')
    save_im(Z_vtu, f"Pipeline CFD: {field_vtu}", "CFD_vtu_on_grid.png", vmin=vmin_shared, vmax=vmax_shared, cmap='viridis')

    # Diff: symmetric range around 0 with diverging colormap
    diff_abs = float(np.nanmax(np.abs(diff))) if np.isfinite(diff).any() else 1.0
    save_im(diff, "|ε_rel|", "diff.png", vmin=-diff_abs, vmax=diff_abs, cmap='coolwarm')

    with open(os.path.join(outdir, "report.txt"), "w") as f:
        f.write(f"NPZ field: {field_npz}\n")
        f.write(f"VTU field: {field_vtu}\n")
        f.write(f"MSE: {mse:.6e}\n")
        f.write(f"RMSE: {rmse:.6e}\n")
        f.write(f"Relative MSE (mse/mean(npz^2)): {rel_mse:.6e}\n")
        f.write(f"RMSE % of NPZ range: {rmse_pct:.3f}%\n")

    # Combined subplot (1x3): NPZ | VTU | Diff
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    im0 = axes[0].imshow(Z_npz, origin='lower', aspect='equal', extent=extent, vmin=vmin_shared, vmax=vmax_shared, cmap='viridis')
    axes[0].set_title(f"Fully DDPM Output: {field_vtu}")
    im1 = axes[1].imshow(Z_vtu, origin='lower', aspect='equal', extent=extent, vmin=vmin_shared, vmax=vmax_shared, cmap='viridis')
    axes[1].set_title(f"Pipeline CFD: {field_vtu}")
    im2 = axes[2].imshow(diff, origin='lower', aspect='equal', extent=extent, vmin=-diff_abs, vmax=diff_abs, cmap='coolwarm')
    axes[2].set_title("|ε_rel|")
    # Colorbars
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    fig.savefig(os.path.join(outdir, "comparison_subplot.png"), dpi=220)
    plt.close(fig)

    return mse, rel_mse, rmse_pct


def extract_line_profile(
    outdir: str,
    X: np.ndarray,
    Y: np.ndarray,
    Z_npz: np.ndarray,
    Z_vtu: np.ndarray,
    row_index: Optional[int] = None,
    y_value: Optional[float] = None,
    normalize: bool = False,
    field_label: str = "density",
    mach_number: Optional[float] = None,
    style_like_sample: bool = True,
) -> Optional[str]:
    """Extract a horizontal line (constant y) profile from NPZ & VTU-on-grid arrays.

    Priority: if row_index provided use it (0-based). Else if y_value provided use closest row.
    Saves CSV (x, npz, vtu) and plot. Returns path to CSV or None if not extracted.
    """
    if row_index is None and y_value is None:
        return None
    ny, nx = Z_npz.shape
    # Determine row index
    if row_index is None:
        # pick closest y
        y_arr = Y[:, 0]  # consistent across row
        j = int(np.argmin(np.abs(y_arr - float(y_value))))
    else:
        j = int(row_index)
    if j < 0 or j >= ny:
        print(f"[LINE] Row index {j} out of range (0..{ny-1}); skipping line extraction")
        return None
    # Extract arrays
    x_line = X[j, :].astype(float)
    npz_line = Z_npz[j, :].astype(float)
    vtu_line = Z_vtu[j, :].astype(float)
    # Normalization (by max of NPZ line) if requested and max>0
    if normalize:
        ref = np.nanmax(npz_line)
        if np.isfinite(ref) and ref != 0:
            npz_line = npz_line / ref
            vtu_line = vtu_line / ref
    # Prepare output
    line_dir = os.path.join(outdir, "lines")
    os.makedirs(line_dir, exist_ok=True)
    y_sel = float(Y[j, 0])
    field_slug = _sanitize_key(field_label or "field")
    csv_path = os.path.join(line_dir, f"{field_slug}_line_yindex_{j}_y_{y_sel:.6g}.csv")
    header = "x,npz,vtu"
    arr = np.column_stack([x_line, npz_line, vtu_line])
    np.savetxt(csv_path, arr, delimiter=",", header=header, comments="")
    # Plot
    plt.figure(figsize=(5.6, 5.6))
    if style_like_sample:
        # Emulate the clean paper style like the provided sample
        plt.grid(True, color="#b0b0b0", alpha=0.4, linestyle='-', linewidth=0.8)
        # CFD (VTU-on-grid): orange solid
        plt.plot(x_line, vtu_line, label="CFD", color="#ff7f0e", linewidth=1.8)
        # NPZ (GT): purple dashed
        plt.plot(x_line, npz_line, label="Fully DDPM", color="#9467bd", linewidth=2.2, linestyle="--")
    else:
        plt.plot(x_line, vtu_line, label="VTU", color="#ff7f0e", linewidth=1.1)
        plt.plot(x_line, npz_line, label="NPZ", color="#1f77b4", linewidth=1.4, linestyle="--")
    plt.xlabel("x-position (m)")
    # Y-axis label: special-case for Mach and Density
    _lbl = (field_label or '').strip()
    _lbl_lower = _lbl.lower()
    # If label was passed like 'Norm. density', strip prefix for base-type detection
    _base_lower = _lbl_lower
    if _base_lower.startswith('norm.'):
        _base_lower = _base_lower[5:].strip()
    if _base_lower == 'mach':
        yl = "Ma Number"
    elif _base_lower in ('density', 'rho'):
        yl = "Norm. Density" if normalize else "Density (kg / m³)"
    elif _base_lower in ('temperature', 'temp', 't'):
        yl = "Norm. Temperature" if normalize else "Temperature (K)"
    else:
        yl = f"{('Norm. ' if normalize else '')}{field_label}" if field_label else ("Norm. value" if normalize else "Value")
    plt.ylabel(yl)
    # No plot title for line plots per request
    plt.legend(frameon=True, loc='upper right')
    plt.tight_layout()
    plot_path = os.path.join(line_dir, f"{field_slug}_line_yindex_{j}_y_{y_sel:.6g}.png")
    plt.savefig(plot_path, dpi=220)
    plt.close()
    # Log
    with open(os.path.join(line_dir, "lines_readme.txt"), "a") as f:
        f.write(
            f"field={field_label}, y_index={j}, y_value={y_sel:.9g}, csv={os.path.basename(csv_path)}, plot={os.path.basename(plot_path)}\n"
        )
    print(f"[LINE] Saved line profile -> {csv_path}")
    return csv_path


# ---------------------------------------------------------------------------
# Density Embedding (UMAP/PCA) Section
# ---------------------------------------------------------------------------
def _gather_npz_files(root: Optional[str]) -> list[Path]:
    files: list[Path] = []
    if not root:
        return files
    p = Path(root)
    if not p.exists():
        print(f"[EMBED][WARN] Root does not exist: {root}")
        return files
    for fp in p.rglob("*.npz"):
        files.append(fp)
    return files


def _load_density_aligned(fp: Path, npz_y_origin: str = "top") -> Optional[np.ndarray]:
    """Load a density-like array from NPZ and apply the same orientation/alignment
    conventions used in compare_vtu_npz:

    - Use keys: density/rho (case-insensitive variants)
    - If x/y coordinate vectors exist, ensure they are 1D, sorted ascending.
    - If y-origin is 'top', flip vertically so output has Y increasing upward.
    - Return a 2D array (ny, nx) in canonical orientation (ascending x, ascending y).
    """
    try:
        data = np.load(str(fp), allow_pickle=True)
    except Exception as e:
        print(f"[EMBED][WARN] Failed loading NPZ {fp.name}: {e}")
        return None

    # Locate density field
    density_key = None
    for k in ("density", "rho", "Density", "Rho"):
        if k in data:
            density_key = k
            break
    if density_key is None:
        return None
    arr = np.asarray(data[density_key]).astype(float)
    # Coerce to 2D slice if higher dimensional
    if arr.ndim > 2:
        arr = arr[..., 0]
    if arr.ndim != 2:
        return None

    ny, nx = arr.shape
    x = data.get('x', data.get('x_coords', None))
    y = data.get('y', data.get('y_coords', None))

    # Flip vertically if origin at top (image style) to convert to math-style (y increasing upward)
    if npz_y_origin.lower() == 'top':
        arr = np.flipud(arr)

    # Sort axes if coordinate vectors present
    try:
        if x is not None:
            x = np.asarray(x).squeeze()
            if x.ndim == 1 and x.size == nx:
                ix = np.argsort(x)
                if not np.all(ix == np.arange(nx)):
                    arr = arr[:, ix]
        if y is not None:
            y = np.asarray(y).squeeze()
            if y.ndim == 1 and y.size == ny:
                iy = np.argsort(y)
                if not np.all(iy == np.arange(ny)):
                    arr = arr[iy, :]
    except Exception:
        # Non-fatal; continue with raw ordering
        pass
    return arr


def _downsample(arr: np.ndarray, stride: int) -> np.ndarray:
    if stride and stride > 1:
        return arr[::stride, ::stride]
    return arr


def _standardize_matrix(X: np.ndarray) -> np.ndarray:
    # X shape: (n_samples, n_features)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return (X - mean) / std


def _pca_2d(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (embedding_2d, variance_ratio[2]) using SVD-based PCA.

    Variance ratio is computed as S^2 / sum(S^2) for the first two singular values.
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    try:
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    except np.linalg.LinAlgError:
        U, S, Vt = np.linalg.svd(Xc + 1e-9 * np.random.randn(*Xc.shape), full_matrices=False)
    comps = U[:, :2] * S[:2]
    var = (S ** 2)
    total = var.sum()
    if total <= 0:
        var_ratio = np.array([0.0, 0.0])
    else:
        var_ratio = var[:2] / total
    return comps, var_ratio


def run_density_embedding(
    fully_gen_root: Optional[str],
    semi_gen_root: Optional[str],
    fully_gt_root: Optional[str],
    semi_gt_root: Optional[str],
    out_dir: str,
    max_per_group: int = 1000,
    stride: int = 1,
    standardize: bool = True,
    use_umap: bool = True,
    umap_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    seed: int = 42,
    npz_y_origin: str = "top",
    mapping_mode: str = "auto",
) -> None:
    """Create a 2D embedding of density fields across three legend groups.

    Legend groups:
      1. Fully-conditioned DDPM  (generated)
      2. Ground Truth (merged fully + semi GT roots)
      3. Semi-conditioned DDPM  (generated)

    Each NPZ file contributes one sample (flattened density after optional stride).
    Shapes must match; the most common shape is selected and others are skipped.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "log.txt")

    # Gather generated NPZ files
    fully_gen_files = _gather_npz_files(fully_gen_root)
    semi_gen_files = _gather_npz_files(semi_gen_root)

    # Attempt to gather CFD VTU ground truth (flow.vtu) paths; if none found, fallback to NPZ GT roots
    def _gather_vtu_map(root: Optional[str]) -> dict[str, Path]:
        mapping: dict[str, Path] = {}
        if not root:
            return mapping
        p = Path(root)
        if not p.exists():
            return mapping
        for vtu in p.rglob('flow.vtu'):
            case_dir = vtu.parent
            key = case_dir.name
            # replicate normalize logic from batch comparison
            key_norm = re.sub(r"(?<=\d)p(?=\d)", ".", key.lower())
            idx = key_norm.find('_interpolated_arrays')
            if idx != -1:
                key_norm = key_norm[:idx]
            mapping[key_norm] = vtu
        return mapping

    fully_vtu_map = _gather_vtu_map(fully_gt_root)
    semi_vtu_map = _gather_vtu_map(semi_gt_root)
    vtu_gt_available = (len(fully_vtu_map) + len(semi_vtu_map)) > 0

    if not vtu_gt_available:
        # Fallback: load NPZ ground truth files (legacy behavior)
        fully_gt_files = _gather_npz_files(fully_gt_root)
        semi_gt_files = _gather_npz_files(semi_gt_root)
        gt_files = fully_gt_files + semi_gt_files
    else:
        gt_files = []  # Will be constructed from VTUs aligned to generated samples

    def load_group(files: list[Path], label: str) -> list[tuple[Path, np.ndarray]]:
        out: list[tuple[Path, np.ndarray]] = []
        for fp in files:
            arr = _load_density_aligned(fp, npz_y_origin=npz_y_origin)
            if arr is None:
                continue
            arr = _downsample(arr, stride)
            out.append((fp, arr))
        print(f"[EMBED] Loaded {len(out)} density arrays for group '{label}'")
        return out

    g_fully = load_group(fully_gen_files, "Fully-conditioned DDPM")
    g_semi = load_group(semi_gen_files, "Semi-conditioned DDPM")
    if vtu_gt_available:
        # Build ground truth arrays by pairing each generated NPZ with matching CFD VTU and interpolating
        g_gt: list[tuple[Path, np.ndarray]] = []
        def _normalize_key_from_npz(fp: Path) -> str:
            base = fp.stem.lower()
            base = re.sub(r"(?<=\d)p(?=\d)", ".", base)
            idx = base.find('_interpolated_arrays')
            if idx != -1:
                base = base[:idx]
            return base

        # Helper to create ground truth array for one NPZ + VTU pair
        def _make_gt_array(npz_fp: Path, vtu_path: Path) -> Optional[np.ndarray]:
            try:
                # Extract VTU density field and bounds first
                pts, vals, _ = extract_vtu_field(str(vtu_path), 'Density')
                xmin, xmax = float(np.nanmin(pts[:,0])), float(np.nanmax(pts[:,0]))
                ymin, ymax = float(np.nanmin(pts[:,1])), float(np.nanmax(pts[:,1]))
                # Build NPZ-aligned grid using same helper as comparison (ensures consistent remapping when NPZ coords are normalized)
                try:
                    Z_npz_dummy, Xg, Yg = load_npz_grid(
                        str(npz_fp),
                        'density' if 'density' in np.load(str(npz_fp)) else ('rho' if 'rho' in np.load(str(npz_fp)) else 'density'),
                        mapping_mode=mapping_mode,
                        vtu_bounds=(xmin, xmax, ymin, ymax),
                        npz_y_origin=npz_y_origin,
                    )
                except Exception:
                    # Fallback simple grid if load_npz_grid fails
                    data = np.load(str(npz_fp), allow_pickle=True)
                    field_key = 'density' if 'density' in data else ('rho' if 'rho' in data else list(data.keys())[0])
                    Znpz = np.asarray(data[field_key])
                    ny, nx = Znpz.shape[:2]
                    Xg, Yg = np.meshgrid(np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny))
                tri = get_vtu_triangulation(str(vtu_path))
                inside_mask = None
                if tri is not None:
                    finder = tri.get_trifinder()
                    tri_ids = finder(Xg, Yg)
                    inside_mask = tri_ids != -1
                Z_vtu_on_grid = interpolate_to_grid(pts, vals, Xg, Yg, inside_mask=inside_mask, outside_value=0.0)
                if stride and stride > 1:
                    Z_vtu_on_grid = Z_vtu_on_grid[::stride, ::stride]
                return Z_vtu_on_grid
            except Exception as e:
                print(f"[EMBED][WARN] Ground truth VTU load failed for {vtu_path}: {e}")
                return None

        # Collect keys from generated groups to attempt pairing
        for gen_group in (g_fully, g_semi):
            for fp, _ in gen_group:
                key = _normalize_key_from_npz(fp)
                vtu_path = fully_vtu_map.get(key) or semi_vtu_map.get(key)
                if not vtu_path:
                    continue
                gt_arr = _make_gt_array(fp, vtu_path)
                if gt_arr is not None:
                    g_gt.append((vtu_path, gt_arr))
        print(f"[EMBED] Built {len(g_gt)} ground truth CFD arrays from VTU files")
    else:
        g_gt = load_group(gt_files, "Ground Truth")

    # Determine most common shape
    all_shapes = [a.shape for _, a in (g_fully + g_semi + g_gt)]
    if not all_shapes:
        print("[EMBED][ERROR] No density arrays loaded; aborting.")
        return
    # Frequency map
    shape_counts: Dict[tuple[int,int], int] = {}
    for s in all_shapes:
        shape_counts[s] = shape_counts.get(s, 0) + 1
    target_shape = max(shape_counts.items(), key=lambda kv: kv[1])[0]
    print(f"[EMBED] Target shape selected: {target_shape} (most common)")

    def filter_shape(group: list[tuple[Path,np.ndarray]], name: str) -> list[tuple[Path,np.ndarray]]:
        kept = [(fp, a) for fp, a in group if a.shape == target_shape]
        dropped = len(group) - len(kept)
        if dropped:
            print(f"[EMBED][WARN] Dropped {dropped} samples in '{name}' due to mismatched shape")
        return kept

    g_fully = filter_shape(g_fully, "Fully-conditioned DDPM")
    g_semi = filter_shape(g_semi, "Semi-conditioned DDPM")
    g_gt = filter_shape(g_gt, "Ground Truth")

    # Optional cap per group
    def cap(group: list[tuple[Path,np.ndarray]], name: str) -> list[tuple[Path,np.ndarray]]:
        if max_per_group and len(group) > max_per_group:
            random.shuffle(group)
            group = group[:max_per_group]
            print(f"[EMBED] Capped '{name}' to {len(group)} samples")
        return group

    g_fully = cap(g_fully, "Fully-conditioned DDPM")
    g_semi = cap(g_semi, "Semi-conditioned DDPM")
    g_gt = cap(g_gt, "Ground Truth")

    # Build matrices
    def flatten_group(group: list[tuple[Path,np.ndarray]]) -> tuple[np.ndarray, list[Path]]:
        if not group:
            return np.empty((0, target_shape[0]*target_shape[1])), []
        X = np.stack([a.ravel() for _, a in group], axis=0)
        files = [fp for fp, _ in group]
        return X, files

    X_fully, files_fully = flatten_group(g_fully)
    X_gt, files_gt = flatten_group(g_gt)
    X_semi, files_semi = flatten_group(g_semi)

    # Concatenate in legend order: fully, gt, semi
    X_all = np.concatenate([X_fully, X_gt, X_semi], axis=0)
    labels = (["Fully-conditioned DDPM"] * len(X_fully) +
              ["Ground Truth"] * len(X_gt) +
              ["Semi-conditioned DDPM"] * len(X_semi))
    file_list = files_fully + files_gt + files_semi

    if X_all.shape[0] < 2:
        print("[EMBED][ERROR] Need at least two samples for embedding.")
        return

    if standardize:
        print("[EMBED] Standardizing features (per-pixel)")
        X_all = _standardize_matrix(X_all)

    method_used = "pca"
    emb = None
    pca_var_ratio: Optional[np.ndarray] = None
    if use_umap:
        try:
            import umap  # type: ignore
            reducer = umap.UMAP(n_components=2, n_neighbors=umap_neighbors, min_dist=umap_min_dist,
                                 metric='euclidean', random_state=seed)
            emb = reducer.fit_transform(X_all)
            method_used = "umap"
            print("[EMBED] Used UMAP for embedding")
        except Exception as e:
            print(f"[EMBED][WARN] UMAP unavailable ({e}); falling back to PCA")
    if emb is None:
        emb, pca_var_ratio = _pca_2d(X_all)
        method_used = "pca"
        print("[EMBED] Used PCA for embedding")

    # Plot (Component axes are dimensionless latent coordinates: PCA -> linear projections of standardized pixel densities; UMAP -> non-linear manifold coordinates.)
    plt.figure(figsize=(6.8, 5.4))
    color_map = {
        "Fully-conditioned DDPM": "#2ca02c",  # green
        "Ground Truth": "#1f77b4",            # blue
        "Semi-conditioned DDPM": "#ff7f0e",    # orange
    }
    for lbl in ["Fully-conditioned DDPM", "Ground Truth", "Semi-conditioned DDPM"]:
        mask = [lbl_candidate == lbl for lbl_candidate in labels]
        if any(mask):
            arr = emb[np.array(mask)]
            plt.scatter(arr[:,0], arr[:,1], s=22, alpha=0.85, label=lbl, color=color_map[lbl], edgecolors='none')
    if method_used == 'pca' and pca_var_ratio is not None:
        plt.xlabel(f"Principal Component 1 ({pca_var_ratio[0]*100:.2f}% variance)")
        plt.ylabel(f"Principal Component 2 ({pca_var_ratio[1]*100:.2f}% variance)")
    elif method_used == 'umap':
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
    else:
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
    # Thesis-style subtle grid
    plt.minorticks_on()
    plt.grid(True, which='major', color='#d0d0d0', linestyle='--', linewidth=0.6, alpha=0.8)
    plt.grid(True, which='minor', color='#f0f0f0', linestyle=':', linewidth=0.5, alpha=0.8)
    # Legend: move to top-right, slightly lower, smaller font to avoid overlap
    plt.legend(frameon=True,
               loc='upper right',
               bbox_to_anchor=(1.0, 0.97),
               fontsize=8,
               borderpad=0.4,
               labelspacing=0.4,
               handlelength=1.2,
               handletextpad=0.6)
    plt.tight_layout()
    fig_path = os.path.join(out_dir, f"density_embedding_{method_used}.png")
    plt.savefig(fig_path, dpi=220)
    plt.close()
    print(f"[EMBED] Saved embedding figure -> {fig_path}")

    # CSV export
    csv_path = os.path.join(out_dir, f"density_embedding_{method_used}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label", "file", "comp1", "comp2"])
        for i, (lbl, fp, (c1, c2)) in enumerate(zip(labels, file_list, emb)):
            writer.writerow([i, lbl, str(fp), f"{c1:.6g}", f"{c2:.6g}"])
    print(f"[EMBED] Saved embedding CSV -> {csv_path}")

    # Log
    with open(log_path, 'w') as f:
        f.write("Density Embedding Log\n")
        f.write(f"method={method_used}\n")
        f.write(f"fully_gen_root={fully_gen_root}\n")
        f.write(f"semi_gen_root={semi_gen_root}\n")
        f.write(f"fully_gt_root={fully_gt_root}\n")
        f.write(f"semi_gt_root={semi_gt_root}\n")
        f.write(f"samples_fully={len(X_fully)}\n")
        f.write(f"samples_gt={len(X_gt)}\n")
        f.write(f"samples_semi={len(X_semi)}\n")
        f.write(f"target_shape={target_shape}\n")
        f.write(f"stride={stride}\n")
        f.write(f"standardize={standardize}\n")
        f.write(f"umap_requested={use_umap}\n")
        f.write(f"umap_neighbors={umap_neighbors}\n")
        f.write(f"umap_min_dist={umap_min_dist}\n")
        f.write(f"seed={seed}\n")
    print(f"[EMBED] Log written -> {log_path}")


def main():
    if USE_INTERNAL_CONFIG:
        class _Args:  # simple namespace
            pass
        args = _Args()
        for k, v in CONFIG.items():
            setattr(args, k, v)
        # mimic argparse flag for --no-clean
        setattr(args, 'no_clean', (not CONFIG.get('clean', True)))
        print("[INFO] Running with internal CONFIG (edit CONFIG dict near top of file).")
    else:
        p = argparse.ArgumentParser(description="Compare VTU field with NPZ field or create density embeddings (mode=embed).")
        p.add_argument("--vtu", default=None)
        p.add_argument("--vtu-field", default=None)
        p.add_argument("--npz", default=None)
        p.add_argument("--npz-field", default=None)
        p.add_argument("--out", default=None)
        p.add_argument("--mapping", choices=["auto","npz","vtu"], default=None, help="How to build NPZ grid: use NPZ coords, VTU bounds, or auto-detect")
        p.add_argument("--npz-y-origin", choices=["top","bottom"], default=None, help="Treat NPZ row 0 as 'top' (image) or 'bottom' (math)")
        p.add_argument("--outside-value", type=float, default=None, help="Value to assign outside the VTU mesh domain on the grid (default 0)")
        p.add_argument("--mode", choices=["auto","single","batch","embed"], default=None, help="Run comparison (single/batch/auto) or density embedding (embed)")
        p.add_argument("--no-clean", action="store_true", help="Do not delete existing output folder(s) before writing")
        # Batch mode options
        p.add_argument("--vtu-root", default=None, help="Root folder to search for VTU cases (recursively, looking for flow.vtu)")
        p.add_argument("--npz-root", default=None, help="Root folder to search for NPZ files (recursively, .npz)")
        p.add_argument("--out-root", default="postprocess_outputs/compare_batch", help="Output root for batch results (each case gets its own folder)")
        p.add_argument("--summary-csv", default="postprocess_outputs/compare_batch_summary.csv", help="CSV file to write per-case MSE summary")
        # Line extraction options
        p.add_argument("--line-row", type=int, default=None, help="Row index (0-based) for horizontal line extraction (constant y)")
        p.add_argument("--line-y", type=float, default=None, help="Physical y-value for horizontal line extraction (closest row is used)")
        p.add_argument("--line-normalize", action="store_true", help="Normalize line values by max of NPZ line (for comparative plotting)")
        p.add_argument("--density-normalize-mode", choices=["max","mean"], default=None, help="When density normalization enabled: choose reference as max or mean (ignored if explicit ref configured)")
        # Embedding options (mode=embed)
        p.add_argument("--fully-gen-root", default=None, help="Root for fully-conditioned DDPM generated NPZ sweep")
        p.add_argument("--semi-gen-root", default=None, help="Root for semi-conditioned DDPM generated NPZ sweep")
        p.add_argument("--fully-gt-root", default=None, help="Root for ground truth fully NPZs")
        p.add_argument("--semi-gt-root", default=None, help="Root for ground truth semi NPZs")
        p.add_argument("--embed-out", default=None, help="Output directory for embedding results")
        p.add_argument("--embed-max-per-group", type=int, default=None, help="Maximum samples per group (after GT merge)")
        p.add_argument("--embed-stride", type=int, default=None, help="Spatial stride (downsample) for density arrays")
        p.add_argument("--embed-no-standardize", action="store_true", help="Disable per-feature standardization prior to embedding")
        p.add_argument("--embed-pca", action="store_true", help="Force PCA even if UMAP is available")
        p.add_argument("--umap-neighbors", type=int, default=None, help="UMAP n_neighbors")
        p.add_argument("--umap-min-dist", type=float, default=None, help="UMAP min_dist")
        p.add_argument("--embed-seed", type=int, default=None, help="Random seed for subsampling & embedding")
        args = p.parse_args()

    vtu = getattr(args,'vtu',None) or DEFAULT_VTU
    vtu_field = getattr(args,'vtu_field',None) or DEFAULT_VTU_FIELD
    npz = getattr(args,'npz',None) or DEFAULT_NPZ
    npz_field = getattr(args,'npz_field',None) or DEFAULT_NPZ_FIELD
    out = getattr(args,'out',None) or DEFAULT_OUT
    mapping_mode = getattr(args,'mapping',None) or DEFAULT_MAPPING_MODE
    npz_y_origin = getattr(args,'npz_y_origin',None) or DEFAULT_NPZ_Y_ORIGIN
    outside_value = getattr(args,'outside_value',None) if getattr(args,'outside_value',None) is not None else 0.0
    mode = getattr(args,'mode',None) or DEFAULT_MODE
    if USE_INTERNAL_CONFIG:
        clean = bool(CONFIG.get('clean', True))
    else:
        clean = DEFAULT_CLEAN and (not getattr(args,'no_clean',False))

    def normalize_case_key(name: str) -> str:
        # Strip extension and take last path component
        base = os.path.splitext(name)[0]
        base = os.path.basename(base)
        # Remove any suffix starting with '_interpolated_arrays'
        idx = base.lower().find('_interpolated_arrays')
        if idx != -1:
            base = base[:idx]
        # Convert digit 'p' digit to decimal point (e.g., 0p011 -> 0.011)
        base = re.sub(r"(?<=\d)p(?=\d)", ".", base)
        # Normalize
        base = base.strip().lower()
        return base

    line_row = getattr(args,'line_row',None)
    line_y = getattr(args,'line_y',None)
    line_norm = bool(getattr(args,'line_normalize',False))

    def run_one(vtu_path: str, npz_path: str, outdir: str) -> Tuple[float, float, float, float, int, float, float, int, float]:
        # Load VTU first for bounds
        pts, vals, vtu_key = extract_vtu_field(vtu_path, vtu_field)
        xmin, xmax = float(np.nanmin(pts[:,0])), float(np.nanmax(pts[:,0]))
        ymin, ymax = float(np.nanmin(pts[:,1])), float(np.nanmax(pts[:,1]))
        Z_npz, X, Y = load_npz_grid(
            npz_path,
            npz_field,
            mapping_mode=mapping_mode,
            vtu_bounds=(xmin, xmax, ymin, ymax),
            npz_y_origin=npz_y_origin,
        )
        tri = get_vtu_triangulation(vtu_path)
        inside_mask = None
        if tri is not None:
            finder = tri.get_trifinder()
            tri_ids = finder(X, Y)
            inside_mask = tri_ids != -1
        Z_vtu = interpolate_to_grid(pts, vals, X, Y, inside_mask=inside_mask, outside_value=outside_value)

        # -----------------------------
        # Optional global density normalization
        # -----------------------------
        density_normalize = CONFIG.get('density_normalize', False) if 'CONFIG' in globals() else False
        density_normalize_ref = CONFIG.get('density_normalize_ref', None) if 'CONFIG' in globals() else None
        density_normalize_mode = (CONFIG.get('density_normalize_mode','max') if 'CONFIG' in globals() else 'max')
        # CLI override (only if not using internal config ref and user supplied mode)
        if not USE_INTERNAL_CONFIG:
            cli_mode = getattr(args,'density_normalize_mode',None)
            if cli_mode:
                density_normalize_mode = cli_mode
        field_is_density = npz_field.lower() in ("density", "rho")
        applied_norm_ref = None
        if density_normalize and field_is_density:
            if density_normalize_ref is None:
                if density_normalize_mode == 'mean':
                    ref_candidate = np.nanmean(Z_npz)
                else:
                    ref_candidate = np.nanmax(Z_npz)
                if not np.isfinite(ref_candidate) or ref_candidate == 0:
                    ref_candidate = 1.0
                applied_norm_ref = ref_candidate
            else:
                applied_norm_ref = float(density_normalize_ref)
                if applied_norm_ref == 0:
                    applied_norm_ref = 1.0
            Z_npz = Z_npz / applied_norm_ref
            Z_vtu = Z_vtu / applied_norm_ref
            # Inform user
            print(f"[NORM] Density normalized by {('explicit ref' if density_normalize_ref is not None else density_normalize_mode)} value {applied_norm_ref:.6g}")
            plot_field_label_npz = f"Norm. {npz_field}"
        else:
            plot_field_label_npz = npz_field

        # Pre-compute batch-aggregatable stats before plotting
        diff = Z_vtu - Z_npz
        # Valid means finite after operations
        valid_mask = np.isfinite(diff)
        n_valid = int(np.count_nonzero(valid_mask))
        sse = float(np.nansum(diff[valid_mask] ** 2)) if n_valid > 0 else float('nan')
        sum_npz_sq = float(np.nansum((Z_npz[valid_mask]) ** 2)) if n_valid > 0 else float('nan')

        # Mean Relative Error (percentage), robust to small denominators
        mre_eps = float(CONFIG.get('mre_ref_eps', 1e-12)) if 'CONFIG' in globals() else 1e-12
        denom = np.abs(Z_npz)
        mre_mask = valid_mask & (denom > mre_eps)
        n_mre = int(np.count_nonzero(mre_mask))
        if n_mre > 0:
            rel_abs = np.abs(diff[mre_mask]) / denom[mre_mask]
            sum_rel_abs = float(np.nansum(rel_abs))
            mre_pct = float(100.0 * (sum_rel_abs / n_mre))
        else:
            sum_rel_abs = float('nan')
            mre_pct = float('nan')

        mse, rel_mse, rmse_pct = plots(outdir, X, Y, Z_npz, Z_vtu, npz_field, vtu_key)
        # Optional line extraction
        try:
            mach_number = _extract_mach_from_text(vtu_path, npz_path, outdir)
            extract_line_profile(
                outdir,
                X,
                Y,
                Z_npz,
                Z_vtu,
                row_index=line_row,
                y_value=line_y,
                normalize=line_norm,
                field_label=plot_field_label_npz,
                mach_number=mach_number,
                style_like_sample=True,
            )
        except Exception as e:
            print(f"[LINE][WARN] Could not extract line profile: {e}")

        # Also create line plots for Temperature and Mach (using same row/params)
        def _make_field_line(npz_key: str, vtu_pref: str, pretty_label: str):
            try:
                pts_f, vals_f, _ = extract_vtu_field(vtu_path, vtu_pref)
                xmin_f, xmax_f = float(np.nanmin(pts_f[:,0])), float(np.nanmax(pts_f[:,0]))
                ymin_f, ymax_f = float(np.nanmin(pts_f[:,1])), float(np.nanmax(pts_f[:,1]))
                Z_npz_f, Xf, Yf = load_npz_grid(
                    npz_path,
                    npz_key,
                    mapping_mode=mapping_mode,
                    vtu_bounds=(xmin_f, xmax_f, ymin_f, ymax_f),
                    npz_y_origin=npz_y_origin,
                )
                tri_f = get_vtu_triangulation(vtu_path)
                inside_mask_f = None
                if tri_f is not None:
                    finder_f = tri_f.get_trifinder()
                    tri_ids_f = finder_f(Xf, Yf)
                    inside_mask_f = tri_ids_f != -1
                Z_vtu_f = interpolate_to_grid(pts_f, vals_f, Xf, Yf, inside_mask=inside_mask_f, outside_value=outside_value)
                extract_line_profile(
                    outdir,
                    Xf,
                    Yf,
                    Z_npz_f,
                    Z_vtu_f,
                    row_index=line_row,
                    y_value=line_y,
                    normalize=line_norm,
                    field_label=pretty_label,
                    mach_number=mach_number,
                    style_like_sample=True,
                )
            except Exception as _e:
                print(f"[LINE][WARN] Skipping line plot for field '{pretty_label}': {_e}")

        _make_field_line('temperature', 'Temperature', 'temperature')
        _make_field_line('mach', 'Mach', 'mach')
        return mse, rel_mse, rmse_pct, mre_pct, n_valid, sse, sum_npz_sq, n_mre, sum_rel_abs

    def compute_field_metrics(vtu_path: str, npz_path: str, npz_key: str, vtu_preference: str) -> Tuple[float, float, float, float, int, float, float, int, float]:
        """Compute metrics for a given field without plotting.

        Returns: (mse, rel_mse, rmse_pct, mre_pct, n_valid, sse, sum_npz_sq, n_mre, sum_rel_abs)
        """
        try:
            pts_f, vals_f, vtu_key_f = extract_vtu_field(vtu_path, vtu_preference)
        except Exception as e:
            print(f"[WARN] VTU field '{vtu_preference}' not found for {os.path.basename(vtu_path)}: {e}")
            return (float('nan'),) * 9
        # Use same bounds logic as run_one
        xmin_f, xmax_f = float(np.nanmin(pts_f[:,0])), float(np.nanmax(pts_f[:,0]))
        ymin_f, ymax_f = float(np.nanmin(pts_f[:,1])), float(np.nanmax(pts_f[:,1]))
        try:
            Z_npz_f, Xf, Yf = load_npz_grid(
                npz_path,
                npz_key,
                mapping_mode=mapping_mode,
                vtu_bounds=(xmin_f, xmax_f, ymin_f, ymax_f),
                npz_y_origin=npz_y_origin,
            )
        except Exception as e:
            print(f"[WARN] NPZ field '{npz_key}' not found for {os.path.basename(npz_path)}: {e}")
            return (float('nan'),) * 9
        tri_f = get_vtu_triangulation(vtu_path)
        inside_mask_f = None
        if tri_f is not None:
            finder_f = tri_f.get_trifinder()
            tri_ids_f = finder_f(Xf, Yf)
            inside_mask_f = tri_ids_f != -1
        Z_vtu_f = interpolate_to_grid(pts_f, vals_f, Xf, Yf, inside_mask=inside_mask_f, outside_value=outside_value)

        # Compute metrics (no normalization for non-density fields)
        diff_f = Z_vtu_f - Z_npz_f
        valid_f = np.isfinite(diff_f)
        n_valid_f = int(np.count_nonzero(valid_f))
        if n_valid_f > 0:
            sse_f = float(np.nansum(diff_f[valid_f] ** 2))
            sum_npz_sq_f = float(np.nansum((Z_npz_f[valid_f]) ** 2))
            mse_f = float(sse_f / n_valid_f)
        else:
            sse_f = float('nan')
            sum_npz_sq_f = float('nan')
            mse_f = float('nan')
        rmse_f = float(np.sqrt(mse_f)) if np.isfinite(mse_f) else float('nan')
        npz_range_f = float(np.nanmax(Z_npz_f) - np.nanmin(Z_npz_f)) if np.isfinite(Z_npz_f).any() else float('nan')
        rmse_pct_f = float(100.0 * rmse_f / npz_range_f) if (np.isfinite(npz_range_f) and npz_range_f > 0) else float('nan')
        rel_mse_f = float(sse_f / sum_npz_sq_f) if (sum_npz_sq_f and np.isfinite(sse_f)) else float('nan')

        # MRE%
        mre_eps = float(CONFIG.get('mre_ref_eps', 1e-12)) if 'CONFIG' in globals() else 1e-12
        denom_f = np.abs(Z_npz_f)
        mre_mask_f = valid_f & (denom_f > mre_eps)
        n_mre_f = int(np.count_nonzero(mre_mask_f))
        if n_mre_f > 0:
            rel_abs_f = np.abs(diff_f[mre_mask_f]) / denom_f[mre_mask_f]
            sum_rel_abs_f = float(np.nansum(rel_abs_f))
            mre_pct_f = float(100.0 * (sum_rel_abs_f / n_mre_f))
        else:
            sum_rel_abs_f = float('nan')
            mre_pct_f = float('nan')

        return mse_f, rel_mse_f, rmse_pct_f, mre_pct_f, n_valid_f, sse_f, sum_npz_sq_f, n_mre_f, sum_rel_abs_f

    # Embedding mode branch (early return)
    if mode == "embed":
        fully_gen_root = getattr(args, 'fully_gen_root', None) or DEFAULT_FULLY_GEN_ROOT
        semi_gen_root = getattr(args, 'semi_gen_root', None) or DEFAULT_SEMI_GEN_ROOT
        fully_gt_root = getattr(args, 'fully_gt_root', None) or DEFAULT_FULLY_GT_ROOT
        semi_gt_root = getattr(args, 'semi_gt_root', None) or DEFAULT_SEMI_GT_ROOT
        embed_out = getattr(args, 'embed_out', None) or DEFAULT_EMBED_OUT
        embed_max = getattr(args, 'embed_max_per_group', None) or DEFAULT_EMBED_MAX_PER_GROUP
        embed_stride = getattr(args, 'embed_stride', None) or DEFAULT_EMBED_STRIDE
        embed_standardize = not bool(getattr(args, 'embed_no_standardize', False)) if not USE_INTERNAL_CONFIG else CONFIG.get('embed_standardize', DEFAULT_EMBED_STANDARDIZE)
        embed_use_umap = (not bool(getattr(args, 'embed_pca', False))) and (getattr(args, 'embed_use_umap', True) if hasattr(args,'embed_use_umap') else DEFAULT_EMBED_USE_UMAP)
        umap_neighbors = getattr(args, 'umap_neighbors', None) or DEFAULT_UMAP_NEIGHBORS
        umap_min_dist = getattr(args, 'umap_min_dist', None) or DEFAULT_UMAP_MIN_DIST
        embed_seed = getattr(args, 'embed_seed', None) or DEFAULT_EMBED_SEED
        print("[EMBED] Running density embedding pipeline ...")
        run_density_embedding(
            fully_gen_root=fully_gen_root,
            semi_gen_root=semi_gen_root,
            fully_gt_root=fully_gt_root,
            semi_gt_root=semi_gt_root,
            out_dir=embed_out,
            max_per_group=embed_max,
            stride=embed_stride,
            standardize=embed_standardize,
            use_umap=embed_use_umap,
            umap_neighbors=umap_neighbors,
            umap_min_dist=umap_min_dist,
            seed=embed_seed,
            npz_y_origin=npz_y_origin,
            mapping_mode=mapping_mode,
        )
        print("[EMBED] Done.")
        return

    # Resolve roots with defaults (batch mode) for comparison functionality
    vtu_root = args.vtu_root or (DEFAULT_VTU_ROOT if os.path.isdir(DEFAULT_VTU_ROOT) else None)
    npz_root = args.npz_root or (DEFAULT_NPZ_ROOT if os.path.isdir(DEFAULT_NPZ_ROOT) else None)

    # Decide mode
    mode = mode or "auto"
    if mode == "batch" and not (vtu_root and npz_root):
        raise FileNotFoundError("Batch mode selected but VTU/NPZ roots are missing or invalid. Provide --vtu-root and --npz-root or ensure defaults exist.")
    if mode == "single" and (vtu_root and npz_root):
        # Force single even if roots exist
        vtu_root = None
        npz_root = None

    # Batch mode when roots are provided or chosen by auto
    if (mode == "batch") or (mode == "auto" and vtu_root and npz_root):
        vtu_root = os.path.abspath(vtu_root)
        npz_root = os.path.abspath(npz_root)
        out_root = os.path.abspath(args.out_root)
        # Clean output root if requested
        if clean and os.path.isdir(out_root):
            import shutil
            shutil.rmtree(out_root)
        os.makedirs(out_root, exist_ok=True)

        # Build NPZ map
        npz_map: Dict[str, str] = {}
        for dirpath, _, filenames in os.walk(npz_root):
            for fname in filenames:
                if fname.lower().endswith('.npz'):
                    key = normalize_case_key(fname)
                    npz_map[key] = os.path.join(dirpath, fname)

        results = []  # list of dicts per case with per-field metrics
        failures = []  # list of (case_key, vtu_path, npz_path_or_blank, reason)
        case_count = 0
        # Each immediate subdirectory of vtu_root is a case folder
        case_dirs = [os.path.join(vtu_root, d) for d in os.listdir(vtu_root) if os.path.isdir(os.path.join(vtu_root, d))]
        case_dirs.sort()
        total_case_dirs = len(case_dirs)
        for case_dir in case_dirs:
            case_name = os.path.basename(case_dir)
            key = normalize_case_key(case_name)
            vtu_path = os.path.join(case_dir, 'flow.vtu')
            if not os.path.isfile(vtu_path):
                reason = "crashed: no flow.vtu in case folder"
                print(f"[CRASH] {key}: {reason}")
                failures.append((key, vtu_path, "", reason))
                continue
            npz_path = npz_map.get(key)
            if not npz_path:
                # Try more aggressive normalization on directory name
                alt_key = normalize_case_key(case_name.replace('p', '.'))
                npz_path = npz_map.get(alt_key)
            if not npz_path:
                reason = f"no NPZ match for key '{key}'"
                print(f"[WARN] No NPZ match for VTU case: {case_dir} (key={key})")
                failures.append((key, vtu_path, "", reason))
                continue
            case_out = os.path.join(out_root, key)
            if clean and os.path.isdir(case_out):
                import shutil
                shutil.rmtree(case_out)
            try:
                # Primary field (density) with plots
                mse, rel_mse, rmse_pct, mre_pct, n_valid, sse, sum_npz_sq, n_mre, sum_rel_abs = run_one(vtu_path, npz_path, case_out)
                # Additional fields: temperature, mach (compute metrics only)
                t_mse, t_rel_mse, t_rmse_pct, t_mre_pct, t_n_valid, t_sse, t_sum_npz_sq, t_n_mre, t_sum_rel_abs = compute_field_metrics(
                    vtu_path, npz_path, 'temperature', 'Temperature'
                )
                m_mse, m_rel_mse, m_rmse_pct, m_mre_pct, m_n_valid, m_sse, m_sum_npz_sq, m_n_mre, m_sum_rel_abs = compute_field_metrics(
                    vtu_path, npz_path, 'mach', 'Mach'
                )
                row = {
                    'case_key': key,
                    'vtu_path': vtu_path,
                    'npz_path': npz_path,
                    # density metrics (kept in legacy generic columns)
                    'mse': mse,
                    'rel_mse': rel_mse,
                    'rmse_pct': rmse_pct,
                    'mre_pct': mre_pct,
                    'n_valid': n_valid,
                    'sse': sse,
                    'sum_npz_sq': sum_npz_sq,
                    'n_mre': n_mre,
                    'sum_rel_abs': sum_rel_abs,
                    # temperature metrics
                    'temperature_mse': t_mse,
                    'temperature_rel_mse': t_rel_mse,
                    'temperature_rmse_pct': t_rmse_pct,
                    'temperature_mre_pct': t_mre_pct,
                    'temperature_n_valid': t_n_valid,
                    'temperature_sse': t_sse,
                    'temperature_sum_npz_sq': t_sum_npz_sq,
                    'temperature_n_mre': t_n_mre,
                    'temperature_sum_rel_abs': t_sum_rel_abs,
                    # mach metrics
                    'mach_mse': m_mse,
                    'mach_rel_mse': m_rel_mse,
                    'mach_rmse_pct': m_rmse_pct,
                    'mach_mre_pct': m_mre_pct,
                    'mach_n_valid': m_n_valid,
                    'mach_sse': m_sse,
                    'mach_sum_npz_sq': m_sum_npz_sq,
                    'mach_n_mre': m_n_mre,
                    'mach_sum_rel_abs': m_sum_rel_abs,
                }
                results.append(row)
                case_count += 1
                print(f"[OK] {key}: density MSE={mse:.6e}, relMSE={rel_mse:.6e}, RMSE%={rmse_pct:.3f}")
            except Exception as e:
                print(f"[FAIL] {key}: {e}")
                failures.append((key, vtu_path, npz_path, str(e)))

        # Write summary CSV
        summary_path = os.path.abspath(args.summary_csv)
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = [
                'case_key', 'vtu_path', 'npz_path', 'npz_field', 'vtu_field', 'mapping', 'npz_y_origin', 'outside_value',
                # density (legacy generic columns)
                'mse', 'rel_mse', 'rmse_pct', 'mre_pct',
                # temperature
                'temperature_mse', 'temperature_rel_mse', 'temperature_rmse_pct', 'temperature_mre_pct',
                # mach
                'mach_mse', 'mach_rel_mse', 'mach_rmse_pct', 'mach_mre_pct',
            ]
            writer.writerow(header)
            for row in results:
                writer.writerow([
                    row['case_key'], row['vtu_path'], row['npz_path'], npz_field, vtu_field, mapping_mode, npz_y_origin, outside_value,
                    f"{row['mse']:.6e}", f"{row['rel_mse']:.6e}", f"{row['rmse_pct']:.3f}", f"{row['mre_pct']:.3f}",
                    f"{row.get('temperature_mse', float('nan')):.6e}", f"{row.get('temperature_rel_mse', float('nan')):.6e}", f"{row.get('temperature_rmse_pct', float('nan')):.3f}", f"{row.get('temperature_mre_pct', float('nan')):.3f}",
                    f"{row.get('mach_mse', float('nan')):.6e}", f"{row.get('mach_rel_mse', float('nan')):.6e}", f"{row.get('mach_rmse_pct', float('nan')):.3f}", f"{row.get('mach_mre_pct', float('nan')):.3f}",
                ])
            # Append a summary section
            writer.writerow([])
            writer.writerow(['SUMMARY'])
            writer.writerow(['total_case_folders', total_case_dirs])
            writer.writerow(['successes', len(results)])
            writer.writerow(['failures', len(failures)])
            # Batch-level aggregate errors (computed over successes only)
            if results:
                # Helper to compute aggregates per key prefix
                def aggregates_for(prefix: str):
                    mselist = np.array([row.get(f'{prefix}mse') for row in results], dtype=float)
                    rel_mselist = np.array([row.get(f'{prefix}rel_mse') for row in results], dtype=float)
                    rmsepct_list = np.array([row.get(f'{prefix}rmse_pct') for row in results], dtype=float)
                    mre_pct_list = np.array([row.get(f'{prefix}mre_pct') for row in results], dtype=float)
                    # Derive non-percent MRE list (divide by 100 where finite). This keeps legacy pct metrics while exposing raw ratio.
                    with np.errstate(invalid='ignore', divide='ignore'):
                        mre_list = mre_pct_list / 100.0
                    n_valid_total = float(np.nansum([row.get(f'{prefix}n_valid', 0.0) for row in results]))
                    sse_total = float(np.nansum([row.get(f'{prefix}sse', 0.0) for row in results]))
                    sum_npz_sq_total = float(np.nansum([row.get(f'{prefix}sum_npz_sq', 0.0) for row in results]))
                    n_mre_total = float(np.nansum([row.get(f'{prefix}n_mre', 0.0) for row in results]))
                    sum_rel_abs_total = float(np.nansum([row.get(f'{prefix}sum_rel_abs', 0.0) for row in results]))
                    macro_mse_mean = float(np.nanmean(mselist))
                    macro_mse_median = float(np.nanmedian(mselist))
                    macro_mse_std = float(np.nanstd(mselist))
                    macro_rel_mse_mean = float(np.nanmean(rel_mselist))
                    macro_rmse_pct_mean = float(np.nanmean(rmsepct_list))
                    macro_mre_pct_mean = float(np.nanmean(mre_pct_list))
                    macro_mre_mean = float(np.nanmean(mre_list))
                    micro_mse = float(sse_total / n_valid_total) if n_valid_total and np.isfinite(sse_total) else float('nan')
                    micro_rmse = float(np.sqrt(micro_mse)) if np.isfinite(micro_mse) else float('nan')
                    micro_rel_mse = float(sse_total / sum_npz_sq_total) if sum_npz_sq_total and np.isfinite(sse_total) else float('nan')
                    micro_mre_pct = float(100.0 * (sum_rel_abs_total / n_mre_total)) if n_mre_total and np.isfinite(sum_rel_abs_total) else float('nan')
                    micro_mre = float(sum_rel_abs_total / n_mre_total) if n_mre_total and np.isfinite(sum_rel_abs_total) else float('nan')
                    return dict(
                        macro_mse_mean=macro_mse_mean,
                        macro_mse_median=macro_mse_median,
                        macro_mse_std=macro_mse_std,
                        macro_rel_mse_mean=macro_rel_mse_mean,
                        macro_rmse_pct_mean=macro_rmse_pct_mean,
                        macro_mre_pct_mean=macro_mre_pct_mean,
                        macro_mre_mean=macro_mre_mean,
                        micro_mse=micro_mse,
                        micro_rmse=micro_rmse,
                        micro_rel_mse=micro_rel_mse,
                        micro_mre=micro_mre,
                        micro_mre_pct=micro_mre_pct,
                    )

                # Density keeps legacy names (no prefix)
                agg_density = aggregates_for('')
                agg_temp = aggregates_for('temperature_')
                agg_mach = aggregates_for('mach_')

                writer.writerow([])
                writer.writerow(['BATCH ERROR METRICS'])
                # Density (legacy labels)
                writer.writerow(['macro_mse_mean', f"{agg_density['macro_mse_mean']:.6e}"])
                writer.writerow(['macro_mse_median', f"{agg_density['macro_mse_median']:.6e}"])
                writer.writerow(['macro_mse_std', f"{agg_density['macro_mse_std']:.6e}"])
                writer.writerow(['macro_rel_mse_mean', f"{agg_density['macro_rel_mse_mean']:.6e}"])
                writer.writerow(['macro_rmse_pct_mean', f"{agg_density['macro_rmse_pct_mean']:.3f}"])
                writer.writerow(['macro_mre_mean', f"{agg_density['macro_mre_mean']:.6e}"])
                writer.writerow(['macro_mre_pct_mean', f"{agg_density['macro_mre_pct_mean']:.3f}"])
                writer.writerow(['micro_mse', f"{agg_density['micro_mse']:.6e}"])
                writer.writerow(['micro_rmse', f"{agg_density['micro_rmse']:.6e}"])
                writer.writerow(['micro_rel_mse', f"{agg_density['micro_rel_mse']:.6e}"])
                writer.writerow(['micro_mre', f"{agg_density['micro_mre']:.6e}"])
                writer.writerow(['micro_mre_pct', f"{agg_density['micro_mre_pct']:.3f}"])
                # Temperature
                writer.writerow(['macro_mse_mean_temperature', f"{agg_temp['macro_mse_mean']:.6e}"])
                writer.writerow(['macro_mse_median_temperature', f"{agg_temp['macro_mse_median']:.6e}"])
                writer.writerow(['macro_mse_std_temperature', f"{agg_temp['macro_mse_std']:.6e}"])
                writer.writerow(['macro_rel_mse_mean_temperature', f"{agg_temp['macro_rel_mse_mean']:.6e}"])
                writer.writerow(['macro_rmse_pct_mean_temperature', f"{agg_temp['macro_rmse_pct_mean']:.3f}"])
                writer.writerow(['macro_mre_mean_temperature', f"{agg_temp['macro_mre_mean']:.6e}"])
                writer.writerow(['macro_mre_pct_mean_temperature', f"{agg_temp['macro_mre_pct_mean']:.3f}"])
                writer.writerow(['micro_mse_temperature', f"{agg_temp['micro_mse']:.6e}"])
                writer.writerow(['micro_rmse_temperature', f"{agg_temp['micro_rmse']:.6e}"])
                writer.writerow(['micro_rel_mse_temperature', f"{agg_temp['micro_rel_mse']:.6e}"])
                writer.writerow(['micro_mre_temperature', f"{agg_temp['micro_mre']:.6e}"])
                writer.writerow(['micro_mre_pct_temperature', f"{agg_temp['micro_mre_pct']:.3f}"])
                # Mach
                writer.writerow(['macro_mse_mean_mach', f"{agg_mach['macro_mse_mean']:.6e}"])
                writer.writerow(['macro_mse_median_mach', f"{agg_mach['macro_mse_median']:.6e}"])
                writer.writerow(['macro_mse_std_mach', f"{agg_mach['macro_mse_std']:.6e}"])
                writer.writerow(['macro_rel_mse_mean_mach', f"{agg_mach['macro_rel_mse_mean']:.6e}"])
                writer.writerow(['macro_rmse_pct_mean_mach', f"{agg_mach['macro_rmse_pct_mean']:.3f}"])
                writer.writerow(['macro_mre_mean_mach', f"{agg_mach['macro_mre_mean']:.6e}"])
                writer.writerow(['macro_mre_pct_mean_mach', f"{agg_mach['macro_mre_pct_mean']:.3f}"])
                writer.writerow(['micro_mse_mach', f"{agg_mach['micro_mse']:.6e}"])
                writer.writerow(['micro_rmse_mach', f"{agg_mach['micro_rmse']:.6e}"])
                writer.writerow(['micro_rel_mse_mach', f"{agg_mach['micro_rel_mse']:.6e}"])
                writer.writerow(['micro_mre_mach', f"{agg_mach['micro_mre']:.6e}"])
                writer.writerow(['micro_mre_pct_mach', f"{agg_mach['micro_mre_pct']:.3f}"])
            # Append failed cases section (always, even if none)
            writer.writerow([])
            writer.writerow(['FAILED CASES'])
            writer.writerow(['case_key', 'vtu_path', 'npz_path', 'reason'])
            if failures:
                for key, vtu_path, npz_path, reason in failures:
                    writer.writerow([key, vtu_path, npz_path, reason])
            else:
                writer.writerow(['none', '', '', 'no failures'])
        print(f"Batch complete. Success: {case_count}, Failures: {len(failures)}. Summary: {summary_path}. Outputs: {out_root}")
        return

    # Single-case mode
    if not os.path.isfile(vtu):
        raise FileNotFoundError(f"VTU not found: {vtu}")
    if not os.path.isfile(npz):
        raise FileNotFoundError(f"NPZ not found: {npz}")

    # Single-case: clean output dir if requested
    if clean and os.path.isdir(out):
        import shutil
        shutil.rmtree(out)
    mse, rel_mse, rmse_pct, *_ = run_one(vtu, npz, out)
    print(f"Done. Wrote comparison to: {out}. MSE={mse:.6e}, relMSE={rel_mse:.6e}, RMSE%={rmse_pct:.3f}")


if __name__ == "__main__":
    main()
