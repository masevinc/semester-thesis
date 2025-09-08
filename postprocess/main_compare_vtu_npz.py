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


# Defaults
DEFAULT_MAPPING_MODE = "auto"  # 'auto' | 'npz' | 'vtu'
DEFAULT_NPZ_Y_ORIGIN = "top"    # 'top' (image-style, like your main code) or 'bottom' (math-style)
# for batch mode
DEFAULT_VTU_ROOT = "double_ramp_configuration/outputs/backward/sweep"
DEFAULT_NPZ_ROOT = "double_ramp_configuration/inputs/double_ramp_npz_files_clamped"
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


def _sanitize_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


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
    save_im(Z_npz, f"NPZ: {field_npz}", "npz.png", vmin=vmin_shared, vmax=vmax_shared, cmap='viridis')
    save_im(Z_vtu, f"VTU→grid: {field_vtu}", "vtu_on_grid.png", vmin=vmin_shared, vmax=vmax_shared, cmap='viridis')

    # Diff: symmetric range around 0 with diverging colormap
    diff_abs = float(np.nanmax(np.abs(diff))) if np.isfinite(diff).any() else 1.0
    save_im(diff, "Diff (CFD - Diffusion Model)", "diff.png", vmin=-diff_abs, vmax=diff_abs, cmap='coolwarm')

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
    axes[0].set_title(f"NPZ: {field_npz}")
    im1 = axes[1].imshow(Z_vtu, origin='lower', aspect='equal', extent=extent, vmin=vmin_shared, vmax=vmax_shared, cmap='viridis')
    axes[1].set_title(f"VTU→grid: {field_vtu}")
    im2 = axes[2].imshow(diff, origin='lower', aspect='equal', extent=extent, vmin=-diff_abs, vmax=diff_abs, cmap='coolwarm')
    axes[2].set_title("Diff (VTU - NPZ)")
    # Colorbars
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    fig.savefig(os.path.join(outdir, "comparison_subplot.png"), dpi=220)
    plt.close(fig)

    return mse, rel_mse, rmse_pct


def main():
    p = argparse.ArgumentParser(description="Compare VTU field with NPZ field.")
    p.add_argument("--vtu", default=None)
    p.add_argument("--vtu-field", default=None)
    p.add_argument("--npz", default=None)
    p.add_argument("--npz-field", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--mapping", choices=["auto","npz","vtu"], default=None, help="How to build NPZ grid: use NPZ coords, VTU bounds, or auto-detect")
    p.add_argument("--npz-y-origin", choices=["top","bottom"], default=None, help="Treat NPZ row 0 as 'top' (image) or 'bottom' (math)")
    p.add_argument("--outside-value", type=float, default=None, help="Value to assign outside the VTU mesh domain on the grid (default 0)")
    p.add_argument("--mode", choices=["auto","single","batch"], default=None, help="Run a single case, batch over roots, or auto-detect")
    p.add_argument("--no-clean", action="store_true", help="Do not delete existing output folder(s) before writing")
    # Batch mode options
    p.add_argument("--vtu-root", default=None, help="Root folder to search for VTU cases (recursively, looking for flow.vtu)")
    p.add_argument("--npz-root", default=None, help="Root folder to search for NPZ files (recursively, .npz)")
    p.add_argument("--out-root", default="postprocess_outputs/compare_batch", help="Output root for batch results (each case gets its own folder)")
    p.add_argument("--summary-csv", default="postprocess_outputs/compare_batch_summary.csv", help="CSV file to write per-case MSE summary")
    args = p.parse_args()

    vtu = args.vtu or DEFAULT_VTU
    vtu_field = args.vtu_field or DEFAULT_VTU_FIELD
    npz = args.npz or DEFAULT_NPZ
    npz_field = args.npz_field or DEFAULT_NPZ_FIELD
    out = args.out or DEFAULT_OUT
    mapping_mode = args.mapping or DEFAULT_MAPPING_MODE
    npz_y_origin = args.npz_y_origin or DEFAULT_NPZ_Y_ORIGIN
    outside_value = args.outside_value if args.outside_value is not None else 0.0
    mode = args.mode or DEFAULT_MODE
    clean = DEFAULT_CLEAN and (not args.no_clean)

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

    def run_one(vtu_path: str, npz_path: str, outdir: str) -> Tuple[float, float, float]:
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
        mse, rel_mse, rmse_pct = plots(outdir, X, Y, Z_npz, Z_vtu, npz_field, vtu_key)
        return mse, rel_mse, rmse_pct

    # Resolve roots with defaults (batch mode)
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

        results = []  # list of (case_key, vtu_path, npz_path, mse, rel_mse, rmse_pct)
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
                mse, rel_mse, rmse_pct = run_one(vtu_path, npz_path, case_out)
                results.append((key, vtu_path, npz_path, mse, rel_mse, rmse_pct))
                case_count += 1
                print(f"[OK] {key}: MSE={mse:.6e}, relMSE={rel_mse:.6e}, RMSE%={rmse_pct:.3f}")
            except Exception as e:
                print(f"[FAIL] {key}: {e}")
                failures.append((key, vtu_path, npz_path, str(e)))

        # Write summary CSV
        summary_path = os.path.abspath(args.summary_csv)
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['case_key', 'vtu_path', 'npz_path', 'npz_field', 'vtu_field', 'mapping', 'npz_y_origin', 'outside_value', 'mse', 'rel_mse', 'rmse_pct'])
            for key, vtu_path, npz_path, mse, rel_mse, rmse_pct in results:
                writer.writerow([key, vtu_path, npz_path, npz_field, vtu_field, mapping_mode, npz_y_origin, outside_value, f"{mse:.6e}", f"{rel_mse:.6e}", f"{rmse_pct:.3f}"])
            # Append a summary section
            writer.writerow([])
            writer.writerow(['SUMMARY'])
            writer.writerow(['total_case_folders', total_case_dirs])
            writer.writerow(['successes', len(results)])
            writer.writerow(['failures', len(failures)])
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
    mse, rel_mse, rmse_pct = run_one(vtu, npz, out)
    print(f"Done. Wrote comparison to: {out}. MSE={mse:.6e}, relMSE={rel_mse:.6e}, RMSE%={rmse_pct:.3f}")


if __name__ == "__main__":
    main()
