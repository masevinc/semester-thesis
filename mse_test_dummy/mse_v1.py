#!/usr/bin/env python3
"""
Test script to compare flow fields between a regular-grid NPZ dataset
and an unstructured VTU dataset.

- Reads NPZ field (e.g., temperature)
- Reads VTU field (e.g., Temperature)
- Interpolates VTU onto NPZ grid
- Masks geometry pixels
- Computes MSE and produces plots
"""

import os
import re
import glob
import numpy as np
import meshio
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

# -------------------------
# USER SETTINGS
# -------------------------
npz_path = "double_ramp_configuration/inputs/double_ramp_npz_files_clamped/double_ramp_0.011_0.0488_ma_2.892_pres_199070_interpolated_arrays.npz"        # Path to your .npz file
npz_field = "temperature"               # Field key in NPZ file

vtu_path = "double_ramp_configuration/outputs/backward/sweep/double_ramp_0p011_0p0488_ma_2p892_pres_199070_interpolated_arrays_density_M2p892_T300p0_P199070p0/flow.vtu"        # Path to your .vtu file
vtu_field = "Temperature"               # Field key in VTU file

mask_threshold = 1e-12                  # Zero cutoff for ramp geometry
outdir = "outputs_test"                 # Directory for outputs
# -------------------------


def load_npz_grid(npz_path: str, field_key: str):
    data = np.load(npz_path, allow_pickle=True)

    if field_key not in data:
        raise KeyError(f"Field '{field_key}' not found in NPZ. Keys: {list(data.keys())}")

    field = data[field_key].astype(float)
    ny, nx = field.shape

    x = data.get('x', data.get('x_coords', None))
    y = data.get('y', data.get('y_coords', None))

    if x is None or y is None:
        x = np.linspace(0.5 / nx, 1 - 0.5 / nx, nx)
        y = np.linspace(0.5 / ny, 1 - 0.5 / ny, ny)
    else:
        x = np.asarray(x).squeeze()
        y = np.asarray(y).squeeze()
        # Handle possible 2D grid inputs or length mismatch
        if x.ndim > 1:
            x = x.ravel()
        if y.ndim > 1:
            y = y.ravel()
        if x.size != nx:
            # Fallback to uniform spacing across x
            x = np.linspace(0.5 / nx, 1 - 0.5 / nx, nx)
        if y.size != ny:
            # Fallback to uniform spacing across y
            y = np.linspace(0.5 / ny, 1 - 0.5 / ny, ny)

    # Ensure ascending axis order and reorder field accordingly
    idx_x = np.argsort(x)
    if not np.all(idx_x == np.arange(x.size)):
        field = field[:, idx_x]
        x = x[idx_x]

    idx_y = np.argsort(y)
    if not np.all(idx_y == np.arange(y.size)):
        field = field[idx_y, :]
        y = y[idx_y]

    X, Y = np.meshgrid(x, y)
    return field, X, Y


def _sanitize_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _choose_key(keys, preferred: str):
    """Pick a key from keys matching preferred, with fallbacks (case/space-insensitive, contains 'temp')."""
    if not keys:
        return None
    if preferred in keys:
        return preferred
    # case-insensitive exact
    for k in keys:
        if k.lower() == preferred.lower():
            return k
    # sanitized exact
    pref_s = _sanitize_key(preferred)
    for k in keys:
        if _sanitize_key(k) == pref_s:
            return k
    # contains 'temp'
    for k in keys:
        if "temp" in k.lower():
            return k
    # fallback to first
    return list(keys)[0]


def _candidate_keys(keys, preferred: str):
    """Return keys ordered by preference: exact/sanitized match, then any containing 'temp', then the rest."""
    if not keys:
        return []
    keys = list(keys)
    exact = []
    sanitized = []
    temp_like = []
    others = []
    pref_s = _sanitize_key(preferred)
    for k in keys:
        if k == preferred or k.lower() == preferred.lower():
            exact.append(k)
        elif _sanitize_key(k) == pref_s:
            sanitized.append(k)
        elif any(tok in k.lower() for tok in ["temp", "temperature", "t"]):
            temp_like.append(k)
        else:
            others.append(k)
    # Deduplicate while preserving order
    seen = set()
    out = []
    for group in (exact, sanitized, temp_like, others):
        for k in group:
            if k not in seen:
                seen.add(k)
                out.append(k)
    return out


def _is_nonconstant(a: np.ndarray, eps: float = 1e-9) -> bool:
    a = np.asarray(a)
    if a.size == 0:
        return False
    return float(np.nanstd(a)) > eps


def _stats(a: np.ndarray):
    a = np.asarray(a)
    return float(np.nanmin(a)), float(np.nanmax(a)), float(np.nanstd(a))


def extract_vtu_field(vtu_path: str, field_key: str):
    """Read a VTU file and return (xy_points, values) for the requested field.

    Tries meshio, then pyvista, then VTK. Handles point_data and cell_data.
    """
    # 1) meshio
    try:
        m = meshio.read(vtu_path)
    except Exception:
        m = None
    if m is not None:
        points = m.points[:, :2]
        # Try point_data candidates, prefer non-constant arrays
        pd_keys = list(m.point_data.keys())
        for k in _candidate_keys(pd_keys, field_key):
            arr = np.asarray(m.point_data[k]).astype(float).ravel()
            if _is_nonconstant(arr):
                return points, arr
        # Try cell_data candidates, prefer non-constant arrays
        cd_keys = list(m.cell_data.keys())
        for k in _candidate_keys(cd_keys, field_key):
            values, centers = [], []
            per_block_arrays = m.cell_data[k]
            for cells_block, cell_vals in zip(m.cells, per_block_arrays):
                cell_vals = np.asarray(cell_vals).astype(float).ravel()
                for conn in cells_block.data:
                    xy = m.points[conn, :2]
                    centers.append(xy.mean(axis=0))
                values.append(cell_vals)
            if values:
                centers = np.asarray(centers)
                values = np.concatenate(values)
                if _is_nonconstant(values):
                    return centers, values

    # 2) pyvista
    try:
        import pyvista as pv  # type: ignore
        pv_mesh = pv.read(vtu_path)
        pts = np.asarray(pv_mesh.points)[:, :2]
        pd_keys = list(pv_mesh.point_data.keys())
        for k in _candidate_keys(pd_keys, field_key):
            vals = np.asarray(pv_mesh.point_data[k]).astype(float).ravel()
            if _is_nonconstant(vals):
                return pts, vals
        cd_keys = list(pv_mesh.cell_data.keys())
        for k in _candidate_keys(cd_keys, field_key):
            vals = np.asarray(pv_mesh.cell_data[k]).astype(float).ravel()
            centers = np.asarray(pv_mesh.cell_centers().points)[:, :2]
            if _is_nonconstant(vals):
                return centers, vals
    except Exception:
        pass

    # 3) VTK
    try:
        import vtk  # type: ignore
        from vtk.util.numpy_support import vtk_to_numpy  # type: ignore

        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(vtu_path)
        reader.Update()
        ug = reader.GetOutput()
        vtk_points = ug.GetPoints()
        if vtk_points is None:
            raise RuntimeError("No points found in VTU (VTK)")
        pts_np = vtk_to_numpy(vtk_points.GetData())[:, :2]

        pd = ug.GetPointData()
        pd_keys = [pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]
        for k in _candidate_keys(pd_keys, field_key):
            arr = pd.GetArray(k)
            if arr is not None:
                vals_np = vtk_to_numpy(arr).astype(float).ravel()
                if _is_nonconstant(vals_np):
                    return pts_np, vals_np

        cd = ug.GetCellData()
        cd_keys = [cd.GetArrayName(i) for i in range(cd.GetNumberOfArrays())]
        for k in _candidate_keys(cd_keys, field_key):
            arr = cd.GetArray(k)
            if arr is not None:
                vals_np = vtk_to_numpy(arr).astype(float).ravel()
                centers = []
                n_cells = ug.GetNumberOfCells()
                for i in range(n_cells):
                    cell = ug.GetCell(i)
                    ids = [cell.GetPointId(j) for j in range(cell.GetNumberOfPoints())]
                    xy = pts_np[ids]
                    centers.append(xy.mean(axis=0))
                if _is_nonconstant(vals_np):
                    return np.asarray(centers), vals_np
    except Exception:
        pass

    raise KeyError(f"Field '{field_key}' not found in VTU: {vtu_path}")


def align_vtu_orientation(pts_xy: np.ndarray, grid_X: np.ndarray, grid_Y: np.ndarray) -> np.ndarray:
    """Currently a no-op; kept for possible flips if needed later."""
    return pts_xy


def resolve_vtu_path(path_in: str) -> str:
    """Prefer a solution/restart VTU over a static flow.vtu if available."""
    d = os.path.dirname(path_in)
    base = os.path.basename(path_in).lower()
    candidates = []
    # Only search if the given file looks like a static setup
    if base == "flow.vtu":
        patterns = ["solution*.vtu", "restart*.vtu", "flow_*.vtu"]
        for pat in patterns:
            candidates.extend(glob.glob(os.path.join(d, pat)))
    # If we found candidates, pick the one with largest trailing number, else latest mtime
    def extract_num(p):
        m = re.search(r"(\d+)(?=\.vtu$)", os.path.basename(p))
        return int(m.group(1)) if m else -1
    if candidates:
        candidates.sort(key=lambda p: (extract_num(p), os.path.getmtime(p)))
        return candidates[-1]
    return path_in


def write_vtu_stats(outdir: str, meta: dict, values: np.ndarray, label: str, path: str):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "vtu_field_stats.txt"), "a") as f:
        f.write(f"{label}: {path}\n")
        if meta:
            f.write(f"  reader={meta.get('reader')} location={meta.get('location')} field={meta.get('field')}\n")
        f.write(f"  min={np.nanmin(values):.6g} max={np.nanmax(values):.6g} std={np.nanstd(values):.6g}\n\n")


def interpolate_to_grid(pts_xy, vals, grid_X, grid_Y):
    Zi = griddata(pts_xy, vals, (grid_X, grid_Y), method='linear')
    nan_mask = np.isnan(Zi)
    if np.any(nan_mask):
        Zi[nan_mask] = griddata(pts_xy, vals, (grid_X[nan_mask], grid_Y[nan_mask]), method='nearest')
    return Zi


def compute_mask(npz_field, threshold):
    return np.isfinite(npz_field) & (np.abs(npz_field) > threshold)


def mse(a, b, mask):
    return float(np.mean((a[mask] - b[mask]) ** 2))


def quicklook_plots(npz_field, vtu_on_grid, mask, outdir, field_key, X, Y):
    os.makedirs(outdir, exist_ok=True)
    diff = np.full_like(npz_field, np.nan)
    diff[mask] = vtu_on_grid[mask] - npz_field[mask]

    def save_im(arr, title, fname):
        plt.figure(figsize=(6, 5))
        extent = [float(X.min()), float(X.max()), float(Y.min()), float(Y.max())]
        im = plt.imshow(arr, origin='lower', aspect='equal', extent=extent)
        plt.title(title)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, fname), dpi=200)
        plt.close()

    save_im(npz_field, f"NPZ {field_key}", f"npz_{field_key}.png")
    save_im(vtu_on_grid, f"VTU→grid {field_key}", f"vtu_{field_key}_on_grid.png")
    save_im(diff, f"Difference (VTU - NPZ) {field_key}", f"diff_{field_key}.png")


def save_debug_matrices(outdir, npz_field, vtu_on_grid, X, Y, mask):
    """Save the two data matrices and related grids for manual comparison."""
    os.makedirs(outdir, exist_ok=True)
    # Compact binary for fidelity and quick reload
    np.savez_compressed(
        os.path.join(outdir, "debug_fields.npz"),
        npz_field=npz_field,
        vtu_on_grid=vtu_on_grid,
        X=X,
        Y=Y,
        mask=mask,
    )
    # CSVs for quick spreadsheet inspection
    np.savetxt(os.path.join(outdir, "npz_field.csv"), npz_field, delimiter=",")
    np.savetxt(os.path.join(outdir, "vtu_on_grid.csv"), vtu_on_grid, delimiter=",")


def save_vtu_sample(outdir, pts_xy, vals, max_rows: int = 2000):
    os.makedirs(outdir, exist_ok=True)
    n = min(len(vals), max_rows)
    sample = np.column_stack([pts_xy[:n, 0], pts_xy[:n, 1], vals[:n]])
    np.savetxt(os.path.join(outdir, "vtu_points_values_sample.csv"), sample, delimiter=",", header="x,y,value", comments="")


if __name__ == "__main__":
    npz_field_data, X, Y = load_npz_grid(npz_path, npz_field)
    mask = compute_mask(npz_field_data, mask_threshold)

    # Prefer a converged solution file if available
    vtu_path_use = resolve_vtu_path(vtu_path)
    if vtu_path_use != vtu_path:
        print(f"Resolved VTU file: {vtu_path_use}")

    pts_xy, vtu_vals = extract_vtu_field(vtu_path_use, vtu_field)
    # Write basic stats and a small sample for debugging (min/max/std)
    write_vtu_stats(outdir, {"reader": "auto", "location": "unknown", "field": vtu_field}, vtu_vals, label="chosen_vtu_field", path=vtu_path_use)
    save_vtu_sample(outdir, pts_xy, vtu_vals, max_rows=5000)
    # Align VTU points orientation to NPZ grid if necessary
    pts_xy = align_vtu_orientation(pts_xy, X, Y)
    vtu_on_grid = interpolate_to_grid(pts_xy, vtu_vals, X, Y)

    # Save matrices for manual debugging BEFORE computing MSE
    save_debug_matrices(outdir, npz_field_data, vtu_on_grid, X, Y, mask)

    value_mse = mse(npz_field_data, vtu_on_grid, mask)
    print(f"MSE (masked) = {value_mse:.6e}")

    quicklook_plots(npz_field_data, vtu_on_grid, mask, outdir, npz_field, X, Y)

    with open(os.path.join(outdir, "report.txt"), "w") as f:
        f.write(f"Field compared: NPZ '{npz_field}' vs VTU '{vtu_field}'\n")
        f.write(f"MSE (masked): {value_mse:.6e}\n")
