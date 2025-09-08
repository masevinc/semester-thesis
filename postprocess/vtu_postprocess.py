#!/usr/bin/env python3
"""
VTU post-processor for SU2 outputs.

Features:
- Read a .vtu, list available fields (point and cell data)
- Extract a chosen field (e.g., Density / Rho) from point or cell data
- Save a CSV (x, y, value) and quick plots (tricontour and histogram)

Dependencies: meshio, matplotlib, numpy
Optional (not required): pyvista for advanced plotting (not used by default)

Examples:
    # Run with CLI args
    python postprocess/vtu_postprocess.py \
        --vtu single_ramp_configuration/youtube_case/ramp_analysis_su2v8_v1/flow.vtu \
        --field density \
        --out postprocess_outputs/example_density

    # Or rely on the defaults defined below (no args)
    python postprocess/vtu_postprocess.py
"""

from __future__ import annotations

import argparse
import os
import re
from typing import List, Optional, Sequence, Tuple

import numpy as np
import meshio
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation


# -------------------------
# Utility helpers
# -------------------------


def _sanitize_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def candidate_keys(keys: Sequence[str], preferred: str, extra_tokens: Optional[Sequence[str]] = None) -> List[str]:
    """Return keys ordered by preference.

    Priority:
    - exact (case-insensitive)
    - sanitized exact
    - contains any token in extra_tokens (case-insensitive)
    - others
    """
    if not keys:
        return []
    keys = list(keys)
    exact: List[str] = []
    sanitized: List[str] = []
    token_like: List[str] = []
    others: List[str] = []
    pref_s = _sanitize_key(preferred)
    tokens = [t.lower() for t in (extra_tokens or [])]
    for k in keys:
        if k.lower() == preferred.lower():
            exact.append(k)
        elif _sanitize_key(k) == pref_s:
            sanitized.append(k)
        elif any(t in k.lower() for t in tokens):
            token_like.append(k)
        else:
            others.append(k)
    # Deduplicate while preserving order
    out: List[str] = []
    seen = set()
    for group in (exact, sanitized, token_like, others):
        for k in group:
            if k not in seen:
                seen.add(k)
                out.append(k)
    return out


def is_nonconstant(a: np.ndarray, eps: float = 1e-12) -> bool:
    if a.size == 0:
        return False
    return float(np.nanstd(a)) > eps


def meshio_triangulation(m: meshio.Mesh) -> Optional[Triangulation]:
    """Build a Matplotlib Triangulation from meshio cells.

    - Uses triangle cells directly
    - Splits quads into two triangles (0-1-2, 0-2-3)
    - Ignores other cell types for 2D plot
    """
    tris: List[Tuple[int, int, int]] = []
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
        # Other 2D types could be handled here if needed.
    if not tris:
        return None
    pts = m.points[:, :2]
    tri = Triangulation(pts[:, 0], pts[:, 1], np.asarray(tris))
    return tri


def extract_field_meshio(m: meshio.Mesh, field_preference: str) -> Tuple[np.ndarray, np.ndarray, str, str]:
    """Try to extract a field from meshio point_data or cell_data.

    Returns:
      (pts_xy, values, location, chosen_key)
    where location is "point" or "cell".
    """
    # Try point data first
    pd_keys = list(m.point_data.keys())
    for key in candidate_keys(pd_keys, field_preference, extra_tokens=["rho", "density"]):
        vals = np.asarray(m.point_data[key]).astype(float).ravel()
        if is_nonconstant(vals):
            pts = m.points[:, :2]
            return pts, vals, "point", key

    # Then cell data
    # In meshio, cell_data is a dict: name -> list[array per cell-block]

    # In meshio, m.cell_data is dict: {name: [arr_per_block...]}
    cd_keys = list(m.cell_data.keys())
    for key in candidate_keys(cd_keys, field_preference, extra_tokens=["rho", "density"]):
        per_block_arrays = m.cell_data[key]
        if not per_block_arrays:
            continue
        values_list: List[np.ndarray] = []
        centers_list: List[np.ndarray] = []
        for cells_block, cell_vals in zip(m.cells, per_block_arrays):
            cell_vals = np.asarray(cell_vals).astype(float).ravel()
            if cell_vals.size == 0:
                continue
            # Compute cell centers in 2D
            for conn in cells_block.data:
                xy = m.points[conn, :2]
                centers_list.append(xy.mean(axis=0))
            values_list.append(cell_vals)
        if values_list:
            centers = np.asarray(centers_list)
            values = np.concatenate(values_list)
            if is_nonconstant(values):
                return centers, values, "cell", key

    raise KeyError(
        f"Field '{field_preference}' not found or constant in either point_data or cell_data.\n"
        f"Point keys: {pd_keys}\nCell keys: {list(m.cell_data.keys())}"
    )


def list_fields(m: meshio.Mesh) -> Tuple[List[str], List[str]]:
    pd_keys = list(m.point_data.keys())
    cd_keys = list(m.cell_data.keys())
    return pd_keys, cd_keys


def quick_plots(outdir: str, tri: Optional[Triangulation], pts_xy: np.ndarray, vals: np.ndarray, title: str, fname_prefix: str):
    os.makedirs(outdir, exist_ok=True)

    # Histogram
    plt.figure(figsize=(5, 4))
    plt.hist(vals[~np.isnan(vals)], bins=60, color="#337ab7")
    plt.title(f"Histogram: {title}")
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{fname_prefix}_hist.png"), dpi=180)
    plt.close()

    # 2D field plot
    plt.figure(figsize=(6, 5))
    if tri is not None:
        tpc = plt.tripcolor(tri, vals, shading="gouraud")
    else:
        # Fallback: scatter if triangulation unavailable
        tpc = plt.scatter(pts_xy[:, 0], pts_xy[:, 1], c=vals, s=3, cmap="viridis")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(title)
    plt.colorbar(tpc, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{fname_prefix}_field.png"), dpi=220)
    plt.close()


def save_csv(outdir: str, pts_xy: np.ndarray, vals: np.ndarray, fname: str):
    os.makedirs(outdir, exist_ok=True)
    arr = np.column_stack([pts_xy[:, 0], pts_xy[:, 1], vals])
    np.savetxt(os.path.join(outdir, fname), arr, delimiter=",", header="x,y,value", comments="")


# -------------------------
# User defaults 
# -------------------------
# If you prefer not to pass CLI arguments, set these and just run the script.
DEFAULT_VTU = "double_ramp_configuration/outputs/backward/sweep/double_ramp_0p011_0p0488_ma_2p892_pres_199070_interpolated_arrays_density_M2p892_T300p0_P199070p0/flow.vtu"  # can be a file or a folder
DEFAULT_FIELD = "density"  # e.g., density/rho, Temperature, Pressure
DEFAULT_OUT = "postprocess_outputs/auto"
DEFAULT_LIST_FIELDS_ONLY = False
DEFAULT_RECURSIVE = True  # when DEFAULT_VTU is a folder, search recursively


def process_single_vtu(vtu_path: str, field: str, outdir: str, list_only: bool = False) -> None:
    os.makedirs(outdir, exist_ok=True)
    used_reader = "meshio"

    # Try meshio first
    m = None
    try:
        m = meshio.read(vtu_path)
    except Exception:
        m = None

    if m is not None:
        pd_keys, cd_keys = list_fields(m)
        # write field list
        with open(os.path.join(outdir, "available_fields.txt"), "w") as f:
            f.write("Point data keys:\n")
            for k in pd_keys:
                f.write(f"  - {k}\n")
            f.write("\nCell data keys:\n")
            for k in cd_keys:
                f.write(f"  - {k}\n")
        if list_only:
            return
        pts_xy, vals, location, chosen_key = extract_field_meshio(m, field)
        tri = meshio_triangulation(m) if location == "point" else None
    else:
        # VTK fallback
        used_reader = "vtk"
        import vtk  # type: ignore
        from vtk.util.numpy_support import vtk_to_numpy  # type: ignore

        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(vtu_path)
        reader.Update()
        ug = reader.GetOutput()

        pd = ug.GetPointData()
        pd_keys = [pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]
        cd = ug.GetCellData()
        cd_keys = [cd.GetArrayName(i) for i in range(cd.GetNumberOfArrays())]

        with open(os.path.join(outdir, "available_fields.txt"), "w") as f:
            f.write("Point data keys:\n")
            for k in pd_keys:
                f.write(f"  - {k}\n")
            f.write("\nCell data keys:\n")
            for k in cd_keys:
                f.write(f"  - {k}\n")
        if list_only:
            return

        vtk_points = ug.GetPoints()
        pts_np3 = vtk_to_numpy(vtk_points.GetData())
        pts2 = pts_np3[:, :2]

        chosen_key = None
        vals = None  # type: ignore
        location = "point"
        for key in candidate_keys(pd_keys, field, extra_tokens=["rho", "density"]):
            arr = pd.GetArray(key)
            if arr is None:
                continue
            vals_np = vtk_to_numpy(arr).astype(float).ravel()
            if is_nonconstant(vals_np):
                chosen_key = key
                vals = vals_np
                location = "point"
                break
        if chosen_key is None:
            centers: List[np.ndarray] = []
            for key in candidate_keys(cd_keys, field, extra_tokens=["rho", "density"]):
                arr = cd.GetArray(key)
                if arr is None:
                    continue
                vals_np = vtk_to_numpy(arr).astype(float).ravel()
                centers.clear()
                n_cells = ug.GetNumberOfCells()
                for i in range(n_cells):
                    cell = ug.GetCell(i)
                    ids = [cell.GetPointId(j) for j in range(cell.GetNumberOfPoints())]
                    xy = pts2[ids]
                    centers.append(xy.mean(axis=0))
                centers_np = np.asarray(centers)
                if is_nonconstant(vals_np):
                    chosen_key = key
                    vals = vals_np
                    pts2 = centers_np
                    location = "cell"
                    break
        if chosen_key is None or vals is None:
            raise KeyError(f"Field '{field}' not found in VTU via VTK.")

        tri = None
        if location == "point":
            tris: List[Tuple[int, int, int]] = []
            n_cells = ug.GetNumberOfCells()
            VTK_TRIANGLE = 5
            VTK_QUAD = 9
            for i in range(n_cells):
                cell = ug.GetCell(i)
                ctype = ug.GetCellType(i)
                ids = [cell.GetPointId(j) for j in range(cell.GetNumberOfPoints())]
                if ctype == VTK_TRIANGLE and len(ids) == 3:
                    tris.append((ids[0], ids[1], ids[2]))
                elif ctype == VTK_QUAD and len(ids) == 4:
                    i0, i1, i2, i3 = ids
                    tris.append((i0, i1, i2))
                    tris.append((i0, i2, i3))
            if tris:
                tri = Triangulation(pts2[:, 0], pts2[:, 1], np.asarray(tris))
        pts_xy = pts2

    # Save results
    base = f"{chosen_key}_{location}"
    save_csv(outdir, pts_xy, vals, f"{base}.csv")
    quick_plots(outdir, tri, pts_xy, vals, title=f"{chosen_key} ({location})", fname_prefix=base)
    with open(os.path.join(outdir, "stats.txt"), "a") as f:
        f.write(f"File: {vtu_path}\n")
        f.write(f"Field: {chosen_key} ({location})\n")
        f.write(f"min={np.nanmin(vals):.6g} max={np.nanmax(vals):.6g} std={np.nanstd(vals):.6g}\n\n")
    print(f"Done. Reader={used_reader}. Wrote CSV and plots to: {outdir}")


def main():
    parser = argparse.ArgumentParser(description="Post-process SU2 VTU outputs and plot fields.")
    parser.add_argument("--vtu", required=False, default=None, help="Path to a .vtu file OR a folder to scan")
    parser.add_argument("--field", required=False, default=None, help="Field name preference (e.g., density, rho, Temperature, Pressure)")
    parser.add_argument("--out", required=False, default=None, help="Output directory for plots and CSV")
    parser.add_argument("--list-fields", action="store_true", help="Only list available fields and exit")
    parser.add_argument("--no-recursive", action="store_true", help="When VTU is a folder, do not search recursively")
    args = parser.parse_args()
    # Determine inputs (CLI overrides defaults)
    vtu_input = args.vtu or DEFAULT_VTU
    field = args.field or DEFAULT_FIELD
    out = args.out or DEFAULT_OUT
    list_only = args.list_fields or DEFAULT_LIST_FIELDS_ONLY
    recursive = DEFAULT_RECURSIVE and not args.no_recursive

    if os.path.isdir(vtu_input):
        # Batch mode: process all .vtu under the folder
        root = os.path.abspath(vtu_input)
        # Walk and process
        count = 0
        for dirpath, dirnames, filenames in os.walk(root):
            if not recursive and dirpath != root:
                continue
            vtus = [f for f in filenames if f.lower().endswith(".vtu")]
            for fname in vtus:
                fpath = os.path.join(dirpath, fname)
                rel_dir = os.path.relpath(dirpath, root)
                outdir = os.path.join(out, rel_dir)
                try:
                    process_single_vtu(fpath, field, outdir, list_only=list_only)
                    count += 1
                except Exception as e:
                    print(f"[WARN] Skipped {fpath}: {e}")
        print(f"Processed {count} VTU files from folder: {vtu_input}")
    else:
        if not os.path.isfile(vtu_input):
            raise FileNotFoundError(f"VTU not found: {vtu_input}")
        process_single_vtu(vtu_input, field, out, list_only=list_only)


if __name__ == "__main__":
    main()
