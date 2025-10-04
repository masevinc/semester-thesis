"""
Mesh convergence study utilities.

Workflow:
- Select a single extracted geometry points file (.npy) as the reference case.
- For a list of (vertical_nodes, horizontal_target_nodes) pairs, generate meshes and SU2 case folders.
- Optionally, after SU2 runs are completed and VTU files exist, sample a chosen field
  along a horizontal line y = Y0 and produce an overlay plot and CSV for convergence assessment.

Dependencies: meshio, matplotlib, numpy
Optional: matplotlib.tri.LinearTriInterpolator for point-data interpolation
"""

from __future__ import annotations

import os
import shutil
from typing import List, Optional, Sequence, Tuple

import numpy as np
import meshio
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator

# Local imports
from src.run_automation_s2 import main as mesh_and_case_generator
from src.postprocess_helpers import (
    extract_field_meshio_safe,
    triangulation_from_mesh,
    candidate_keys,
    is_nonconstant,
)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _pick_points_file(points_dir: str, preferred_filename: Optional[str] = None) -> str:
    """Return absolute path to a selected .npy points file.

    - If preferred_filename is provided and exists, use it (match exact filename or basename).
    - Else, pick the first .npy sorted by name.
    """
    if preferred_filename:
        # Allow passing either basename or full path
        cand = preferred_filename
        if not os.path.isabs(cand):
            cand = os.path.join(points_dir, preferred_filename)
        if os.path.isfile(cand):
            return os.path.abspath(cand)
        # Try matching by basename within directory
        base = os.path.basename(preferred_filename)
        maybe = os.path.join(points_dir, base)
        if os.path.isfile(maybe):
            return os.path.abspath(maybe)
        raise FileNotFoundError(f"Preferred points file not found: {preferred_filename}")
    # Fallback: first .npy
    npys = sorted([f for f in os.listdir(points_dir) if f.endswith('.npy')])
    if not npys:
        raise FileNotFoundError(f"No .npy points found under: {points_dir}")
    return os.path.abspath(os.path.join(points_dir, npys[0]))


def _prep_single_case_points_dir(selected_points_file: str, work_root: str) -> str:
    """Create a temporary points directory containing only the selected points file.
    Returns path to the created directory.
    """
    out_dir = os.path.join(work_root, 'points_single_case')
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(selected_points_file, os.path.join(out_dir, os.path.basename(selected_points_file)))
    return out_dir


def _pair_grid(vertical_list: Sequence[int], horizontal_list: Sequence[int], explicit_pairs: Optional[Sequence[Tuple[int, int]]] = None) -> List[Tuple[int, int]]:
    """Build list of (V, H) pairs.
    Priority:
      - If explicit_pairs provided, use those.
      - Else if len(vertical_list) == len(horizontal_list), zip them.
      - Else take Cartesian product of both lists.
    """
    if explicit_pairs and len(explicit_pairs) > 0:
        return list(explicit_pairs)
    if len(vertical_list) == len(horizontal_list):
        return list(zip(vertical_list, horizontal_list))
    return [(v, h) for v in vertical_list for h in horizontal_list]


def generate_meshes_and_cases_for_pairs(
    points_file: str,
    work_root: str,
    pairs: Sequence[Tuple[int, int]],
    *,
    mesh_format: str,
    cfg_template: str | Sequence[str],
    inlet_temperatures: Sequence[float],
    su2_output_root: str,
    slurm_partition: str = "standard",
    slurm_time: str = "01:00:00",
    slurm_nodes: int = 1,
    slurm_ntasks: int = 4,
    module_load: str = "module load su2/8.0.0",
    clear_before: bool = True,
    write_master_local_script: bool = True,
) -> List[Tuple[str, Tuple[int, int]]]:
    """For each (V,H) pair, generate a mesh and corresponding SU2 case folder(s).

    Returns list of (case_output_root_dir, (V,H)). Each case_output_root_dir will contain
    per-temperature subfolders produced by the sweep generator.
    """
    selected_dir = _prep_single_case_points_dir(points_file, work_root)
    base_mesh_root = os.path.join(work_root, 'meshes')
    results: List[Tuple[str, Tuple[int, int]]] = []

    for (v_nodes, h_nodes) in pairs:
        mesh_dir = os.path.join(base_mesh_root, f"V{v_nodes}_H{h_nodes}")
        error_log = os.path.join(mesh_dir, "mesh_errors.csv")
        case_out_root = os.path.join(su2_output_root, f"V{v_nodes}_H{h_nodes}")

        mesh_and_case_generator(
            points_dir=selected_dir,
            mesh_dir=mesh_dir,
            error_log=error_log,
            expected_num_points=12,
            run_sweep=True,
            sweep_cfg_template=cfg_template,
            sweep_output_root=case_out_root,
            sweep_inlet_temperatures=list(inlet_temperatures),
            sweep_slurm_partition=slurm_partition,
            sweep_slurm_time=slurm_time,
            sweep_slurm_nodes=slurm_nodes,
            sweep_slurm_ntasks=slurm_ntasks,
            sweep_module_load=module_load,
            sweep_clear_output_before_run=clear_before,
            sweep_write_master_slurm_script=True,
            mesh_format=mesh_format,
            sweep_write_master_local_script=write_master_local_script,
            vertical_nodes=int(v_nodes),
            horizontal_target_nodes=int(h_nodes),
        )
        results.append((case_out_root, (v_nodes, h_nodes)))
    return results


def _find_vtu_files(root: str) -> List[str]:
    vtus: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith('.vtu'):
                vtus.append(os.path.join(dirpath, f))
    return sorted(vtus)


def sample_yline_from_vtu(vtu_path: str, field: str, y_value: float, n_samples: int = 400) -> Tuple[np.ndarray, np.ndarray]:
    """Sample the field along y=y_value.

    Returns (x_sorted, vals) where x covers [xmin, xmax] of the mesh domain.
    Uses linear interpolation on point-data when available; falls back to nearest-neighbor
    from provided (x,y) locations when interpolation is not possible.
    """
    # Try MeshIO first, then fallback to VTK if needed
    use_vtk_fallback = False
    try:
        m = meshio.read(vtu_path)
        pts_xy, vals, location, _chosen_key = extract_field_meshio_safe(m, field)
    except Exception:
        use_vtk_fallback = True

    if not use_vtk_fallback:
        # Determine x-range
        xmin = float(np.nanmin(pts_xy[:, 0]))
        xmax = float(np.nanmax(pts_xy[:, 0]))
        xs = np.linspace(xmin, xmax, int(n_samples))
        ys = np.full_like(xs, float(y_value))

        # Try interpolation using triangulation when we have point data
        tri = None
        if location == 'point':
            tri = triangulation_from_mesh(m)

        if tri is not None and isinstance(tri, Triangulation):
            interp = LinearTriInterpolator(tri, vals)
            v = interp(xs, ys)
            v = np.asarray(v, dtype=float)
            return xs, v

        # Fallback: nearest neighbor among given sample locations
        out = np.empty_like(xs)
        for i, x in enumerate(xs):
            dx = pts_xy[:, 0] - x
            dy = pts_xy[:, 1] - y_value
            j = int(np.argmin(dx * dx + dy * dy))
            out[i] = float(vals[j])
        return xs, out

    # --- VTK fallback ---
    try:
        import vtk  # type: ignore
        from vtk.util.numpy_support import vtk_to_numpy  # type: ignore
    except Exception as e:
        raise RuntimeError(
            f"VTK fallback unavailable and MeshIO failed for {vtu_path}: {e}."
        )

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(vtu_path)
    reader.Update()
    ug = reader.GetOutput()

    vtk_points = ug.GetPoints()
    pts_np3 = vtk_to_numpy(vtk_points.GetData())
    pts2 = pts_np3[:, :2]

    pd = ug.GetPointData()
    pd_keys = [pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]
    cd = ug.GetCellData()
    cd_keys = [cd.GetArrayName(i) for i in range(cd.GetNumberOfArrays())]

    chosen_key = None
    vals = None  # type: ignore
    location = 'point'

    # Point-data preference: scalar, name contains density/rho
    for key in candidate_keys(pd_keys, field, extra_tokens=['rho', 'density']):
        arr = pd.GetArray(key)
        if arr is None:
            continue
        num_comps = arr.GetNumberOfComponents()
        if num_comps != 1:
            continue
        vals_np = vtk_to_numpy(arr).astype(float).ravel()
        if vals_np.size != pts2.shape[0]:
            continue
        if is_nonconstant(vals_np):
            chosen_key = key
            vals = vals_np
            location = 'point'
            break

    if chosen_key is None:
        # Cell-data path: compute centers and use scalar arrays only
        centers: List[np.ndarray] = []
        for key in candidate_keys(cd_keys, field, extra_tokens=['rho', 'density']):
            arr = cd.GetArray(key)
            if arr is None:
                continue
            num_comps = arr.GetNumberOfComponents()
            if num_comps != 1:
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
            if vals_np.size == centers_np.shape[0] and is_nonconstant(vals_np):
                chosen_key = key
                vals = vals_np
                pts2 = centers_np
                location = 'cell'
                break

    if chosen_key is None or vals is None:
        raise KeyError(f"Field '{field}' not found as scalar in VTK dataset for {vtu_path}.")

    # Now sample along y=y0 using nearest neighbor (no triangulation in VTK path)
    xmin = float(np.nanmin(pts2[:, 0]))
    xmax = float(np.nanmax(pts2[:, 0]))
    xs = np.linspace(xmin, xmax, int(n_samples))
    out = np.empty_like(xs)
    for i, x in enumerate(xs):
        dx = pts2[:, 0] - x
        dy = pts2[:, 1] - y_value
        j = int(np.argmin(dx * dx + dy * dy))
        out[i] = float(vals[j])
    return xs, out


def _pretty_field_label(field: str) -> str:
    f = field.lower()
    if f in ("density", "rho"):
        return "Density (kg/m³)"
    if f in ("pressure", "p"):
        return "Pressure (Pa)"
    if f in ("temperature", "t"):
        return "Temperature (K)"
    if f in ("mach", "m"):
        return "Mach (-)"
    # Fallback
    return field.capitalize()


def plot_yline_overlay(
    samples: List[Tuple[Tuple[int, int], np.ndarray, np.ndarray]],
    out_dir: str,
    field: str,
    y_value: float,
    zoom_range: Tuple[float, float] | None = (0.35, 0.50),
) -> None:
    """Create an overlay plot of value(x) at y=y0 for multiple (V,H) resolutions.
    Also writes a CSV with columns: x, V{v}_H{h}...

    Produces two figures:
      1) Full x-range plot
      2) Zoomed-in plot for the given x-range (default: 0.35 to 0.50)
    """
    _ensure_dir(out_dir)
    # Build common x-axis by choosing the densest sampling (largest length)
    max_len = max(len(xs) for _, xs, _ in samples)
    common_x = None
    for (_, xs, _vals) in samples:
        if len(xs) == max_len:
            common_x = xs
            break
    assert common_x is not None

    y_label = _pretty_field_label(field)
    x_label = "x-position (m)"

    # Common style cycles for better distinction
    linestyles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '>']
    lw = 1.8
    ms = 4.5
    markevery = max(1, len(common_x) // 25)

    plt.figure(figsize=(8.5, 4.6))
    csv_cols = [common_x]
    headers = ["x"]

    for idx, ((v, h), xs, vs) in enumerate(samples):
        lbl = f"V{v}_H{h}"
        ls = linestyles[idx % len(linestyles)]
        mk = markers[idx % len(markers)]
        # If xs differ from common_x, interpolate for CSV alignment
        if xs is not common_x:
            vs_aligned = np.interp(common_x, xs, vs)
        else:
            vs_aligned = vs
        plt.plot(
            common_x, vs_aligned,
            label=lbl,
            linestyle=ls,
            marker=mk,
            markevery=markevery,
            linewidth=lw,
            markersize=ms,
        )
        csv_cols.append(vs_aligned)
        headers.append(lbl)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # No main title by request
    plt.legend(ncols=2, fontsize=9, frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    png_path = os.path.join(out_dir, f"yline_{field}_y{str(y_value).replace('.', 'p')}.png")
    plt.savefig(png_path, dpi=220)
    plt.close()

    arr = np.column_stack(csv_cols)
    csv_path = os.path.join(out_dir, f"yline_{field}_y{str(y_value).replace('.', 'p')}.csv")
    np.savetxt(csv_path, arr, delimiter=",", header=",".join(headers), comments="")
    print(f"[mesh-conv] Wrote overlay plot: {png_path}\n[mesh-conv] Wrote CSV: {csv_path}")

    # Zoomed-in helper plot
    if zoom_range is not None:
        x0, x1 = zoom_range
        plt.figure(figsize=(8.5, 4.2))
        for idx, ((v, h), xs, vs) in enumerate(samples):
            lbl = f"V{v}_H{h}"
            ls = linestyles[idx % len(linestyles)]
            mk = markers[idx % len(markers)]
            if xs is not common_x:
                vs_aligned = np.interp(common_x, xs, vs)
            else:
                vs_aligned = vs
            mask = (common_x >= x0) & (common_x <= x1)
            if not np.any(mask):
                continue
            plt.plot(
                common_x[mask], vs_aligned[mask],
                label=lbl,
                linestyle=ls,
                marker=mk,
                markevery=max(1, int(markevery/2)),
                linewidth=lw,
                markersize=ms,
            )
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        # No title
        plt.legend(ncols=2, fontsize=9, frameon=True)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        zoom_tag = f"x{str(x0).replace('.', 'p')}_{str(x1).replace('.', 'p')}"
        png_zoom = os.path.join(out_dir, f"yline_{field}_y{str(y_value).replace('.', 'p')}_zoom_{zoom_tag}.png")
        plt.savefig(png_zoom, dpi=230)
        plt.close()
        print(f"[mesh-conv] Wrote zoomed overlay plot: {png_zoom}")


def overlay_contour_with_yline(vtu_path: str, field: str, y_value: float, out_dir: str, title_suffix: str = "") -> None:
    """Produce a field contour plot with the y=y0 line overlaid for visual reference.
    No main title is added. Axes are labeled; colorbar gets field label when possible.
    """
    _ensure_dir(out_dir)
    m = meshio.read(vtu_path)
    pts_xy, vals, location, chosen_key = extract_field_meshio_safe(m, field)
    tri = triangulation_from_mesh(m) if location == 'point' else None

    plt.figure(figsize=(6.8, 5.2))
    if tri is not None:
        tpc = plt.tripcolor(tri, vals, shading="gouraud")
    else:
        tpc = plt.scatter(pts_xy[:, 0], pts_xy[:, 1], c=vals, s=3, cmap="viridis")
    plt.axhline(y_value, color='r', lw=1.2, alpha=0.8, label=f"y={y_value}")
    plt.gca().set_aspect('equal', adjustable='box')
    # No main title
    cbar = plt.colorbar(tpc, fraction=0.046, pad=0.04)
    cbar.set_label(_pretty_field_label(chosen_key), rotation=90)
    plt.xlabel("x-position (m)")
    plt.ylabel("y-position (m)")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(out_dir, f"contour_{field}_y{str(y_value).replace('.', 'p')}.png")
    plt.savefig(fname, dpi=210)
    plt.close()
    print(f"[mesh-conv] Wrote contour: {fname}")


def run_mesh_convergence(
    *,
    points_dir: str,
    preferred_points_file: Optional[str],
    work_root: str,
    resolutions_vertical: Sequence[int],
    resolutions_horizontal: Sequence[int],
    resolution_pairs: Optional[Sequence[Tuple[int, int]]] = None,
    mesh_format: str = 'su2',
    sweep_cfg_template: str | Sequence[str] = './double_ramp_configuration/inputs/hybrid_dbl_ramp.cfg',
    inlet_temperatures: Sequence[float] = (300.0,),
    su2_output_root: str = './double_ramp_configuration/outputs/backward/mesh_convergence',
    field: str = 'density',
    y_value: float = 0.2,
    do_generate: bool = True,
    do_postprocess: bool = True,
    slurm_partition: str = 'standard',
    slurm_time: str = '01:00:00',
    slurm_nodes: int = 1,
    slurm_ntasks: int = 4,
    module_load: str = 'module load su2/8.0.0',
) -> None:
    """High-level orchestrator for mesh convergence study.

    If do_generate is True, creates meshes and SU2 case folders for the selected case across
    the requested resolutions. You can then run the generated run_all_local.sh scripts.

    If do_postprocess is True, searches for VTU results under su2_output_root and samples the
    requested field along y=y_value; produces overlay plot/CSV.
    """
    _ensure_dir(work_root)
    _ensure_dir(su2_output_root)

    sel_points = _pick_points_file(points_dir, preferred_points_file)
    case_name = os.path.splitext(os.path.basename(sel_points))[0]
    study_root = os.path.join(su2_output_root, case_name)
    _ensure_dir(study_root)

    pairs = _pair_grid(list(resolutions_vertical), list(resolutions_horizontal), resolution_pairs)

    if do_generate:
        print(f"[mesh-conv] Generating meshes and cases for {len(pairs)} resolutions...")
        generate_meshes_and_cases_for_pairs(
            points_file=sel_points,
            work_root=os.path.join(study_root, 'work'),
            pairs=pairs,
            mesh_format=mesh_format,
            cfg_template=sweep_cfg_template,
            inlet_temperatures=inlet_temperatures,
            su2_output_root=os.path.join(study_root, 'cases'),
            slurm_partition=slurm_partition,
            slurm_time=slurm_time,
            slurm_nodes=slurm_nodes,
            slurm_ntasks=slurm_ntasks,
            module_load=module_load,
            clear_before=True,
            write_master_local_script=True,
        )
        print(f"[mesh-conv] Cases created under: {os.path.join(study_root, 'cases')}\n"
              f"  -> Run each case's run.sh (or the auto-generated run_all_local.sh) to produce flow.vtu before post-processing.")

    if do_postprocess:
        print("[mesh-conv] Post-processing VTUs for y-line sampling...")
        cases_root = os.path.join(study_root, 'cases')
        # For each res subdir (Vxx_Hyy), look for a VTU (pick first found)
        samples: List[Tuple[Tuple[int, int], np.ndarray, np.ndarray]] = []
        for (v, h) in pairs:
            res_dir = os.path.join(cases_root, f"V{v}_H{h}")
            vtus = _find_vtu_files(res_dir)
            if not vtus:
                print(f"[mesh-conv][warn] No .vtu found under {res_dir}; skip this resolution.")
                continue
            vtu_path = vtus[0]
            try:
                xs, vs = sample_yline_from_vtu(vtu_path, field=field, y_value=y_value)
                samples.append(((v, h), xs, vs))
                # Also write a contour with the y-line overlay for visual context
                overlay_contour_with_yline(vtu_path, field, y_value, out_dir=os.path.join(study_root, 'viz', f"V{v}_H{h}"), title_suffix=f"V{v} H{h}")
            except Exception as e:
                print(f"[mesh-conv][warn] Failed sampling {vtu_path}: {e}")
        if samples:
            plot_yline_overlay(samples, out_dir=os.path.join(study_root, 'viz'), field=field, y_value=y_value)
        else:
            print("[mesh-conv][info] No samples collected. Ensure SU2 cases have been run and produced VTU files.")

