"""Pipeline entrypoint (argparse based)

Usage examples:

  # Extract points only (process all) and create first-case visualization
  python pipeline.py --visualize-points --extract-only --no-random

  # Extract a random sample of 5, visualize first, then generate meshes (no sweep)
  python pipeline.py --visualize-points --sample-size 5 --skip-sweep

This script intentionally does NOT import mesh generation modules until
after the extraction stage (and only if not --extract-only) so that a
missing gmsh dependency does not block simple visualization / QA.
"""

from __future__ import annotations

import argparse
from src.extract_points_s1 import extract_points_batch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Double ramp automated pipeline (backward config)")
    p.add_argument('--visualize-points', action='store_true', help='Generate visualization for the first processed geometry.')
    p.add_argument('--no-random', action='store_true', help='Disable random sampling (process all filtered files).')
    p.add_argument('--sample-size', type=int, default=1, help='Sample size when random sampling enabled.')
    p.add_argument('--mesh-format', default='su2', choices=['msh', 'su2', 'cgns'], help='Output mesh format.')
    p.add_argument('--skip-sweep', action='store_true', help='Skip sweep generation step.')
    p.add_argument('--extract-only', action='store_true', help='Only run extraction + optional visualization (skip mesh stage).')
    return p.parse_args()


def run():
    args = parse_args()

    use_random = not args.no_random
    random_seed = 98

    # Step 1: Extraction
    extract_points_batch(
        data_dir="./double_ramp_configuration/inputs/double_ramp_npz_files_clamped",
        output_dir="./double_ramp_configuration/outputs/backward/extracted_points",
        filters={"ramp1": None, "ramp2": None, "min_ma": None, "max_ma": None},
        selected_keys=["density"],
        physical_height=256,
        clear_output_before_run=True,
        enable_random_sampling=use_random,
        sample_size=args.sample_size,
        random_seed=random_seed,
        manifest_filename="selection_manifest.csv",
        manifest_in_parent=True,
        visualize_first=args.visualize_points,
        visualization_output_dir="./double_ramp_configuration/outputs/backward/visualizations",
        visualization_filename_prefix="first_case"
    )

    if args.extract_only:
        print("Extraction-only completed. (Meshes/sweeps skipped)")
        return

    # Step 2: Mesh + optional sweep (lazy import to avoid gmsh dependency earlier)
    try:
        from src.run_automation_s2 import main as run_mesh_automation  # type: ignore
    except ModuleNotFoundError as e:
        print("Mesh stage skipped: missing dependency:", e)
        print("Install gmsh (pip install gmsh) or rerun with --extract-only for just visualization.")
        return

    run_mesh_automation(
        points_dir="./double_ramp_configuration/outputs/backward/extracted_points",
        mesh_dir="./double_ramp_configuration/outputs/backward/mesh",
        error_log="./double_ramp_configuration/outputs/backward/mesh_errors.csv",
        expected_num_points=12,
        run_sweep=not args.skip_sweep,
        sweep_cfg_template="./double_ramp_configuration/inputs/hybrid_dbl_ramp.cfg",
        sweep_output_root="./double_ramp_configuration/outputs/backward/sweep",
        sweep_inlet_temperatures=[300.0],
        sweep_slurm_partition="standard",
        sweep_slurm_time="01:00:00",
        sweep_slurm_nodes=1,
        sweep_slurm_ntasks=4,
        sweep_module_load="module load su2/8.1.0",
        sweep_clear_output_before_run=True,
        sweep_write_master_slurm_script=True,
        mesh_format=args.mesh_format,
        sweep_write_master_local_script=True
    )


if __name__ == '__main__':
    run()
