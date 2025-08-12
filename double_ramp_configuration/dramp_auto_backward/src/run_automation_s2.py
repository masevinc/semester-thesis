"""

Step 2

run_automation.py

Processes extracted geometric points to generate mesh files and logs any errors encountered.
Optionally, can trigger sweep calculations for all generated meshes.
"""

import os
import csv
import numpy as np
import shutil
from src.mesh_generator import generate_mesh_from_points
from src.sweep_calculations_bw import generate_sweeps_for_mesh_folder

def clear_output_directory(directory):
    """
    Removes all files in the specified directory. Creates the directory if it does not exist.
    """
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")
    else:
        os.makedirs(directory, exist_ok=True)

def main(
    points_dir,
    mesh_dir,
    error_log,
    expected_num_points,
    run_sweep=False,
    sweep_cfg_template=None,
    sweep_output_root=None,
    sweep_inlet_temperatures=None,
    sweep_slurm_partition="standard",
    sweep_slurm_time="01:00:00",
    sweep_slurm_nodes=1,
    sweep_slurm_ntasks=4,
    sweep_module_load="module load su2/4.1.0",
    sweep_clear_output_before_run=True,
    sweep_write_master_slurm_script=True
):
    """
    Main routine to generate mesh files from extracted points and log errors.
    Optionally runs sweep calculations for all meshes.
    """
    clear_output_directory(mesh_dir)
    os.makedirs(mesh_dir, exist_ok=True)

    with open(error_log, mode='w', newline='') as errfile:
        writer = csv.writer(errfile)
        writer.writerow(['filename', 'error'])

        for fname in os.listdir(points_dir):
            if not fname.endswith('.npy'):
                continue

            full_path = os.path.join(points_dir, fname)
            points = np.load(full_path)

            print(f"Processing file: {fname}")
            if len(points) != expected_num_points:
                msg = f"expected {expected_num_points} points, got {len(points)}"
                writer.writerow([fname, msg])
                continue

            base_name = os.path.splitext(fname)[0]
            mesh_out = os.path.join(mesh_dir, f"{base_name}.msh")

            try:
                generate_mesh_from_points(points, mesh_out, show_gui=False)
                print(f"  Mesh saved to {mesh_out}")
            except Exception as e:
                msg = str(e)
                print(f"  Failed to generate mesh: {msg}")
                writer.writerow([fname, msg])

    if run_sweep and sweep_cfg_template and sweep_output_root and sweep_inlet_temperatures:
        generate_sweeps_for_mesh_folder(
            mesh_dir=mesh_dir,
            cfg_template=sweep_cfg_template,
            output_root=sweep_output_root,
            inlet_temperatures=sweep_inlet_temperatures,
            slurm_partition=sweep_slurm_partition,
            slurm_time=sweep_slurm_time,
            slurm_nodes=sweep_slurm_nodes,
            slurm_ntasks=sweep_slurm_ntasks,
            module_load=sweep_module_load,
            clear_output_before_run=sweep_clear_output_before_run,
            write_master_slurm_script_flag=sweep_write_master_slurm_script
        )

if __name__ == "__main__":
    # Default configuration (can be overridden by importing and calling main)
    POINTS_DIR = './double_ramp_configuration/outputs/backward/extracted_points'
    MESH_DIR = './double_ramp_configuration/outputs/backward/mesh'
    ERROR_LOG = './double_ramp_configuration/outputs/backward/mesh_errors.csv'
    EXPECTED_NUM_POINTS = 12

    SWEEP_CFG_TEMPLATE = './double_ramp_configuration/inputs/inv_wedge_HLLC.cfg'
    SWEEP_OUTPUT_ROOT = './double_ramp_configuration/outputs/backward/sweep'
    SWEEP_INLET_TEMPERATURES = [250.0, 275.0, 300.0]
    SWEEP_SLURM_PARTITION = "standard"
    SWEEP_SLURM_TIME = "01:00:00"
    SWEEP_SLURM_NODES = 1
    SWEEP_SLURM_NTASKS = 4
    SWEEP_MODULE_LOAD = "module load su2/4.1.0"
    SWEEP_CLEAR_OUTPUT_BEFORE_RUN = True
    SWEEP_WRITE_MASTER_SLURM_SCRIPT = True
    RUN_SWEEP = True

    main(
        points_dir=POINTS_DIR,
        mesh_dir=MESH_DIR,
        error_log=ERROR_LOG,
        expected_num_points=EXPECTED_NUM_POINTS,
        run_sweep=RUN_SWEEP,
        sweep_cfg_template=SWEEP_CFG_TEMPLATE,
        sweep_output_root=SWEEP_OUTPUT_ROOT,
        sweep_inlet_temperatures=SWEEP_INLET_TEMPERATURES,
        sweep_slurm_partition=SWEEP_SLURM_PARTITION,
        sweep_slurm_time=SWEEP_SLURM_TIME,
        sweep_slurm_nodes=SWEEP_SLURM_NODES,
        sweep_slurm_ntasks=SWEEP_SLURM_NTASKS,
        sweep_module_load=SWEEP_MODULE_LOAD,
        sweep_clear_output_before_run=SWEEP_CLEAR_OUTPUT_BEFORE_RUN,
        sweep_write_master_slurm_script=SWEEP_WRITE_MASTER_SLURM_SCRIPT
    )