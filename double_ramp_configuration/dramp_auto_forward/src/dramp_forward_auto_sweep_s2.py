"""

Step 2

run_automation.py

Processes generated .msh mesh files for the forward case, creates sweep folders and cases,
and makes them ready for HPC runs.
"""

import os
import shutil
from src.dramp_sweep_calc import generate_sweeps_for_mesh_folder

# === CONFIGURATION ===
FORWARD_MESH_DIR = './double_ramp_configuration/outputs/forward/mesh'
SWEEP_OUTPUT_DIR = 'double_ramp_configuration/outputs/forward/sweep'

def clear_output_directory(directory):
    """
    Removes all files and folders in the specified directory. Creates the directory if it does not exist.
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

def write_master_slurm_script(sweep_dir):
    """
    Writes a master SLURM script that submits all submit.slurm jobs in subdirectories.
    """
    script_path = os.path.join(sweep_dir, "submit_all.slurm")
    with open(script_path, "w") as f:
        f.write("#!/bin/bash\n\n")
        for case_dir in sorted(os.listdir(sweep_dir)):
            full_case_dir = os.path.join(sweep_dir, case_dir)
            submit_path = os.path.join(full_case_dir, "submit.slurm")
            if os.path.isfile(submit_path):
                f.write(f"cd {full_case_dir}\n")
                f.write("sbatch submit.slurm\n")
                f.write("cd - > /dev/null\n")
    os.chmod(script_path, 0o755)
    print(f"\n+++ Master SLURM script created: {script_path}")

def main():
    # Optional: clear previous sweep outputs
    clear_output_directory(SWEEP_OUTPUT_DIR)
    os.makedirs(SWEEP_OUTPUT_DIR, exist_ok=True)

    # Generate sweep cases for all .msh files in the forward mesh directory
    print(f"Generating sweep cases for all meshes in {FORWARD_MESH_DIR} ...")
    generate_sweeps_for_mesh_folder(FORWARD_MESH_DIR, SWEEP_OUTPUT_DIR)
    print("All sweep cases are ready.")

    # Write master SLURM script
    write_master_slurm_script(SWEEP_OUTPUT_DIR)

if __name__ == "__main__":
    main()