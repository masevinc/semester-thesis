""" 
Step 2.2

sweep_calculations_bw.py

Batch generates SU2 sweep cases for all mesh files in a directory, using extracted Mach/pressure from mesh filenames.
All configuration is set via function arguments.
"""

import os
import math
import shutil
import re

GAMMA = 1.4
R = 287.058

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

def write_master_slurm_script(output_root):
    """
    Writes a master SLURM script that submits all submit.slurm jobs in subdirectories.
    """
    script_path = os.path.join(output_root, "submit_all.slurm")
    with open(script_path, "w") as f:
        f.write("#!/bin/bash\n\n")
        for case_dir in sorted(os.listdir(output_root)):
            full_case_dir = os.path.join(output_root, case_dir)
            submit_path = os.path.join(full_case_dir, "submit.slurm")
            if os.path.isfile(submit_path):
                f.write(f"cd {full_case_dir}\n")
                f.write("sbatch submit.slurm\n")
                f.write("cd - > /dev/null\n")
    os.chmod(script_path, 0o755)
    print(f"\n+++ Master SLURM script created: {script_path}")

def compute_velocity_x(mach, temp):
    a = math.sqrt(GAMMA * R * temp)
    return mach * a

def replace_value(text, key, new_value):
    pattern = rf"({key}\s*=\s*)\S+"
    return re.sub(pattern, lambda m: f"{m.group(1)}{new_value}", text)

def replace_marker_inlet(text, temp, pressure, velocity_x):
    pattern = r"(MARKER_SUPERSONIC_INLET\s*=\s*\(\s*Inlet\s*,\s*)[\d\.Ee+-]+,\s*[\d\.Ee+-]+,\s*[\d\.Ee+-]+"
    replacement = rf"\g<1>{temp}, {pressure}, {velocity_x:.6f}"
    return re.sub(pattern, replacement, text)

def modify_cfg(cfg_text, mach, temp, pressure, mesh_file):
    velocity_x = compute_velocity_x(mach, temp)

    cfg_text = replace_value(cfg_text, "MACH_NUMBER", mach)
    cfg_text = replace_value(cfg_text, "FREESTREAM_TEMPERATURE", temp)
    cfg_text = replace_value(cfg_text, "FREESTREAM_PRESSURE", pressure)
    cfg_text = replace_marker_inlet(cfg_text, temp, pressure, velocity_x)
    cfg_text = replace_value(cfg_text, "MESH_FILENAME", mesh_file)

    mesh_format = os.path.splitext(mesh_file)[1].replace('.', '').upper()
    cfg_text = replace_value(cfg_text, "MESH_FORMAT", mesh_format)

    return cfg_text

def extract_ma_pres_from_filename(filename):
    """
    Extracts Mach and pressure from filename.
    Supports format: ..._ma_{mach}_pres_{pressure}...
    """
    base = os.path.splitext(filename)[0]
    match = re.search(r'_ma_([0-9\.]+)_pres_([0-9\.]+)', base)
    if not match:
        raise ValueError(f"Filename {filename} does not contain Mach and Pressure info.")
    mach_str, pres_str = match.groups()
    mach = float(mach_str)
    pressure = float(pres_str)
    return mach, pressure

def write_slurm_script(folder, job_name, slurm_partition, slurm_time, slurm_nodes, slurm_ntasks, module_load):
    content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --time={slurm_time}
#SBATCH --partition={slurm_partition}
#SBATCH --nodes={slurm_nodes}
#SBATCH --ntasks={slurm_ntasks}

{module_load}
srun SU2_CFD case.cfg
"""
    script_path = os.path.join(folder, "submit.slurm")
    with open(script_path, "w") as f:
        f.write(content)

def generate_sweeps_for_mesh_folder(
    mesh_dir,
    cfg_template,
    output_root,
    inlet_temperatures,
    slurm_partition="standard",
    slurm_time="01:00:00",
    slurm_nodes=1,
    slurm_ntasks=4,
    module_load="module load su2/4.1.0",
    clear_output_before_run=True,
    write_master_slurm_script_flag=True
):
    """
    Generate sweep cases for all .msh files in a given directory using extracted mach/pressure values.
    """
    if clear_output_before_run:
        clear_output_directory(output_root)
    os.makedirs(output_root, exist_ok=True)

    with open(cfg_template, 'r') as f:
        base_cfg = f.read()

    for fname in os.listdir(mesh_dir):
        if not fname.endswith('.msh'):
            continue

        try:
            mach, pressure = extract_ma_pres_from_filename(fname)
        except ValueError as e:
            print(f"Skipping file due to error: {e}")
            continue

        mesh_path = os.path.join(mesh_dir, fname)
        mesh_base = os.path.splitext(fname)[0]

        for temp in inlet_temperatures:
            case_name = f"{mesh_base}_M{mach:.3f}_T{temp:.1f}_P{pressure}".replace(".", "p")
            case_dir = os.path.join(output_root, case_name)
            os.makedirs(case_dir, exist_ok=True)

            mesh_dest = os.path.join(case_dir, fname)
            shutil.copy(mesh_path, mesh_dest)

            cfg_text = modify_cfg(base_cfg, mach, temp, pressure, fname)
            with open(os.path.join(case_dir, "case.cfg"), "w") as fcfg:
                fcfg.write(cfg_text)

            with open(os.path.join(case_dir, "run.sh"), "w") as frun:
                frun.write("#!/bin/bash\nSU2_CFD case.cfg\n")
            os.chmod(os.path.join(case_dir, "run.sh"), 0o755)

            write_slurm_script(
                case_dir, case_name,
                slurm_partition, slurm_time, slurm_nodes, slurm_ntasks, module_load
            )

    print(f"\n+++ SU2 cases created in: {output_root}/")

    if write_master_slurm_script_flag:
        write_master_slurm_script(output_root)
