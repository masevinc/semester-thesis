import os
import math
import shutil
from itertools import product
from generate_mesh import create_geometry, get_mesh_filename

# ------------------- SWEEP PARAMETERS ------------------- #
mach_numbers = [2.0, 2.5, 3.0]
inlet_temperatures = [250.0, 275.0, 300.0]
ramp_angles = [0, 5, 10]

# Force mesh regeneration even if mesh already exists
force_remesh = False
export_mesh_ext = ".su2"  # could be ".msh" or ".cgns"

# SLURM job config
slurm_partition = "standard"
slurm_time = "01:00:00"
slurm_nodes = 1
slurm_ntasks = 4
module_load = "module load su2/4.1.0"

# File paths
base_cfg_path = "thesis/auto_msh_cgf/inv_wedge_HLLC.cfg"
output_root = "sweep_cases"

# Physics constants
GAMMA = 1.4
R = 287.058

# ------------------- UTILITIES ------------------- #
def compute_velocity_x(mach, temp):
    a = math.sqrt(GAMMA * R * temp)
    return mach * a

def replace_value(text, key, new_value):
    import re
    # pattern = rf"({key}\s*=\s*)[\d\.Ee+-]+"
    pattern = rf"({key}\s*=\s*)\S+"
    return re.sub(pattern, lambda m: f"{m.group(1)}{new_value}", text)

def replace_marker_inlet(text, temp, pressure, velocity_x):
    import re
    pattern = r"(MARKER_SUPERSONIC_INLET\s*=\s*\(\s*Inlet\s*,\s*)[\d\.Ee+-]+,\s*[\d\.Ee+-]+,\s*[\d\.Ee+-]+"
    replacement = rf"\g<1>{temp}, {pressure}, {velocity_x:.6f}"
    return re.sub(pattern, replacement, text)

def modify_cfg(cfg_text, mach, temp, mesh_file):
    velocity_x = compute_velocity_x(mach, temp)
    pressure = 101325.0

    cfg_text = replace_value(cfg_text, "MACH_NUMBER", mach)
    cfg_text = replace_value(cfg_text, "FREESTREAM_TEMPERATURE", temp)
    cfg_text = replace_marker_inlet(cfg_text, temp, pressure, velocity_x)
    cfg_text = replace_value(cfg_text, "MESH_FILENAME", mesh_file)

    mesh_format = os.path.splitext(mesh_file)[1].replace('.', '').upper()
    cfg_text = replace_value(cfg_text, "MESH_FORMAT", mesh_format)

    return cfg_text

def write_slurm_script(folder, job_name):
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

# ------------------- MAIN SWEEP ------------------- #
with open(base_cfg_path, "r") as file:
    base_cfg = file.read()

os.makedirs(output_root, exist_ok=True)

for mach, temp, angle in product(mach_numbers, inlet_temperatures, ramp_angles):
    folder_name = f"M{mach:.1f}_T{temp:.1f}_A{angle}".replace(".", "p")
    case_dir = os.path.join(output_root, folder_name)
    os.makedirs(case_dir, exist_ok=True)

    # Get or create mesh
    mesh_path = get_mesh_filename(angle, export_mesh_ext)
    if force_remesh or not os.path.exists(mesh_path):
        print(f"o Generating mesh for angle {angle}°...")
        mesh_path = create_geometry(angle, export_typ=export_mesh_ext)
    else:
        print(f"+ Reusing mesh for angle {angle}°.")

    # Copy mesh to case folder
    mesh_basename = os.path.basename(mesh_path)
    mesh_dest = os.path.join(case_dir, mesh_basename)
    shutil.copy(mesh_path, mesh_dest)

    # Create SU2 config
    cfg_content = modify_cfg(base_cfg, mach, temp, mesh_basename)
    cfg_path = os.path.join(case_dir, "case.cfg")
    with open(cfg_path, "w") as f:
        f.write(cfg_content)

    # Write run.sh
    run_script_path = os.path.join(case_dir, "run.sh")
    with open(run_script_path, "w") as f:
        f.write("#!/bin/bash\nSU2_CFD case.cfg\n")
    os.chmod(run_script_path, 0o755)

    # Write SLURM script
    write_slurm_script(case_dir, folder_name)

print(f"\n+++ All cases generated under: {output_root}/")
