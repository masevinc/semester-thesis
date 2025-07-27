import os
import math
import shutil
from itertools import product

# Configuration
CFG_TEMPLATE = './dramp_auto_backwards_v3/inv_wedge_HLLC.cfg'
OUTPUT_ROOT = './dramp_auto_backwards_v3/outputs/sweep_cases'
mach_numbers = [2.0, 2.5, 3.0]
inlet_temperatures = [250.0, 275.0, 300.0]
freestream_pressures = [90000.0, 101325.0, 110000.0]

GAMMA = 1.4
R = 287.058

# SLURM config
slurm_partition = "standard"
slurm_time = "01:00:00"
slurm_nodes = 1
slurm_ntasks = 4
module_load = "module load su2/4.1.0"

def compute_velocity_x(mach, temp):
    a = math.sqrt(GAMMA * R * temp)
    return mach * a

def replace_value(text, key, new_value):
    import re
    pattern = rf"({key}\s*=\s*)\S+"
    return re.sub(pattern, lambda m: f"{m.group(1)}{new_value}", text)

def replace_marker_inlet(text, temp, pressure, velocity_x):
    import re
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

def generate_sweeps_for_mesh_folder(mesh_dir):
    """
    Generate sweep cases for all .msh files in a given directory.
    """
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    with open(CFG_TEMPLATE, 'r') as f:
        base_cfg = f.read()

    for fname in os.listdir(mesh_dir):
        if not fname.endswith('.msh'):
            continue

        mesh_path = os.path.join(mesh_dir, fname)
        mesh_base = os.path.splitext(fname)[0]

        for mach, temp, pressure in product(mach_numbers, inlet_temperatures, freestream_pressures):
            case_name = f"{mesh_base}_M{mach:.1f}_T{temp:.1f}_P{int(pressure)}".replace(".", "p")
            case_dir = os.path.join(OUTPUT_ROOT, case_name)
            os.makedirs(case_dir, exist_ok=True)

            mesh_dest = os.path.join(case_dir, fname)
            shutil.copy(mesh_path, mesh_dest)

            cfg_text = modify_cfg(base_cfg, mach, temp, pressure, fname)
            with open(os.path.join(case_dir, "case.cfg"), "w") as fcfg:
                fcfg.write(cfg_text)

            with open(os.path.join(case_dir, "run.sh"), "w") as frun:
                frun.write("#!/bin/bash\nSU2_CFD case.cfg\n")
            os.chmod(os.path.join(case_dir, "run.sh"), 0o755)

            write_slurm_script(case_dir, case_name)

    print(f"\n+++ SU2 cases created in: {OUTPUT_ROOT}/")
