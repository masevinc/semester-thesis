""" 
Step 2.1

dramp_sweep_calc.py
"""

import os
import math
import shutil
from itertools import product
import re

GAMMA = 1.4
R = 287.058

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
    output_dir,
    cfg_template,
    mach_numbers,
    inlet_temperatures,
    freestream_pressures,
    slurm_partition="standard",
    slurm_time="01:00:00",
    slurm_nodes=1,
    slurm_ntasks=4,
    module_load="module load su2/4.1.0",
    cfg_stage_templates=None,
    stage_cfg_names=None,
):
    """
    Generate sweep cases for all .msh files in a given directory.

    Modes:
      - Single-stage: supply cfg_template (string) only.
      - Multi-stage : supply cfg_stage_templates (list of template paths). Each stage cfg is generated and run sequentially via run.sh.
    """
    os.makedirs(output_dir, exist_ok=True)

    multi_stage = cfg_stage_templates is not None and len(cfg_stage_templates) > 0
    if multi_stage:
        stage_texts = []
        for path in cfg_stage_templates:
            with open(path, 'r') as f:
                stage_texts.append(f.read())
        if stage_cfg_names is None:
            stage_cfg_names = [f"stage{i+1}.cfg" for i in range(len(stage_texts))]
    else:
        with open(cfg_template, 'r') as f:
            base_cfg = f.read()

    for fname in os.listdir(mesh_dir):
        if not fname.endswith('.msh'):
            continue
        mesh_path = os.path.join(mesh_dir, fname)
        mesh_base = os.path.splitext(fname)[0]

        for mach, temp, pressure in product(mach_numbers, inlet_temperatures, freestream_pressures):
            case_name = f"{mesh_base}_M{mach:.1f}_T{temp:.1f}_P{int(pressure)}".replace(".", "p")
            case_dir = os.path.join(output_dir, case_name)
            os.makedirs(case_dir, exist_ok=True)
            shutil.copy(mesh_path, os.path.join(case_dir, fname))

            if multi_stage:
                stage_paths = []
                for text, out_name in zip(stage_texts, stage_cfg_names):
                    mod_txt = modify_cfg(text, mach, temp, pressure, fname)
                    out_path = os.path.join(case_dir, out_name)
                    with open(out_path, 'w') as fc:
                        fc.write(mod_txt)
                    stage_paths.append(out_path)

                run_lines = ["#!/bin/bash", "set -e", "echo 'Starting multi-stage run (forward pipeline)'" ]
                for idx, sp in enumerate(stage_paths):
                    run_lines.append(f"echo '--- Stage {idx+1}: {os.path.basename(sp)}'")
                    run_lines.append(f"SU2_CFD {os.path.basename(sp)}")
                    if idx < len(stage_paths)-1:
                        # simple restart propagation if names differ
                        try:
                            with open(stage_paths[idx+1]) as nf:
                                nxt = nf.read()
                            with open(stage_paths[idx]) as cf:
                                cur = cf.read()
                            m_next = re.search(r"RESTART_FILENAME\s*=\s*([^\n\r]+)", nxt)
                            m_curr = re.search(r"RESTART_FILENAME\s*=\s*([^\n\r]+)", cur)
                            if m_next and m_curr:
                                rn = m_next.group(1).strip()
                                rc = m_curr.group(1).strip()
                                if rn != rc:
                                    run_lines.append(f"cp {rc} {rn} 2>/dev/null || true")
                        except Exception:
                            pass
                with open(os.path.join(case_dir, "run.sh"), 'w') as fr:
                    fr.write("\n".join(run_lines) + "\n")
                os.chmod(os.path.join(case_dir, "run.sh"), 0o755)

                # SLURM script mirrors run.sh sequential execution
                slurm_content = [
                    "#!/bin/bash",
                    f"#SBATCH --job-name={case_name}",
                    "#SBATCH --output=output.log",
                    "#SBATCH --error=error.log",
                    f"#SBATCH --time={slurm_time}",
                    f"#SBATCH --partition={slurm_partition}",
                    f"#SBATCH --nodes={slurm_nodes}",
                    f"#SBATCH --ntasks={slurm_ntasks}",
                    "",
                    module_load,
                ] + [line for line in run_lines if not line.startswith('#!')]
                with open(os.path.join(case_dir, "submit.slurm"), 'w') as fsb:
                    fsb.write("\n".join(slurm_content) + "\n")
            else:
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

    mode = 'multi-stage' if multi_stage else 'single-stage'
    print(f"\n+++ SU2 cases ({mode}) created in: {output_dir}/")

# CLI usage for backwards compatibility
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate SU2 sweep cases for mesh files.")
    parser.add_argument("mesh_dir", help="Directory containing .msh mesh files")
    parser.add_argument("output_dir", help="Directory to output generated cases")
    parser.add_argument("--cfg_template", default="./double_ramp_configuration/inputs/inv_wedge_HLLC.cfg", help="Path to SU2 config template")
    parser.add_argument("--mach_numbers", nargs="+", type=float, default=[2.0, 2.5, 3.0], help="Mach numbers (space separated)")
    parser.add_argument("--inlet_temperatures", nargs="+", type=float, default=[250.0, 275.0, 300.0], help="Inlet temperatures (space separated)")
    parser.add_argument("--freestream_pressures", nargs="+", type=float, default=[90000.0, 101325.0, 110000.0], help="Freestream pressures (space separated)")
    args = parser.parse_args()

    generate_sweeps_for_mesh_folder(
        args.mesh_dir,
        args.output_dir,
        args.cfg_template,
        args.mach_numbers,
        args.inlet_temperatures,
        args.freestream_pressures,
    )
