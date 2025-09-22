from src.dramp_mesh_s1 import generate_mesh_sweep
from src.dramp_sweep_calc import generate_sweeps_for_mesh_folder

# Toggle single vs multi-stage SU2 execution
USE_TWO_STAGE_CFG = True
SINGLE_STAGE_TEMPLATE = "./double_ramp_configuration/inputs/inv_wedge_HLLC.cfg"
TWO_STAGE_TEMPLATES = [
    "./double_ramp_configuration/inputs/double_ramp_stage1.cfg",
    "./double_ramp_configuration/inputs/double_ramp_stage2.cfg",
]

mult = 256
mesh_params = {
    "h_top": 1.0 * mult,
    "flat1": 0.2 * mult,
    "ramp1_len": 0.2 * mult,
    "flat2": 0.15 * mult,
    "ramp2_len": 0.3 * mult,
    "domain_len": 1.0 * mult,
    "mesh_size": 1.0 * mult,
    "z": 0.0,
    "start_tag": 1
}

generate_mesh_sweep(
    output_dir="././double_ramp_configuration/outputs/forward/mesh",
    export_type=".msh",
    angles_1=[0, 5, 10, 15, 20],
    angles_2=[2.5, 7.5, 12.5, 17.5, 22.5],
    mult=mult,
    mesh_params=mesh_params,
    clear_output_before_run=True
)

# Choose appropriate template argument
if USE_TWO_STAGE_CFG:
    cfg_template_arg = None
    cfg_stage_templates_arg = TWO_STAGE_TEMPLATES
else:
    cfg_template_arg = SINGLE_STAGE_TEMPLATE
    cfg_stage_templates_arg = None

generate_sweeps_for_mesh_folder(
    mesh_dir="./double_ramp_configuration/outputs/forward/mesh",
    output_dir="./double_ramp_configuration/outputs/forward/sweep",
    cfg_template=cfg_template_arg if cfg_template_arg else SINGLE_STAGE_TEMPLATE,
    mach_numbers=[2.0, 2.5, 3.0],
    inlet_temperatures=[250.0, 275.0, 300.0],
    freestream_pressures=[90000.0, 101325.0, 110000.0],
    cfg_stage_templates=cfg_stage_templates_arg,
)