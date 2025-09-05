from src.extract_points_s1 import extract_points_batch
from src.run_automation_s2 import main

USE_RANDOM = True   # Toggle this to enable/disable random sampling easily
SAMPLE_SIZE = 50     # Only used if USE_RANDOM=True (or legacy if left True with value)
RANDOM_SEED = 123
MESH_FORMAT = 'su2'  # choose among: 'msh', 'su2', 'cgns'

extract_points_batch(
    data_dir="./double_ramp_configuration/inputs/double_ramp_npz_files_clamped",
    output_dir="./double_ramp_configuration/outputs/backward/extracted_points",
    filters={"ramp1": 0.011, "ramp2": None, "min_ma": None, "max_ma": None},
    selected_keys=["density"],
    physical_height=256,
    clear_output_before_run=True,
    enable_random_sampling=USE_RANDOM,
    sample_size=SAMPLE_SIZE,
    random_seed=RANDOM_SEED,
    manifest_filename="selection_manifest.csv",
    manifest_in_parent=True
)

main(
    points_dir="./double_ramp_configuration/outputs/backward/extracted_points",
    mesh_dir="./double_ramp_configuration/outputs/backward/mesh",
    error_log="./double_ramp_configuration/outputs/backward/mesh_errors.csv",
    expected_num_points=12,
    run_sweep=True,
    sweep_cfg_template="./double_ramp_configuration/inputs/inv_wedge_HLLC.cfg",
    sweep_output_root="./double_ramp_configuration/outputs/backward/sweep",
    sweep_inlet_temperatures=[250.0, 275.0, 300.0],
    sweep_slurm_partition="standard",
    sweep_slurm_time="01:00:00",
    sweep_slurm_nodes=1,
    sweep_slurm_ntasks=4,
    sweep_module_load="module load su2/8.1.0",
    sweep_clear_output_before_run=True,
    sweep_write_master_slurm_script=True,
    mesh_format=MESH_FORMAT
)