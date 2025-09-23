import os
import sys

# Allow running this script from project root by adding its own src directory to sys.path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_THIS_DIR, 'src')
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from src.extract_points_s1 import extract_points_batch  # type: ignore  # noqa: E402
from src.evaluation_viz import (  # noqa: E402
    visualize_first_extracted_case,  # type: ignore
    visualize_all_extracted_cases    # type: ignore
)  # Visualization utilities

USE_RANDOM = True    # Toggle this to enable/disable random sampling easily
DO_EVAL_VIZ = True   # If True: produce evaluation overlays after extraction
VIZ_ALL_CASES = True # If True: visualize every extracted case; if False: only first
MAX_VIZ_CASES = None # Optional int limit (e.g., 50) when VIZ_ALL_CASES True; None = no limit
DO_MESH = True      # If True: proceed to mesh & sweep stage (requires gmsh). False = extraction + viz only
SAMPLE_SIZE = 50     # Only used if USE_RANDOM=True (or legacy if left True with value)
RANDOM_SEED = 87
MESH_FORMAT = 'su2'  # choose among: 'msh', 'su2', 'cgns'
WRITE_LOCAL_SWEEP_SCRIPT = True  # new: create run_all_local.sh for sequential local execution

# Mesh resolution controls (threaded through to mesh generator)
VERTICAL_NODES = 201
HORIZONTAL_TARGET_NODES = 521  # total nodes along each top/bottom chain target (will be proportionally split)

DATA_DIR = "./double_ramp_configuration/inputs/double_ramp_npz_files_clamped"
POINTS_DIR = "./double_ramp_configuration/outputs/backward/extracted_points"
# --- Geometry scaling configuration ---
# Legacy mode: domain treated as square (height == width == PHYSICAL_HEIGHT)
# New optional rectangular scaling: specify physical width & height explicitly.
USE_RECTANGULAR_PHYSICAL_SCALE = True  # <- Toggle this to True to enable 0.62 x 0.40 scaling
PHYSICAL_HEIGHT = 1                     # Legacy reference size (kept for backward compatibility)
RECT_PHYSICAL_HEIGHT = 0.40             # New physical height (y direction) when rectangular scaling is on
RECT_PHYSICAL_WIDTH = 0.62              # New physical width  (x direction) when rectangular scaling is on

if USE_RECTANGULAR_PHYSICAL_SCALE:
    _phys_height = RECT_PHYSICAL_HEIGHT
    _phys_width = RECT_PHYSICAL_WIDTH
else:
    _phys_height = PHYSICAL_HEIGHT
    _phys_width = None  # None signals square scaling internally

extract_points_batch(
    data_dir=DATA_DIR,
    output_dir=POINTS_DIR,
    filters={"ramp1": None, "ramp2": None, "min_ma": None, "max_ma": None},
    selected_keys=["temperature"],
    physical_height=_phys_height,
    physical_width=_phys_width,
    clear_output_before_run=True,
    enable_random_sampling=USE_RANDOM,
    sample_size=SAMPLE_SIZE,
    random_seed=RANDOM_SEED,
    manifest_filename="selection_manifest.csv",
    manifest_in_parent=True
)

if DO_EVAL_VIZ:
    viz_out_dir = "./double_ramp_configuration/outputs/backward/evaluation_viz"
    # Clean previous evaluation visualization outputs to avoid clutter
    if os.path.isdir(viz_out_dir):
        for _name in os.listdir(viz_out_dir):
            _p = os.path.join(viz_out_dir, _name)
            try:
                if os.path.isfile(_p) or os.path.islink(_p):
                    os.unlink(_p)
                elif os.path.isdir(_p):
                    # Unlikely, but handle nested dirs just in case
                    import shutil
                    shutil.rmtree(_p)
            except Exception as _e:
                print(f"[main] Warning: failed to delete {_p}: {_e}")
    else:
        os.makedirs(viz_out_dir, exist_ok=True)
    if VIZ_ALL_CASES:
        visualize_all_extracted_cases(
            data_dir=DATA_DIR,
            points_dir=POINTS_DIR,
            output_dir=viz_out_dir,
            physical_height=_phys_height,
            physical_width=_phys_width,
            data_key="temperature",
            max_cases=MAX_VIZ_CASES
        )
    else:
        visualize_first_extracted_case(
            data_dir=DATA_DIR,
            points_dir=POINTS_DIR,
            output_dir=viz_out_dir,
            physical_height=_phys_height,
            physical_width=_phys_width,
            data_key="temperature",
            figure_name="first_case_overlay"
        )

if DO_MESH:
    try:
        from src.run_automation_s2 import main as run_mesh_automation  # type: ignore
    except ModuleNotFoundError as e:
        print("[main] Mesh stage skipped (missing dependency):", e)
    else:
        run_mesh_automation(
            points_dir=POINTS_DIR,
            mesh_dir="./double_ramp_configuration/outputs/backward/mesh",
            error_log="./double_ramp_configuration/outputs/backward/mesh_errors.csv",
            expected_num_points=12,
            run_sweep=True,
            sweep_cfg_template="./double_ramp_configuration/inputs/hybrid_dbl_ramp.cfg",  # or inv_wedge_HLLC.cfg
            sweep_output_root="./double_ramp_configuration/outputs/backward/sweep",
            sweep_inlet_temperatures=[300.0],  # adjust list as desired
            sweep_slurm_partition="standard",
            sweep_slurm_time="01:00:00",
            sweep_slurm_nodes=1,
            sweep_slurm_ntasks=4,
            sweep_module_load="module load su2/8.0.0",
            sweep_clear_output_before_run=True,
            sweep_write_master_slurm_script=True,
            mesh_format=MESH_FORMAT,
            sweep_write_master_local_script=WRITE_LOCAL_SWEEP_SCRIPT,
            vertical_nodes=VERTICAL_NODES,
            horizontal_target_nodes=HORIZONTAL_TARGET_NODES,
        )
else:
    print("[main] DO_MESH=False -> Skipping mesh + sweep stage (extraction + evaluation only).")