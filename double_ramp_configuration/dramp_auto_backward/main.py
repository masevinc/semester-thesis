import os
import sys
import inspect

# --- Environment validation (to prevent accidental use of wrong interpreter) ---
# This ensures VS Code 'Run' button (which may launch a cached/older interpreter) uses
# the expected conda environment. Adjust EXPECTED_* if you intentionally change versions.
EXPECTED_ENV_NAME = "dramp"
EXPECTED_PYTHON_MAJOR = 3
EXPECTED_PYTHON_MINOR = 11
EXPECTED_NUMPY_MAJOR = 1
EXPECTED_NUMPY_MINOR = 26
EXPECTED_MPL_MAJOR = 3

def _env_guard():
    import numpy as _np
    import matplotlib as _mpl
    py_major, py_minor = sys.version_info[:2]
    problems = []
    # Detect conda env name from path if possible
    active_prefix = os.environ.get('CONDA_PREFIX', '')
    if EXPECTED_ENV_NAME not in active_prefix:
        problems.append(f"Not running in expected conda env '{EXPECTED_ENV_NAME}' (CONDA_PREFIX={active_prefix or 'unset'}).")
    if (py_major, py_minor) != (EXPECTED_PYTHON_MAJOR, EXPECTED_PYTHON_MINOR):
        problems.append(f"Python version {py_major}.{py_minor} != expected {EXPECTED_PYTHON_MAJOR}.{EXPECTED_PYTHON_MINOR}.")
    np_version = tuple(int(x) for x in _np.__version__.split('.')[:2])
    if np_version != (EXPECTED_NUMPY_MAJOR, EXPECTED_NUMPY_MINOR):
        problems.append(f"NumPy version {_np.__version__} != expected {EXPECTED_NUMPY_MAJOR}.{EXPECTED_NUMPY_MINOR}.x")
    mpl_version = tuple(int(x) for x in _mpl.__version__.split('.')[:1])  # major only for robustness
    if mpl_version[0] != EXPECTED_MPL_MAJOR:
        problems.append(f"Matplotlib major {_mpl.__version__} != expected {EXPECTED_MPL_MAJOR}.x")
    if problems:
        print("[env][warn] Environment mismatch detected:\n  - " + "\n  - ".join(problems))
        print(f"[env][info] current python: {sys.executable}")
        # Attempt auto re-exec if allowed
        if not os.environ.get('NO_AUTO_REEXEC'):
            # Heuristic expected path
            expected_python = f"/opt/anaconda3/envs/{EXPECTED_ENV_NAME}/bin/python"
            if os.path.exists(expected_python):
                print(f"[env][action] Re-executing with expected interpreter: {expected_python}")
                os.execv(expected_python, [expected_python] + sys.argv)
            else:
                print(f"[env][warn] Expected interpreter not found at {expected_python}; cannot auto-switch.")
        print("[env][hint] Activate:  conda activate dramp")
        print("[env][hint] Or disable auto re-exec by setting NO_AUTO_REEXEC=1")
        sys.exit(2)

_env_guard()

# --- Path / import hygiene -----------------------------------------------------
# Original code inserted the *src* directory itself into sys.path and then imported
# using the prefix 'src.'. That only works if the *parent* of 'src' is on sys.path.
# Otherwise Python may resolve a totally different installed package named 'src'.
# We correct this by adding the parent directory ( _THIS_DIR ) and ensuring a
# package marker in ./src ( __init__.py already created ).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_THIS_DIR, 'src')
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# Fallback: if a conflicting external 'src' is ahead of ours, we force reload from path.
def _import_local(module_name: str):
    """Import a module guaranteed from our local src directory."""
    import importlib.util
    module_path = os.path.join(_SRC_DIR, module_name + '.py')
    if not os.path.isfile(module_path):
        raise ImportError(f"Local module {module_name} not found at {module_path}")
    spec = importlib.util.spec_from_file_location(f"_local_{module_name}", module_path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod

try:
    # Prefer normal relative-style import now that parent is on sys.path
    from src.extract_points_s1 import extract_points_batch  # type: ignore  # noqa: E402
    from src.evaluation_viz import (  # type: ignore  # noqa: E402
        visualize_first_extracted_case,
        visualize_all_extracted_cases
    )
except Exception:
    # Force local fallback (guards against picking up an unrelated pip package named 'src')
    extract_points_s1 = _import_local('extract_points_s1')
    evaluation_viz = _import_local('evaluation_viz')
    extract_points_batch = extract_points_s1.extract_points_batch  # type: ignore
    visualize_first_extracted_case = evaluation_viz.visualize_first_extracted_case  # type: ignore
    visualize_all_extracted_cases = evaluation_viz.visualize_all_extracted_cases    # type: ignore

# Debug: show actual file locations to verify no external package confusion.
try:
    import src.cv_processing as _cvp  # type: ignore
    print(f"[debug] Using cv_processing from: {inspect.getfile(_cvp)}")
except Exception as _e_dbg:
    print("[debug] Failed to import cv_processing via 'src.':", _e_dbg)
    try:
        _cvp_local = _import_local('cv_processing')
        print(f"[debug] Fallback local cv_processing path: {getattr(_cvp_local, '__file__', '?')}")
    except Exception as _e_dbg2:
        print("[debug] Could not load local cv_processing either:", _e_dbg2)

USE_RANDOM = False    # Toggle this to enable/disable random sampling easily
DO_EVAL_VIZ = True   # If True: produce evaluation overlays after extraction
VIZ_ALL_CASES = True # If True: visualize every extracted case; if False: only first
MAX_VIZ_CASES = None # Optional int limit (e.g., 50) when VIZ_ALL_CASES True; None = no limit
DO_MESH = True      # If True: proceed to mesh & sweep stage (requires gmsh). False = extraction + viz only
SAMPLE_SIZE = 4     # Only used if USE_RANDOM=True (or legacy if left True with value)
RANDOM_SEED = 11
MESH_FORMAT = 'cgns'  # choose among: 'msh', 'su2', 'cgns'
WRITE_LOCAL_SWEEP_SCRIPT = True  # new: create run_all_local.sh for sequential local execution

# Mesh resolution controls (threaded through to mesh generator)
VERTICAL_NODES = 201
HORIZONTAL_TARGET_NODES = 521  # total nodes along each top/bottom chain target (will be proportionally split)

# --- Optional: Mesh convergence study toggle & parameters ---
ENABLE_MESH_CONVERGENCE = False
# Variations to try (you can also provide explicit pair list below)
MESH_CONV_VERTICAL_LIST = [101, 151, 201, 301,401]
MESH_CONV_HORIZONTAL_LIST = [321, 421, 521, 621,721]
# Alternatively, define exact pairs like: [(101,321), (151,421), (201,521)]
MESH_CONV_EXPLICIT_PAIRS = None
# Choose which extracted case to use; if None, first .npy in points dir is picked
MESH_CONV_PREFERRED_POINTS = None  # e.g. "double_ramp_0p012_0p034_interpolated_arrays_density.npy"
# Post-process sampling settings
MESH_CONV_FIELD = 'temperature'  # e.g., 'density', 'pressure', 'temperature'
MESH_CONV_Y_VALUE = 0.20  # y=constant line to sample in physical units
MESH_CONV_DO_GENERATE = False    # create meshes & cases
MESH_CONV_DO_POST = True       # set True after runs finished to collect VTUs and plot overlay

DATA_DIR = './double_ramp_configuration/inputs/npz_recons_input'#'./double_ramp_configuration/inputs/double_ramp_npz_files_clamped' #"./double_ramp_configuration/inputs/denorm/DDPM_semi"
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

        # Optional mesh convergence study, disabled by default
        if ENABLE_MESH_CONVERGENCE:
            try:
                from src.mesh_convergence import run_mesh_convergence  # type: ignore
            except Exception as e:
                print("[main] Mesh convergence feature unavailable:", e)
            else:
                run_mesh_convergence(
                    points_dir=POINTS_DIR,
                    preferred_points_file=MESH_CONV_PREFERRED_POINTS,
                    work_root="./double_ramp_configuration/outputs/backward/mesh_convergence_work",
                    resolutions_vertical=MESH_CONV_VERTICAL_LIST,
                    resolutions_horizontal=MESH_CONV_HORIZONTAL_LIST,
                    resolution_pairs=MESH_CONV_EXPLICIT_PAIRS,
                    mesh_format=MESH_FORMAT,
                    sweep_cfg_template="./double_ramp_configuration/inputs/hybrid_dbl_ramp.cfg",
                    inlet_temperatures=[300.0],
                    su2_output_root="./double_ramp_configuration/outputs/backward/mesh_convergence",
                    field=MESH_CONV_FIELD,
                    y_value=MESH_CONV_Y_VALUE,
                    do_generate=MESH_CONV_DO_GENERATE,
                    do_postprocess=MESH_CONV_DO_POST,
                    slurm_partition="standard",
                    slurm_time="01:00:00",
                    slurm_nodes=1,
                    slurm_ntasks=4,
                    module_load="module load su2/8.0.0",
                )
else:
    print("[main] DO_MESH=False -> Skipping mesh + sweep stage (extraction + evaluation only).")