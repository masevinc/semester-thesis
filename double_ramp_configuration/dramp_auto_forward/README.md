# Double Ramp Pipeline

## Overview
The Double Ramp Pipeline is a computational framework designed to generate double ramp geometries and meshes using Gmsh, and subsequently process these meshes for computational fluid dynamics (CFD) simulations. This project automates the workflow from geometry generation to preparing simulation cases for high-performance computing (HPC) environments.

## Project Structure
The project consists of the following files:

- **main.py**: The master script that gathers input selections for the pipeline, validates the inputs, and runs the `dramp_mesh_s1.py` and `dramp_forward_auto_sweep_s2.py` scripts sequentially.
  
- **dramp_mesh_s1.py**: Generates double ramp geometries and meshes for a sweep of ramp angles. It includes configuration settings for output directories, ramp angles, mesh parameters, and functions to create geometries and meshes.

- **dramp_forward_auto_sweep_s2.py**: Processes the generated mesh files for the forward case, creates sweep folders and cases, and prepares them for HPC runs. It includes functions to clear output directories, generate sweep cases, and write SLURM scripts.

- **dramp_sweep_calc.py**: Generates sweep cases for all .msh files in a specified directory. It includes functions to compute velocities, modify configuration files, and write SLURM scripts for job submission.

- **dramp_points.py**: Defines a function to create points for a double ramp geometry based on input parameters. It computes the coordinates of the points and adds them to Gmsh.


## Usage Instructions
1. **Setup Environment**: Ensure that you have Python and Gmsh installed on your system. You may also need to install any required Python packages.

2. **Run the Pipeline**:
   - Execute the `main.py` script to start the pipeline. This script will handle the generation of geometries and meshes, as well as the preparation of sweep cases for HPC runs.
   - The script will automatically run `dramp_mesh_s1.py` to generate the necessary meshes and then proceed to run `dramp_forward_auto_sweep_s2.py` to prepare the simulation cases.

3. **Check Output**: After running the pipeline, check the output directories specified in the configuration settings within the scripts for generated meshes and sweep cases.

## Notes
- Ensure that the configuration settings in the scripts are adjusted according to your specific requirements before running the pipeline.
- For any issues or questions, please refer to the documentation within each script or contact the project maintainer.