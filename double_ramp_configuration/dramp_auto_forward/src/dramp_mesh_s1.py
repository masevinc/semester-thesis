"""

Step 1

dramp_mesh.py

Batch generates double ramp geometries and meshes using Gmsh, for a sweep of ramp angles.
All configuration is set via function arguments.
"""

import gmsh
import os
import shutil
from itertools import product
from src.dramp_points import create_double_ramp_points

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

def create_geometry(ramp_1_angle_deg, ramp_2_angle_deg, mesh_id, mult, mesh_params, output_dir, export_typ):
    """
    Creates and writes a double ramp mesh for the given angles.
    """
    gmsh.model.add(f"ramp_mesh_{mesh_id}")

    # Update ramp parameters for this geometry
    ramp_params = mesh_params.copy()
    ramp_params["theta1_deg"] = ramp_1_angle_deg
    ramp_params["theta2_deg"] = ramp_2_angle_deg

    tags, coors = create_double_ramp_points(ramp_params)

    # Add lines (assumes 12 points, 12 lines)
    for i in range(1, 13):
        gmsh.model.geo.addLine(i, i % 12 + 1, i)

    gmsh.model.geo.addCurveLoop(list(range(1, 13)), 1)
    gmsh.model.geo.addPlaneSurface([1], 1)

    # Synchronize before using mesh and physical APIs
    gmsh.model.geo.synchronize()

    # Physical groups
    gmsh.model.addPhysicalGroup(1, [12], 9)  # Inlet
    gmsh.model.setPhysicalName(1, 9, "Inlet")
    gmsh.model.addPhysicalGroup(1, [6], 10)  # Outlet
    gmsh.model.setPhysicalName(1, 10, "Outlet")
    gmsh.model.addPhysicalGroup(1, [1, 2, 3, 4, 5, 7, 8, 9, 10, 11], 11)  # Wall
    gmsh.model.setPhysicalName(1, 11, "Wall")

    # Transfinite mesh settings
    gmsh.model.mesh.setTransfiniteSurface(1, cornerTags=[12, 7, 6, 1])

    gmsh.model.mesh.setTransfiniteCurve(12, 201, coef=1)
    gmsh.model.mesh.setTransfiniteCurve(6, 201, coef=1)
    gmsh.model.mesh.setTransfiniteCurve(1, 11, coef=1)
    gmsh.model.mesh.setTransfiniteCurve(11, 11, coef=1)
    gmsh.model.mesh.setTransfiniteCurve(2, 31, coef=1)
    gmsh.model.mesh.setTransfiniteCurve(10, 31, coef=1)
    gmsh.model.mesh.setTransfiniteCurve(3, 21, coef=1)
    gmsh.model.mesh.setTransfiniteCurve(9, 21, coef=1)
    gmsh.model.mesh.setTransfiniteCurve(4, 31, coef=1)
    gmsh.model.mesh.setTransfiniteCurve(8, 31, coef=1)
    gmsh.model.mesh.setTransfiniteCurve(5, 61, coef=1)
    gmsh.model.mesh.setTransfiniteCurve(7, 61, coef=1)

    # Recombine surface to generate quadrangles
    gmsh.model.mesh.setRecombine(2, 1)

    # Generate the mesh
    gmsh.model.mesh.generate(2)

    # Set mesh file version and save all elements
    if export_typ == '.msh':
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.SaveAll", 1)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    mesh_filename = os.path.join(
        output_dir,
        f"mesh_angle_r1_{ramp_1_angle_deg}_r2_{ramp_2_angle_deg}{export_typ}"
    )
    gmsh.write(mesh_filename)
    print(f"Generated: {mesh_filename}")

    gmsh.model.remove()

def generate_mesh_sweep(
    output_dir,
    export_type,
    angles_1,
    angles_2,
    mult,
    mesh_params,
    clear_output_before_run=True
):
    """
    Batch generate double ramp meshes for all combinations of angles.
    """
    if clear_output_before_run:
        clear_output_directory(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    gmsh.initialize()
    try:
        for idx, (angle1, angle2) in enumerate(product(angles_1, angles_2)):
            create_geometry(angle1, angle2, idx, mult, mesh_params, output_dir, export_type)
    finally:
        gmsh.finalize()

# CLI usage for backwards compatibility
if __name__ == "__main__":
    # Default configuration (can be overridden by importing and calling generate_mesh_sweep)
    OUTPUT_DIR = "./double_ramp_configuration/outputs/forward/mesh"
    EXPORT_TYPE = ".msh"
    ANGLES_1 = [0, 5, 10, 15, 20]
    ANGLES_2 = [2.5, 7.5, 12.5, 17.5, 22.5]
    mult = 256
    MESH_PARAMS = {
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
    CLEAR_OUTPUT_BEFORE_RUN = True

    generate_mesh_sweep(
        output_dir=OUTPUT_DIR,
        export_type=EXPORT_TYPE,
        angles_1=ANGLES_1,
        angles_2=ANGLES_2,
        mult=mult,
        mesh_params=MESH_PARAMS,
        clear_output_before_run=CLEAR_OUTPUT_BEFORE_RUN
    )
