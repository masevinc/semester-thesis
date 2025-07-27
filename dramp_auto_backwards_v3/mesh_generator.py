# mesh_generator.py

import gmsh
from point_transfer import generate_gmsh_point_code  # Must return GMSH Python API lines


def generate_mesh_from_points(points, mesh_output_path, show_gui=False):
    gmsh.initialize()
    gmsh.model.add("double_ramp_python")

    # Generate Gmsh point definitions from (x, y)
    gmsh_code_lines = generate_gmsh_point_code(points)

    for line in gmsh_code_lines:
        exec(line)  # Add points dynamically

    # Add lines manually — assumes 12 points
    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 5, 4)
    gmsh.model.geo.addLine(5, 6, 5)
    gmsh.model.geo.addLine(6, 7, 6)
    gmsh.model.geo.addLine(7, 8, 7)
    gmsh.model.geo.addLine(8, 9, 8)
    gmsh.model.geo.addLine(9, 10, 9)
    gmsh.model.geo.addLine(10, 11, 10)
    gmsh.model.geo.addLine(11, 12, 11)
    gmsh.model.geo.addLine(12, 1, 12)

    gmsh.model.geo.addCurveLoop([12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)
    gmsh.model.geo.synchronize()

    # Physical groups
    gmsh.model.addPhysicalGroup(1, [12], 9)
    gmsh.model.setPhysicalName(1, 9, "Inlet")

    gmsh.model.addPhysicalGroup(1, [6], 10)
    gmsh.model.setPhysicalName(1, 10, "Outlet")

    gmsh.model.addPhysicalGroup(1, [1, 2, 3, 4, 5, 7, 8, 9, 10, 11], 11)
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
    # TODO: Here a better definition for node definement is needed - Hardcoded for now
    
    # Use quads
    gmsh.model.mesh.setRecombine(2, 1)

    # Generate mesh
    gmsh.model.mesh.generate(2)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.SaveAll", 1)
    gmsh.write(mesh_output_path)

    if show_gui:
        gmsh.fltk.run()

    gmsh.finalize()
