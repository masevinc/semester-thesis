# generate_mesh.py

import gmsh
import math

def get_mesh_filename(angle, export_typ='.su2'):
    return f"./thesis/ramp_automatize/mesh_angle_{angle}{export_typ}"

def create_geometry(ramp_angle_deg, mesh_id=None, export_typ='.su2'):
    gmsh.initialize()
    gmsh.model.add(f"ramp_mesh_{mesh_id or ramp_angle_deg}")

    # Define geometry
    gmsh.model.geo.addPoint(0, 0, 0, 1.0, 1)
    gmsh.model.geo.addPoint(0.5, 0, 0, 1.0, 2)

    angle_rad = math.radians(ramp_angle_deg)
    ramp_x = 1.5
    ramp_y = math.tan(angle_rad) * (ramp_x - 0.5)
    gmsh.model.geo.addPoint(ramp_x, ramp_y, 0, 1.0, 3)
    gmsh.model.geo.addPoint(ramp_x, 0.9342, 0, 1.0, 4)
    gmsh.model.geo.addPoint(0, 0.9342, 0, 1.0, 5)

    # Lines & surface
    gmsh.model.geo.addLine(1, 5, 1)
    gmsh.model.geo.addLine(5, 4, 2)
    gmsh.model.geo.addLine(4, 3, 3)
    gmsh.model.geo.addLine(3, 2, 4)
    gmsh.model.geo.addLine(2, 1, 5)
    gmsh.model.geo.addCurveLoop([2, 3, 4, 5, 1], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)

    gmsh.model.geo.synchronize()

    # Physical markers
    gmsh.model.addPhysicalGroup(1, [1], 6)
    gmsh.model.setPhysicalName(1, 6, "Inlet")
    gmsh.model.addPhysicalGroup(1, [3], 7)
    gmsh.model.setPhysicalName(1, 7, "Outlet")
    gmsh.model.addPhysicalGroup(1, [2, 5, 4], 8)
    gmsh.model.setPhysicalName(1, 8, "Wall")

    # Mesh settings
    gmsh.model.mesh.setTransfiniteSurface(1, cornerTags=[5, 4, 3, 1])
    gmsh.model.mesh.setTransfiniteCurve(1, 201)
    gmsh.model.mesh.setTransfiniteCurve(3, 201)
    gmsh.model.mesh.setTransfiniteCurve(5, 51)
    gmsh.model.mesh.setTransfiniteCurve(4, 101)
    gmsh.model.mesh.setTransfiniteCurve(2, 151)
    gmsh.model.mesh.setRecombine(2, 1)

    gmsh.model.mesh.generate(2)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.SaveAll", 1)

    mesh_filename = get_mesh_filename(ramp_angle_deg, export_typ)
    gmsh.write(mesh_filename)
    print(f"✅ Mesh generated: {mesh_filename}")

    gmsh.finalize()
    return mesh_filename
