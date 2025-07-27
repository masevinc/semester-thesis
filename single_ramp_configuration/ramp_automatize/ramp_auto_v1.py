import gmsh
import math
import os

gmsh.initialize()

def create_geometry(ramp_angle_deg, mesh_id, export_typ='.msh'):
    gmsh.model.add(f"ramp_mesh_{mesh_id}")
    gmsh.model.geo.addPoint(0, 0, 0, 1.0, 1)
    gmsh.model.geo.addPoint(0.5, 0, 0, 1.0, 2)

    # Calculate new Y for ramp based on angle
    angle_rad = math.radians(ramp_angle_deg)
    ramp_length = 1.0
    ramp_x = 1.5
    ramp_y = math.tan(angle_rad) * (ramp_x - 0.5) + 0  # Assuming base at y=0

    gmsh.model.geo.addPoint(ramp_x, ramp_y, 0, 1.0, 3)
    gmsh.model.geo.addPoint(ramp_x, 0.9342, 0, 1.0, 4)
    gmsh.model.geo.addPoint(0, 0.9342, 0, 1.0, 5)

    # Lines
    gmsh.model.geo.addLine(1, 5, 1)
    gmsh.model.geo.addLine(5, 4, 2)
    gmsh.model.geo.addLine(4, 3, 3)
    gmsh.model.geo.addLine(3, 2, 4)
    gmsh.model.geo.addLine(2, 1, 5)

    # Surface
    gmsh.model.geo.addCurveLoop([2, 3, 4, 5, 1], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)

    gmsh.model.geo.synchronize()

    # Physical groups
    gmsh.model.addPhysicalGroup(1, [1], 6)
    gmsh.model.setPhysicalName(1, 6, "Inlet")

    gmsh.model.addPhysicalGroup(1, [3], 7)
    gmsh.model.setPhysicalName(1, 7, "Outlet")

    gmsh.model.addPhysicalGroup(1, [2, 5, 4], 8)
    gmsh.model.setPhysicalName(1, 8, "Wall")

    # Transfinite surface
    gmsh.model.mesh.setTransfiniteSurface(1, cornerTags=[5, 4, 3, 1])

    gmsh.model.mesh.setTransfiniteCurve(1, 201, coef=1)
    gmsh.model.mesh.setTransfiniteCurve(3, 201, coef=1)
    gmsh.model.mesh.setTransfiniteCurve(5, 51, coef=1)
    gmsh.model.mesh.setTransfiniteCurve(4, 101, coef=1)
    gmsh.model.mesh.setTransfiniteCurve(2, 151, coef=1)

    gmsh.model.mesh.setRecombine(2, 1)

    gmsh.model.mesh.generate(2)
    
    # to ensure compatibility
    if export_typ == '.msh':
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.SaveAll", 1)

    mesh_filename = f"./thesis/ramp_automatize/mesh_angle_{ramp_angle_deg}{export_typ}"
    gmsh.write(mesh_filename)
    print(f"Generated: {mesh_filename}")

    gmsh.model.remove()

# List of angles to generate
angles = [0, 5, 10, 15, 20]
export_typ = '.su2' # or '.msh' , '.cgns'

for idx, angle in enumerate(angles):
    create_geometry(angle, idx, export_typ)

gmsh.finalize()
