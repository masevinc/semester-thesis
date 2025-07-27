import gmsh
import math
import os
from create_dramp_points import create_double_ramp_points
from itertools import product

gmsh.initialize()

def create_geometry(ramp_1_angle_deg, ramp_2_angle_deg, mesh_id, export_typ='.msh'):
    gmsh.model.add(f"ramp_mesh_{mesh_id}")
    
    ramp_params = {
    "h_top": 1.0,
    "flat1": 0.2,
    "theta1_deg": ramp_1_angle_deg,
    "ramp1_len": 0.2,
    "flat2": 0.15,
    "theta2_deg": ramp_2_angle_deg,
    "ramp2_len": 0.3,
    "domain_len": 1.0,
    "mesh_size": 1.0,
    "z": 0.0,
    "start_tag": 1
}

    tags, coors = create_double_ramp_points(ramp_params)
    
    # gmsh.model.geo.addPoint(0, 0, 0, 1.0, 1)
    # gmsh.model.geo.addPoint(0.5, 0, 0, 1.0, 2)

    # # Calculate new Y for ramp based on angle
    # angle_rad = math.radians(ramp_angle_deg)
    # ramp_length = 1.0
    # ramp_x = 1.5
    # ramp_y = math.tan(angle_rad) * (ramp_x - 0.5) + 0  # Assuming base at y=0

    # gmsh.model.geo.addPoint(ramp_x, ramp_y, 0, 1.0, 3)
    # gmsh.model.geo.addPoint(ramp_x, 0.9342, 0, 1.0, 4)
    # gmsh.model.geo.addPoint(0, 0.9342, 0, 1.0, 5)

    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 5, 4)
    gmsh.model.geo.addLine(5, 6, 5)
    gmsh.model.geo.addLine(6, 7, 6)
    gmsh.model.geo.addLine(7, 8, 7)
    gmsh.model.geo.addLine(8, 1, 8)


    gmsh.model.geo.addCurveLoop([8, 1, 2, 3, 4, 5, 6, 7], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)

    # Synchronize before using mesh and physical APIs
    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(1, [8],9)  # Inlet
    gmsh.model.setPhysicalName(1, 9, "Inlet")

    gmsh.model.addPhysicalGroup(1, [6], 10)  # Outlet
    gmsh.model.setPhysicalName(1, 10, "Outlet")

    gmsh.model.addPhysicalGroup(1, [1, 2, 3, 4, 5, 7], 11)  # Wall
    gmsh.model.setPhysicalName(1, 11, "Wall")

    gmsh.model.mesh.setTransfiniteSurface(1, cornerTags=[8, 7, 6,1])

    gmsh.model.mesh.setTransfiniteCurve(8, 201, coef=1) # Inlet
    gmsh.model.mesh.setTransfiniteCurve(6, 201, coef=1) # Outlet

    gmsh.model.mesh.setTransfiniteCurve(1, 31, coef=1)  # Upper Wall - 1
    gmsh.model.mesh.setTransfiniteCurve(2, 31, coef=1) # Upper Wall - 2
    gmsh.model.mesh.setTransfiniteCurve(3, 21, coef=1) # Upper Wall - 3
    gmsh.model.mesh.setTransfiniteCurve(4, 31, coef=1) # Upper Wall - 4
    gmsh.model.mesh.setTransfiniteCurve(5, 41, coef=1) # Upper Wall - 5

    gmsh.model.mesh.setTransfiniteCurve(7, 151, coef=1) # Aft Wall

    # Recombine surface to generate quadrangles
    gmsh.model.mesh.setRecombine(2, 1)  # 2 = surface dimension, 1 = tag of the surface

    # Generate the mesh
    gmsh.model.mesh.generate(2)
    # to ensure compatibility
    if export_typ == '.msh':
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.SaveAll", 1)

    mesh_filename = f"./double_ramp_automatize/output_auto/mesh_angle_r1_{ramp_1_angle_deg}_r2_{ramp_2_angle_deg}{export_typ}"
    gmsh.write(mesh_filename)
    print(f"Generated: {mesh_filename}")

    gmsh.model.remove()

# List of angles to generate
angles_1 = [0, 5, 10, 15, 20] # Ramp 1 angles
angles_2 = [2.5, 7.5, 12.5, 17.5, 22.5] # Ramp 2 angles

export_typ = '.msh' # or '.msh' , '.cgns'

for idx, (angle1, angle2) in enumerate(product(angles_1, angles_2)):
    create_geometry(angle1, angle2, idx, export_typ)

gmsh.finalize()
