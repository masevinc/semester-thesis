import gmsh
import math
import sys
from point_transfer import generate_gmsh_point_code
from binary_mask_dr import process_image


gmsh.initialize()
gmsh.model.add("double_ramp_python")

points = process_image(
    input_image_path="./double_ramp_configuration/dramp_auto_backwards/density_plot.png", 
    physical_domain_height=1.0,
    save_steps=False,
)

# Generate Gmsh point definitions
gmsh_code_lines = generate_gmsh_point_code(points)

# Execute each generated line dynamically
for line in gmsh_code_lines:
    exec(line)

# gmsh.model.geo.addPoint(0, 1, 0, 1, 1)
# gmsh.model.geo.addPoint(0.2, 1, 0, 1, 2)
# gmsh.model.geo.addPoint(0.4, 0.9, 0, 1, 3)
# gmsh.model.geo.addPoint(0.55, 0.9, 0, 1, 4)
# gmsh.model.geo.addPoint(0.7, 0.75, 0, 1, 5)
# gmsh.model.geo.addPoint(1, 0.75, 0, 1, 6)
# gmsh.model.geo.addPoint(1, 0, 0, 1, 7)
# gmsh.model.geo.addPoint(0, 0, 0, 1, 8)

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
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.option.setNumber("Mesh.SaveAll", 1)
# Save and finalize
gmsh.write("./double_ramp_configuration/dramp_auto_backwards/d_ramp_python_v2.msh") 

# Launch the GUI to see the results:
if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()

#gmsh.finalize()
