import gmsh
import math
import sys

gmsh.initialize()
gmsh.model.add("ramp_python")

gmsh.model.geo.addPoint(0, 0, 0, 1, 1)
gmsh.model.geo.addPoint(0.5, 0, 0, 1, 2)
gmsh.model.geo.addPoint(1.5, 0.2309, 0, 1, 3)
gmsh.model.geo.addPoint(1.5, 0.9342, 0, 1, 4)
gmsh.model.geo.addPoint(0, 0.9342, 0, 1, 5)

gmsh.model.geo.addLine(1, 5, 1)
gmsh.model.geo.addLine(5, 4, 2)
gmsh.model.geo.addLine(4, 3, 3)
gmsh.model.geo.addLine(3, 2, 4)
gmsh.model.geo.addLine(2, 1, 5)

gmsh.model.geo.addCurveLoop([2, 3, 4, 5, 1], 1)
gmsh.model.geo.addPlaneSurface([1], 1)

# Synchronize before using mesh and physical APIs
gmsh.model.geo.synchronize()

gmsh.model.addPhysicalGroup(1, [1], 6)  # Inlet
gmsh.model.setPhysicalName(1, 6, "Inlet")

gmsh.model.addPhysicalGroup(1, [3], 7)  # Outlet
gmsh.model.setPhysicalName(1, 7, "Outlet")

gmsh.model.addPhysicalGroup(1, [2, 5, 4], 8)  # Wall
gmsh.model.setPhysicalName(1, 8, "Wall")

gmsh.model.mesh.setTransfiniteSurface(1, cornerTags=[5, 4, 3, 1])

gmsh.model.mesh.setTransfiniteCurve(1, 201, coef=1) # Inlet
gmsh.model.mesh.setTransfiniteCurve(3, 201, coef=1) # Outlet
gmsh.model.mesh.setTransfiniteCurve(5, 51, coef=1)  # Aft Wall - Flat Part
gmsh.model.mesh.setTransfiniteCurve(4, 101, coef=1) # Aft Wall - Ramp
gmsh.model.mesh.setTransfiniteCurve(2, 151, coef=1) # Upper Wall

# Recombine surface to generate quadrangles
gmsh.model.mesh.setRecombine(2, 1)  # 2 = surface dimension, 1 = tag of the surface

# Generate the mesh
gmsh.model.mesh.generate(2)

# to ensure compatibility
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.option.setNumber("Mesh.SaveAll", 1)
# Save and finalize
gmsh.write("./thesis/ramp_python_v1.msh")

# Launch the GUI to see the results:
if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()

#gmsh.finalize()
