import gmsh
from create_dramp_points import create_double_ramp_points

gmsh.initialize()
gmsh.model.add("dummykekek")

ramp_params = {
    "h_top": 1.0,
    "flat1": 0.2,
    "theta1_deg": 30,
    "ramp1_len": 0.2,
    "flat2": 0.15,
    "theta2_deg": 45,
    "ramp2_len": 0.3,
    "domain_len": 1.0,
    "mesh_size": 1.0,
    "z": 0.0,
    "start_tag": 1
}

create_double_ramp_points(ramp_params)

gmsh.model.geo.synchronize()
gmsh.fltk.run()  # This opens the Gmsh GUI with your geometry
gmsh.finalize()
