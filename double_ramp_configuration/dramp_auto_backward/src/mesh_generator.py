"""

Step 2.1

mesh_generator.py

"""

import gmsh
from src.point_transfer import generate_gmsh_point_code  # Must return GMSH Python API lines

DEFAULT_VERTICAL_NODES = 151
DEFAULT_HORIZONTAL_TARGET_NODES = 251

#  Helpers for proportional horizontal node allocation 
def _dist(p, q):
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return (dx * dx + dy * dy) ** 0.5

def _split_proportional(total: int, weights, minimum: int = 3):
    """Distribute an integer 'total' across len(weights) buckets in proportion
    to 'weights' using largest remainder method. Enforce at least 'minimum' per bucket.
    Returns a list of ints that sum to 'total'.
    """
    n = len(weights)
    if n == 0:
        return []
    # Avoid zero-sum weights -> equal split
    s = sum(weights)
    if s <= 0:
        weights = [1.0] * n
        s = float(n)

    raw = [total * (w / s) for w in weights]
    floors = [max(minimum, int(x)) for x in map(lambda v: int(v // 1), raw)]

    # Recompute floors without the premature max(minimum, ..) using a safer approach
    floors = []
    for r in raw:
        floors.append(int(r))
    # Ensure minimum constraint
    for i in range(n):
        if floors[i] < minimum:
            floors[i] = minimum

    current = sum(floors)
    # If we've exceeded total due to minimums, reduce from the smallest-weight buckets
    if current > total:
        excess = current - total
        order = sorted(range(n), key=lambda i: (weights[i], floors[i]))  # lowest weight first
        idx = 0
        while excess > 0 and idx < n:
            i = order[idx]
            if floors[i] > minimum:
                floors[i] -= 1
                excess -= 1
            else:
                idx += 1
        return floors

    # Distribute remaining using largest remainders
    remainder = total - current
    fracs = [raw[i] - int(raw[i]) for i in range(n)]
    order = sorted(range(n), key=lambda i: fracs[i], reverse=True)
    k = 0
    while remainder > 0 and k < n:
        floors[order[k]] += 1
        remainder -= 1
        k += 1
    return floors

def _compute_horizontal_counts(points, target_total_nodes: int = 257):
    """Compute node counts for the 5 horizontal segments so that the total
    number of nodes across the entire top/bottom chain equals target_total_nodes.

    Geometry (12 points, 12 boundary lines):
      - Top wall segments: curves 1..5  (pairs left-to-right)
      - Bottom wall segments: curves 11..7 (paired with 1..5 respectively)
      - Vertical inlet/outlet: curves 12 and 6

    We compute pair-wise average lengths (top_i with bottom_i) and split
    target_total_nodes + 4 across the 5 segments ("+4" compensates for shared
    junction nodes between the 5 consecutive segments).
    """
    # Expect 12 boundary points (x,y)
    if len(points) < 12:
        # Fallback to a simple equal split if input is unexpected
        per = target_total_nodes // 5
        rest = target_total_nodes - per * 5
        base = [per] * 5
        for i in range(rest):
            base[i] += 1
        return base

    # Points are assumed ordered as used for addLine(1-2, 2-3, ..., 12-1)
    p = points
    # Top 5 segments lengths: (1-2), (2-3), (3-4), (4-5), (5-6)
    top_len = [
        _dist(p[0], p[1]),
        _dist(p[1], p[2]),
        _dist(p[2], p[3]),
        _dist(p[3], p[4]),
        _dist(p[4], p[5]),
    ]
    # Bottom 5 segments lengths (paired left-to-right with top):
    # Pairs: (1,11), (2,10), (3,9), (4,8), (5,7)
    bottom_len = [
        _dist(p[10], p[11]),  # curve 11: (11-12)
        _dist(p[9], p[10]),   # curve 10: (10-11)
        _dist(p[8], p[9]),    # curve 9:  (9-10)
        _dist(p[7], p[8]),    # curve 8:  (8-9)
        _dist(p[6], p[7]),    # curve 7:  (7-8)
    ]

    # Average lengths as weights
    weights = [(t + b) * 0.5 for t, b in zip(top_len, bottom_len)]

    # Sum of per-segment node counts over 5 segments must be target + 4
    # because the concatenation shares 4 interior junction nodes.
    total_over_segments = target_total_nodes + 4
    counts = _split_proportional(total_over_segments, weights, minimum=3)
    return counts  # [n1, n2, n3, n4, n5]

def _compute_horizontal_counts_8p(points, target_total_nodes: int = 257):
    """Compute node counts for the 3 horizontal segments for 8-point geometry.

    Top segments: (1-2), (2-3), (3-4) -> curves 1,2,3
    Bottom segments (left-to-right): (7-8), (6-7), (5-6) -> curves 7,6,5
    Vertical: (8-1) curve 8, (4-5) curve 4
    """
    if len(points) < 8:
        per = target_total_nodes // 3
        rest = target_total_nodes - per * 3
        base = [per] * 3
        for i in range(rest):
            base[i] += 1
        return base
    p = points
    top_len = [
        _dist(p[0], p[1]),
        _dist(p[1], p[2]),
        _dist(p[2], p[3]),
    ]
    bottom_len = [
        _dist(p[6], p[7]),  # curve 7: (7-8)
        _dist(p[5], p[6]),  # curve 6: (6-7)
        _dist(p[4], p[5]),  # curve 5: (5-6)
    ]
    weights = [(t + b) * 0.5 for t, b in zip(top_len, bottom_len)]
    total_over_segments = target_total_nodes + (3 - 1)
    return _split_proportional(total_over_segments, weights, minimum=3)

def _compute_horizontal_counts_10p(points, target_total_nodes: int = 257):
    """Compute node counts for the 4 horizontal segments for 10-point geometry.

    Top segments: (1-2), (2-3), (3-4), (4-5) -> curves 1,2,3,4
    Bottom segments (left-to-right): (9-10), (8-9), (7-8), (6-7) -> curves 9,8,7,6
    Vertical: (10-1) curve 10, (5-6) curve 5
    """
    if len(points) < 10:
        per = target_total_nodes // 4
        rest = target_total_nodes - per * 4
        base = [per] * 4
        for i in range(rest):
            base[i] += 1
        return base
    p = points
    top_len = [
        _dist(p[0], p[1]),
        _dist(p[1], p[2]),
        _dist(p[2], p[3]),
        _dist(p[3], p[4]),
    ]
    bottom_len = [
        _dist(p[8], p[9]),  # curve 9: (9-10)
        _dist(p[7], p[8]),  # curve 8: (8-9)
        _dist(p[6], p[7]),  # curve 7: (7-8)
        _dist(p[5], p[6]),  # curve 6: (6-7)
    ]
    weights = [(t + b) * 0.5 for t, b in zip(top_len, bottom_len)]
    total_over_segments = target_total_nodes + (4 - 1)
    return _split_proportional(total_over_segments, weights, minimum=3)

def _prepare_output(mesh_output_path, mesh_format=None):
    # Determine format
    if mesh_format is None:
        # infer from extension
        ext = mesh_output_path.split('.')[-1].lower()
        mesh_format = ext  # e.g. msh or su2 or cgns
    if mesh_format == 'msh':
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.SaveAll", 1)
    return mesh_format

# 12 points expected for double ramp

def generate_mesh_from_points(
    points,
    mesh_output_path,
    show_gui: bool = False,
    mesh_format=None,
    vertical_nodes: int = DEFAULT_VERTICAL_NODES,
    horizontal_target_nodes: int = DEFAULT_HORIZONTAL_TARGET_NODES,
):
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
    gmsh.model.addPhysicalGroup(1, [12], 13)
    gmsh.model.setPhysicalName(1, 13, "Inlet")

    gmsh.model.addPhysicalGroup(1, [6], 14)
    gmsh.model.setPhysicalName(1, 14, "Outlet")

    gmsh.model.addPhysicalGroup(1, [1, 2, 3, 4, 5, 7, 8, 9, 10, 11], 15)
    gmsh.model.setPhysicalName(1, 15, "Wall")

    # Transfinite mesh settings
    gmsh.model.mesh.setTransfiniteSurface(1, cornerTags=[12, 7, 6, 1])
    # Vertical (inlet/outlet)
    gmsh.model.mesh.setTransfiniteCurve(12, vertical_nodes, coef=1)
    gmsh.model.mesh.setTransfiniteCurve(6, vertical_nodes, coef=1)

    # Horizontal: distribute to reach total of 257 nodes across the chain
    n1, n2, n3, n4, n5 = _compute_horizontal_counts(points, target_total_nodes=horizontal_target_nodes)
    # Pairs: (1,11), (2,10), (3,9), (4,8), (5,7)
    gmsh.model.mesh.setTransfiniteCurve(1, n1, coef=1)
    gmsh.model.mesh.setTransfiniteCurve(11, n1, coef=1)

    gmsh.model.mesh.setTransfiniteCurve(2, n2, coef=1)
    gmsh.model.mesh.setTransfiniteCurve(10, n2, coef=1)

    gmsh.model.mesh.setTransfiniteCurve(3, n3, coef=1)
    gmsh.model.mesh.setTransfiniteCurve(9, n3, coef=1)

    gmsh.model.mesh.setTransfiniteCurve(4, n4, coef=1)
    gmsh.model.mesh.setTransfiniteCurve(8, n4, coef=1)

    gmsh.model.mesh.setTransfiniteCurve(5, n5, coef=1)
    gmsh.model.mesh.setTransfiniteCurve(7, n5, coef=1)
    
    # Use quads
    gmsh.model.mesh.setRecombine(2, 1)

    # Generate mesh
    gmsh.model.mesh.generate(2)
    _prepare_output(mesh_output_path, mesh_format)
    gmsh.write(mesh_output_path)

    if show_gui:
        gmsh.fltk.run()

    gmsh.finalize()

# 8 points expected for single ramp

def generate_mesh_from_points_8pnt(
    points,
    mesh_output_path,
    show_gui: bool = False,
    mesh_format=None,
    vertical_nodes: int = DEFAULT_VERTICAL_NODES,
    horizontal_target_nodes: int = DEFAULT_HORIZONTAL_TARGET_NODES,
):
    gmsh.initialize()
    gmsh.model.add("8pnt_ramp_python")

    # Generate Gmsh point definitions from (x, y)
    gmsh_code_lines = generate_gmsh_point_code(points)

    for line in gmsh_code_lines:
        exec(line)  # Add points dynamically

    # Add lines manually — assumes 8 points
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
    gmsh.model.geo.synchronize()

    # Physical groups
    gmsh.model.addPhysicalGroup(1, [8], 9)
    gmsh.model.setPhysicalName(1, 9, "Inlet")

    gmsh.model.addPhysicalGroup(1, [4], 10)
    gmsh.model.setPhysicalName(1, 10, "Outlet")

    gmsh.model.addPhysicalGroup(1, [1, 2, 3, 5, 6, 7], 11)
    gmsh.model.setPhysicalName(1, 11, "Wall")

    # Transfinite mesh settings
    gmsh.model.mesh.setTransfiniteSurface(1, cornerTags=[8, 5, 4, 1])
    # Vertical curves
    gmsh.model.mesh.setTransfiniteCurve(8, vertical_nodes, coef=1)
    gmsh.model.mesh.setTransfiniteCurve(4, vertical_nodes, coef=1)

    # Horizontal: 3 segments -> total nodes 257 across chain
    h1, h2, h3 = _compute_horizontal_counts_8p(points, target_total_nodes=horizontal_target_nodes)
    # Pairs: (1,7), (2,6), (3,5)
    gmsh.model.mesh.setTransfiniteCurve(1, h1, coef=1)
    gmsh.model.mesh.setTransfiniteCurve(7, h1, coef=1)

    gmsh.model.mesh.setTransfiniteCurve(2, h2, coef=1)
    gmsh.model.mesh.setTransfiniteCurve(6, h2, coef=1)

    gmsh.model.mesh.setTransfiniteCurve(3, h3, coef=1)
    gmsh.model.mesh.setTransfiniteCurve(5, h3, coef=1)    
    
    # Use quads
    gmsh.model.mesh.setRecombine(2, 1)

    # Generate mesh
    gmsh.model.mesh.generate(2)
    _prepare_output(mesh_output_path, mesh_format)
    gmsh.write(mesh_output_path)

    if show_gui:
        gmsh.fltk.run()

    gmsh.finalize()
    
    
def generate_mesh_from_points_10pnt(
    points,
    mesh_output_path,
    show_gui: bool = False,
    mesh_format=None,
    vertical_nodes: int = DEFAULT_VERTICAL_NODES,
    horizontal_target_nodes: int = DEFAULT_HORIZONTAL_TARGET_NODES,
):
    gmsh.initialize()
    gmsh.model.add("10pnt_ramp_python")

    # Generate Gmsh point definitions from (x, y)
    gmsh_code_lines = generate_gmsh_point_code(points)

    for line in gmsh_code_lines:
        exec(line)  # Add points dynamically

    # Add lines manually — assumes 10 points
    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 5, 4)
    gmsh.model.geo.addLine(5, 6, 5)
    gmsh.model.geo.addLine(6, 7, 6)
    gmsh.model.geo.addLine(7, 8, 7)
    gmsh.model.geo.addLine(8, 9, 8)
    gmsh.model.geo.addLine(9, 10, 9)
    gmsh.model.geo.addLine(10, 1, 10)

    gmsh.model.geo.addCurveLoop([10, 1, 2, 3, 4, 5, 6, 7, 8, 9], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)
    gmsh.model.geo.synchronize()

    # Physical groups
    gmsh.model.addPhysicalGroup(1, [10], 11)
    gmsh.model.setPhysicalName(1, 11, "Inlet")

    gmsh.model.addPhysicalGroup(1, [5], 12)
    gmsh.model.setPhysicalName(1, 12, "Outlet")

    gmsh.model.addPhysicalGroup(1, [1, 2, 3, 4, 6, 7, 8, 9], 13)
    gmsh.model.setPhysicalName(1, 13, "Wall")

    # Transfinite mesh settings
    gmsh.model.mesh.setTransfiniteSurface(1, cornerTags=[10, 6, 5, 1])
    # Vertical curves
    gmsh.model.mesh.setTransfiniteCurve(10, vertical_nodes, coef=1)
    gmsh.model.mesh.setTransfiniteCurve(5, vertical_nodes, coef=1)

    # Horizontal: 4 segments -> total nodes 257 across chain
    k1, k2, k3, k4 = _compute_horizontal_counts_10p(points, target_total_nodes=horizontal_target_nodes)
    # Pairs: (1,9), (2,8), (3,7), (4,6)
    gmsh.model.mesh.setTransfiniteCurve(1, k1, coef=1)
    gmsh.model.mesh.setTransfiniteCurve(9, k1, coef=1)

    gmsh.model.mesh.setTransfiniteCurve(2, k2, coef=1)
    gmsh.model.mesh.setTransfiniteCurve(8, k2, coef=1)

    gmsh.model.mesh.setTransfiniteCurve(3, k3, coef=1)
    gmsh.model.mesh.setTransfiniteCurve(7, k3, coef=1)

    gmsh.model.mesh.setTransfiniteCurve(4, k4, coef=1)
    gmsh.model.mesh.setTransfiniteCurve(6, k4, coef=1)
    
    # Use quads
    gmsh.model.mesh.setRecombine(2, 1)

    # Generate mesh
    gmsh.model.mesh.generate(2)
    _prepare_output(mesh_output_path, mesh_format)
    gmsh.write(mesh_output_path)

    if show_gui:
        gmsh.fltk.run()

    gmsh.finalize()