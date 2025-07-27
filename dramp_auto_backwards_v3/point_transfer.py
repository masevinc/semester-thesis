def generate_gmsh_point_code(sorted_points, z=0.0, lc=1.0, start_id=1):
    """
    Generates gmsh.model.geo.addPoint(...) code lines from a sorted list

    Parameters:
        sorted_points (list of tuples)
        z (float): Z-coordinate
        lc (float): Characteristic length
        start_id (int): index
    """
    lines = []
    for i, (x, y) in enumerate(sorted_points):
        point_id = start_id + i
        line = f"gmsh.model.geo.addPoint({x}, {y}, {z}, {lc}, {point_id})"
        lines.append(line)
    return lines