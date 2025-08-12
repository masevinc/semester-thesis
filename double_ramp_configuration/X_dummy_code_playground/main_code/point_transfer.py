def generate_gmsh_point_code(sorted_points, z=0.0, lc=1.0, start_id=1):
    """
    Generates gmsh.model.geo.addPoint(...) code lines from a sorted list of (x, y) points.

    Parameters:
        sorted_points (list of tuples): List of (x, y) tuples, already sorted.
        z (float): Z-coordinate (default 0.0)
        lc (float): Characteristic length (default 1.0)
        start_id (int): Starting index for point IDs (default 1)

    Returns:
        list of str: Gmsh code lines
    """
    lines = []
    for i, (x, y) in enumerate(sorted_points):
        point_id = start_id + i
        line = f"gmsh.model.geo.addPoint({x}, {y}, {z}, {lc}, {point_id})"
        lines.append(line)
    return lines