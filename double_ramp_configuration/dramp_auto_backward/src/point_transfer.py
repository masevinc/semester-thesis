"""

Step 2.1.1

point_transfer.py

"""

def generate_gmsh_point_code(sorted_points, z=0.0, lc=1.0, start_id=1):
    """
    Generates Gmsh geometry script lines for adding points.

    Args:
        sorted_points (list of tuple): List of (x, y) coordinates for points.
        z (float): Z-coordinate for all points (default: 0.0).
        lc (float): Characteristic length for mesh sizing (default: 1.0).
        start_id (int): Starting index for Gmsh point IDs (default: 1).

    Returns:
        list of str: List of Gmsh script lines for adding points.
    """
    lines = []
    for i, (x, y) in enumerate(sorted_points):
        point_id = start_id + i
        line = f"gmsh.model.geo.addPoint({x}, {y}, {z}, {lc}, {point_id})"
        lines.append(line)
    return lines