import gmsh
import math

def create_double_ramp_points(params):
    """
    Creates points for a double ramp geometry using input parameters.
    Only defines points; does not create lines or surfaces.

    Parameters (dict):
        h_top: float - Top height of the domain
        flat1: float - Horizontal flat segment before first ramp
        theta1_deg: float - First ramp angle in degrees
        ramp1_len: float - Horizontal length of first ramp
        flat2: float - Horizontal segment between ramps
        theta2_deg: float - Second ramp angle in degrees
        ramp2_len: float - Horizontal length of second ramp
        domain_len: float - Total x-extent of the domain
        mesh_size: float - Characteristic mesh length at each point
        z: float - Z-coordinate for 2D geometry (default 0.0)
        start_tag: int - Starting ID for point tags (default 1)

    Returns:
        List of point tags created
    """
    # Extract parameters
    h_top = params["h_top"]
    flat1 = params["flat1"]
    theta1 = math.radians(params["theta1_deg"])
    ramp1_len = params["ramp1_len"]
    flat2 = params["flat2"]
    theta2 = math.radians(params["theta2_deg"])
    ramp2_len = params["ramp2_len"]
    domain_len = params["domain_len"]
    mesh_size = params["mesh_size"]
    z = params.get("z", 0.0)
    start_tag = params.get("start_tag", 1)

    # Compute coordinates
    points = []
    x, y = 0.0, h_top

    points.append((x, y))  # Point 1: start
    x += flat1
    points.append((x, y))  # Point 2: after flat1
    x += ramp1_len
    y -= ramp1_len * math.tan(theta1)
    points.append((x, y))  # Point 3: end of first ramp
    x += flat2
    points.append((x, y))  # Point 4: after flat2
    x += ramp2_len
    y -= ramp2_len * math.tan(theta2)
    points.append((x, y))  # Point 5: end of second ramp
    x = domain_len
    points.append((x, y))  # Point 6: right edge at ramp base
    points.append((x, 0.0))      # Point 7: bottom right
    points.append((0.0, 0.0))    # Point 8: bottom left

    # Add points to gmsh and return their tags
    tags = []
    for i, (px, py) in enumerate(points, start=start_tag):
        # print(f"Point {i}: x = {px:.3f}, y = {py:.3f}")  # Output for user
        gmsh.model.geo.addPoint(px, py, z, mesh_size, i)
        tags.append(i)

    return tags, points