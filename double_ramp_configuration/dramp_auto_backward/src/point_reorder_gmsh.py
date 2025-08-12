"""

Step 1.1.1

point_reorder_gmsh.py

"""

import math

def sort_points(points):
    """
    Sorts a list of 2D points in clockwise order around their centroid,
    starting from the top-left (minimum x, maximum y) point.

    Args:
        points (list of tuple): List of (x, y) coordinates.

    Returns:
        list of tuple: Ordered list of points, clockwise from top-left.
    """
    # Remove duplicate points
    unique_points = list(set(points))

    # Find the starting point (top-left: min x, max y)
    start_point = min(unique_points, key=lambda p: (p[0], -p[1]))

    # Compute centroid for angle calculation
    cx = sum(p[0] for p in unique_points) / len(unique_points)
    cy = sum(p[1] for p in unique_points) / len(unique_points)

    def angle_from_center(p):
        """Returns the angle from the centroid to point p."""
        return math.atan2(p[1] - cy, p[0] - cx)

    # Sort points in clockwise order around the centroid
    sorted_points = sorted(unique_points, key=angle_from_center, reverse=True)

    # Rotate list so it starts from the top-left point
    start_index = sorted_points.index(start_point)
    ordered_points = sorted_points[start_index:] + sorted_points[:start_index]

    return ordered_points
