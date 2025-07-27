'''def sort_points(points):
    # Convert list of points to a set to ensure uniqueness
    unique_points = list(set(points))

    # Identify the first point: min x, max y
    first_point = min(unique_points, key=lambda p: (p[0], -p[1]))

    # Identify the last point: min x, min y
    last_point = min(unique_points, key=lambda p: (p[0], p[1]))

    # Identify the second-last point: max x, min y
    second_last_point = max(unique_points, key=lambda p: (p[0], -p[1]))

    # Exclude the identified points from the remaining points
    remaining_points = [p for p in unique_points if p not in {first_point, second_last_point, last_point}]

    # Sort the remaining points by increasing x-coordinate
    remaining_sorted = sorted(remaining_points, key=lambda p: p[0])

    # Combine the points in the desired order
    ordered_points = [first_point] + remaining_sorted + [second_last_point, last_point]

    return ordered_points'''

import math

def sort_points(points):
    # Remove duplicates
    unique_points = list(set(points))

    # Find the start point (min x, max y)
    start_point = min(unique_points, key=lambda p: (p[0], -p[1]))

    # Compute centroid for angle reference
    cx = sum(p[0] for p in unique_points) / len(unique_points)
    cy = sum(p[1] for p in unique_points) / len(unique_points)

    def angle_from_center(p):
        angle = math.atan2(p[1] - cy, p[0] - cx)
        return angle

    # Sort points clockwise around centroid
    sorted_points = sorted(unique_points, key=angle_from_center, reverse=True)

    # Rotate list to start from top-left point
    start_index = sorted_points.index(start_point)
    ordered_points = sorted_points[start_index:] + sorted_points[:start_index]

    return ordered_points
