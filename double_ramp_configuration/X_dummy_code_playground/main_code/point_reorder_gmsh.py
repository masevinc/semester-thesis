def sort_points(points):
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

    return ordered_points