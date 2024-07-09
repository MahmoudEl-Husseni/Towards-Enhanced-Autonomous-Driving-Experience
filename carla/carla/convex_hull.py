# Function to compute the cross product of vectors (p1-p0) and (p2-p0)
def cross_product(p0, p1, p2):
    return (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])

# Function to compute the convex hull using Graham Scan algorithm
def ConvexHull(points):
    # Sort points lexicographically (first by x, then by y)
    points = points.tolist()
    points = sorted(points)
    
    # Function to check if turning left
    def is_turning_left(p0, p1, p2):
        return cross_product(p0, p1, p2) > 0
    
    # Initialize hull with first two points
    hull = [points[0], points[1]]
    
    # Build the upper hull
    for i in range(2, len(points)):
        hull.append(points[i])
        while len(hull) > 2 and not is_turning_left(hull[-3], hull[-2], hull[-1]):
            hull.pop(-2)
    
    # Build the lower hull
    lower_hull = [points[-1]]
    for i in range(len(points) - 2, -1, -1):
        lower_hull.append(points[i])
        while len(lower_hull) > 2 and not is_turning_left(lower_hull[-3], lower_hull[-2], lower_hull[-1]):
            lower_hull.pop(-2)
    
    # Combine upper and lower hulls
    return hull + lower_hull[1:-1]