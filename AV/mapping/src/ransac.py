import numpy as np

class RANSACPlaneSegmentation:
    def __init__(self, num_iterations=1000, distance_threshold=0.1, min_points_in_plane=50):
        self.num_iterations = num_iterations
        self.distance_threshold = distance_threshold
        self.min_points_in_plane = min_points_in_plane

    def fit_plane(self, points):
        # Fit a plane using the least squares method
        A = np.c_[points[:, 0], points[:, 1], np.ones(points.shape[0])]
        b = points[:, 2]
        plane_params = np.linalg.lstsq(A, b, rcond=None)[0]
        return plane_params

    def get_inliers(self, points, plane_params):
        # Calculate the distance from each point to the plane
        distances = np.abs(np.dot(points[:, :3], plane_params[:3]) + plane_params[3]) / np.linalg.norm(plane_params[:3])

        # Find the indices of inliers
        inlier_indices = np.where(distances < self.distance_threshold)[0]
        inliers = points[inlier_indices]
        return inliers

    def segment_plane(self, points):
        best_plane = None
        best_inliers = []

        for _ in range(self.num_iterations):
            # Randomly sample 3 points
            sample_indices = np.random.choice(len(points), 3, replace=False)
            sample_points = points[sample_indices]

            # Fit a plane using the sampled points
            plane = self.fit_plane(sample_points)

            # Find the inliers (points that are close enough to the plane)
            inliers = self.get_inliers(points, plane)

            # Check if this plane is the best so far
            if len(inliers) > len(best_inliers):
                best_plane = plane
                best_inliers = inliers

            # If enough inliers found, break the loop
            if len(best_inliers) >= self.min_points_in_plane:
                break

        return best_plane, best_inliers

# Example usage:
# Assuming `points` is a numpy array of shape (N, 4) where N is the number of points and each point has [x, y, z, intensity]
# ransac = RANSACPlaneSegmentation()
# plane, inliers = ransac.segment_plane(points)

