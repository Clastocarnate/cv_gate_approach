import cv2
import numpy as np

# Load the image
frame = cv2.imread("smoothed_image.png")
framea = frame[:, 0:500]
frame2 = cv2.bitwise_not(framea)

gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

# Initial threshold values
threshold_min = 180
threshold_max = 255

# Minimum line length for filtering
min_line_length = 50  # Adjust based on your specific needs

def group_points(points, distance_threshold):
    """ Group points that are within the distance_threshold of each other. """
    if not points:
        return []

    # Sort points based on x coordinates (helps with clustering)
    points = sorted(points, key=lambda x: x[0])
    clusters = []
    cluster = [points[0]]

    for point in points[1:]:
        if np.linalg.norm(np.array(point) - np.array(cluster[-1])) < distance_threshold:
            cluster.append(point)
        else:
            if len(cluster) > 1:
                clusters.append(cluster)
            cluster = [point]
    if len(cluster) > 1:
        clusters.append(cluster)

    # Calculate the centroid of each cluster
    centroids = []
    for cluster in clusters:
        x_coords = [p[0] for p in cluster]
        y_coords = [p[1] for p in cluster]
        centroid = (sum(x_coords) // len(cluster), sum(y_coords) // len(cluster))
        centroids.append(centroid)
    return centroids

def apply_threshold(gray, threshold_min, threshold_max, min_line_length):
    _, threshold = cv2.threshold(gray, threshold_min, threshold_max, cv2.THRESH_BINARY)
    threshold_rgb = cv2.cvtColor(threshold, cv2.COLOR_GRAY2RGB)

    # Find and approximate contours
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = []

    # Draw contours and check for quadrilateral
    for cnt in contours:
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Collect points and draw lines if they meet the line length criterion
        for i in range(len(approx)):
            pt1 = tuple(approx[i][0])
            pt2 = tuple(approx[(i + 1) % len(approx)][0])
            line_length = np.linalg.norm(np.array(pt1) - np.array(pt2))
            if line_length >= min_line_length:
                points.append(pt1)
                points.append(pt2)

    # Group and average nearby points
    centroids = group_points(points, 35)
    for centroid in centroids:
        cv2.circle(threshold_rgb, centroid, 5, (0, 255, 0), -1)
        cv2.putText(threshold_rgb, str(centroid), centroid, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    return threshold_rgb

# Process the image
processed_image = apply_threshold(gray, threshold_min, threshold_max, min_line_length)

# Display the processed image
cv2.imshow("Processed Image", processed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()