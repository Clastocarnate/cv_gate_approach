import cv2 as cv
import numpy as np
from ultralytics import YOLO
import pickle

# Load the camera matrix and distortion coefficients
with open("cameraMatrix.pkl", "rb") as f:
    cameraMatrix = pickle.load(f)

with open("dist.pkl", "rb") as f:
    distCoeffs = pickle.load(f)

# Define the real-world 3D coordinates of the gate corners (60cm x 40cm)
objectPoints = np.array([
    [-0.3, -0.2, 0],   # Bottom-left corner (3D space in meters, relative to the center)
    [ 0.3, -0.2, 0],   # Bottom-right corner
    [ 0.3,  0.2, 0],   # Top-right corner
    [-0.3,  0.2, 0]    # Top-left corner
], dtype=np.float32)

# Load YOLOv8 model for object detection
model = YOLO('best-3.pt')

# Start video capture
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 Inference on the frame
    results = model(frame)
    bboxes = results[0].boxes.xyxy.cpu().numpy()  # Extract bounding boxes
    if len(bboxes) > 0:
        # Process the first bounding box
        x1, y1, x2, y2 = map(int, bboxes[0])

        # Calculate the center of the bounding box
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        # Find the corners of the bounding box in the image (2D image points)
        imagePoints = np.array([
            [x1, y1],  # Bottom-left corner
            [x2, y1],  # Bottom-right corner
            [x2, y2],  # Top-right corner
            [x1, y2]   # Top-left corner
        ], dtype=np.float32)

        # Solve for rotation and translation vectors (pose estimation)
        success, rvec, tvec = cv.solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs)

        if success:
            # Project the 3D axis points to the 2D image plane for visualization
            axis = np.float32([[0.3, 0, 0], [0, 0.3, 0], [0, 0, -0.3]]).reshape(-1, 3)
            imgpts, jac = cv.projectPoints(axis, rvec, tvec, cameraMatrix, distCoeffs)

            # Convert imgpts to integer tuples for drawing lines
            imgpts = np.int32(imgpts).reshape(-1, 2)

            # Draw the 3D axes on the frame at the center point
            frame = cv.line(frame, (center_x, center_y), tuple(imgpts[0]), (0, 0, 255), 5)  # X-axis (red)
            frame = cv.line(frame, (center_x, center_y), tuple(imgpts[1]), (0, 255, 0), 5)  # Y-axis (green)
            frame = cv.line(frame, (center_x, center_y), tuple(imgpts[2]), (255, 0, 0), 5)  # Z-axis (blue)

            # Display the translation vector (distance from the camera to the center of the gate)
            distance = np.linalg.norm(tvec)
            cv.putText(frame, f"Distance: {distance:.2f}m", (center_x, center_y - 10), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)

    # Display the frame with 3D axes
    cv.imshow('Frame', frame)

    # Exit on pressing 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release capture
cap.release()
cv.destroyAllWindows()