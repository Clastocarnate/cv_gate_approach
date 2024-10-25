import cv2
import numpy as np

def adjust_thresholds(val):
    # Redefine the thresholds from the trackbar values
    blur_ksize = cv2.getTrackbarPos('Kernel Size', 'Detected Gate') * 2 + 1  # to keep it odd
    lower_threshold = cv2.getTrackbarPos('Lower Threshold', 'Detected Gate')
    upper_threshold = cv2.getTrackbarPos('Upper Threshold', 'Detected Gate')

    # Reapply Gaussian Blur with new kernel size
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    # Reapply Canny with new thresholds
    edges = cv2.Canny(blurred, lower_threshold, upper_threshold)

    # Find contours and draw them
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_image = image.copy()
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            cv2.drawContours(detected_image, [approx], -1, (0, 255, 0), 3)

    # Show the results
    cv2.imshow('Detected Gate', detected_image)

# Load the image
image = cv2.imread('smoothed_image.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create a window
cv2.namedWindow('Detected Gate')

# Create trackbars
cv2.createTrackbar('Kernel Size', 'Detected Gate', 1, 10, adjust_thresholds)  # Kernel size from 1 to 20
cv2.createTrackbar('Lower Threshold', 'Detected Gate', 50, 255, adjust_thresholds)
cv2.createTrackbar('Upper Threshold', 'Detected Gate', 150, 255, adjust_thresholds)

# Initial call to function to display the initial image
adjust_thresholds(0)  # the argument doesn't matter here

# Wait until user exits
cv2.waitKey(0)
cv2.destroyAllWindows()