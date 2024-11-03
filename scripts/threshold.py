import cv2
import numpy as np

def on_trackbar_change(_):
    # Retrieve the trackbar positions for min and max YUV values
    y_min = cv2.getTrackbarPos('Y Min', 'YUV Thresholds')
    u_min = cv2.getTrackbarPos('U Min', 'YUV Thresholds')
    v_min = cv2.getTrackbarPos('V Min', 'YUV Thresholds')
    y_max = cv2.getTrackbarPos('Y Max', 'YUV Thresholds')
    u_max = cv2.getTrackbarPos('U Max', 'YUV Thresholds')
    v_max = cv2.getTrackbarPos('V Max', 'YUV Thresholds')
    
    # Create the min and max arrays
    min_array = np.array([y_min, u_min, v_min], np.uint8)
    max_array = np.array([y_max, u_max, v_max], np.uint8)
    
    # Threshold the YUV image to get only colors in the specified range
    mask = cv2.inRange(yuv, min_array, max_array)
    result = cv2.bitwise_and(image, image, mask=mask)
    
    # Display the result
    cv2.imshow('YUV Thresholds', result)

# Load the image
image = cv2.imread('frame_007243.png')
# Convert to YUV
yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

# Create a window
cv2.namedWindow('YUV Thresholds')

# Create trackbars for Y, U, and V channels
cv2.createTrackbar('Y Min', 'YUV Thresholds', 0, 255, on_trackbar_change)
cv2.createTrackbar('U Min', 'YUV Thresholds', 0, 255, on_trackbar_change)
cv2.createTrackbar('V Min', 'YUV Thresholds', 0, 255, on_trackbar_change)
cv2.createTrackbar('Y Max', 'YUV Thresholds', 255, 255, on_trackbar_change)
cv2.createTrackbar('U Max', 'YUV Thresholds', 255, 255, on_trackbar_change)
cv2.createTrackbar('V Max', 'YUV Thresholds', 255, 255, on_trackbar_change)

# Initialize display
on_trackbar_change(0)  # Initial call to update the display

# Wait until the user exits the window
cv2.waitKey(0)
cv2.destroyAllWindows()