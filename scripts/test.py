import cv2
import numpy as np
import os

# Specify the folder containing the images
folder_path = 'images'  # Replace with your folder path

# Get a list of all image files in the folder
image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path)
               if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Define HSV threshold ranges
lower_hsv = np.array([0, 225, 100])    # Hue: 0, Saturation: 225, Value: 100
upper_hsv = np.array([180, 255, 150])  # Hue: 180, Saturation: 255, Value: 150

# Create a named window (optional)
cv2.namedWindow('Processed Image', cv2.WINDOW_NORMAL)

# Process each image
for image_file in image_files:
    # Read the image
    image = cv2.imread(image_file)
    if image is None:
        print(f"Error reading image {image_file}")
        continue

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Apply HSV thresholding to create a binary mask
    mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Process each contour
    for cnt in contours:
        # Calculate the bounding rectangle for each contour
        x, y, w, h = cv2.boundingRect(cnt)
        # Calculate the area of the bounding rectangle
        area = w * h
        # Check if the area is greater than 5000 pixels
        if area > 20000:
            # Draw the bounding rectangle on the original image
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green color with thickness 2

    # Display the processed image using OpenCV
    cv2.imshow('Processed Image', image)

    # Wait for 500 milliseconds; exit if 'q' is pressed
    if cv2.waitKey(500) & 0xFF == ord('q'):
        break

# Clean up and close windows
cv2.destroyAllWindows()