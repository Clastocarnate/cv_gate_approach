import cv2
import numpy as np

# Load the enhanced image
image = cv2.imread('enhanced_image.png')

# Apply Gaussian Blur to smoothen the image
# You can adjust the kernel size and sigma values to fine-tune the smoothing
blurred_image = cv2.GaussianBlur(image, (11, 11), 0)

# Optionally, apply a median filter to reduce noise while keeping edges sharp
median_filtered = cv2.medianBlur(blurred_image, 5)

# Save or display the result
cv2.imwrite('smoothed_image.png', median_filtered)
cv2.imshow('Original', image)
cv2.imshow('Smoothed Image', median_filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()