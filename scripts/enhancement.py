import cv2
import numpy as np

# Load the image
image = cv2.imread('frame_007243.png')

# Step 1: Color Balance (Removing excessive blue/green)
# Convert to YUV
image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
# Equalize the histogram of the Y channel
image_yuv[:,:,0] = cv2.equalizeHist(image_yuv[:,:,0])
# Convert back to BGR
image_rgb = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)

# Step 2: Contrast Enhancement using CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(9, 9))
lab = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2Lab)  # Convert to LAB color space
lab[:,:,0] = clahe.apply(lab[:,:,0])  # Apply CLAHE to the L-channel
image_enhanced = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)

# Step 3: Optional Sharpening
kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
image_sharpened = cv2.filter2D(image_enhanced, -1, kernel)

# Save or display the result
cv2.imwrite('enhanced_image.png', image_sharpened)
cv2.imshow('Original', image)
cv2.imshow('Enhanced Image', image_sharpened)
cv2.waitKey(0)
cv2.destroyAllWindows()