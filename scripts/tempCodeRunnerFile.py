import cv2
import os
import numpy as np

# Define the image enhancement function (from the previous code)
def enhance_image(image):
    # Step 1: White Balancing (Gray-World Assumption)
    b, g, r = cv2.split(image)
    b_avg, g_avg, r_avg = np.mean(b), np.mean(g), np.mean(r)
    avg_gray = (b_avg + g_avg + r_avg) / 3
    b = np.clip(b * (avg_gray / b_avg), 0, 255).astype(np.uint8)
    g = np.clip(g * (avg_gray / g_avg), 0, 255).astype(np.uint8)
    r = np.clip(r * (avg_gray / r_avg), 0, 255).astype(np.uint8)
    white_balanced = cv2.merge([b, g, r])
    
    # Step 2: Apply Bilateral Filtering to reduce noise while keeping edges
    filtered = cv2.bilateralFilter(white_balanced, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Step 3: Convert to LAB color space and apply CLAHE on the L-channel
    lab = cv2.cvtColor(filtered, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    contrast_enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # Step 4: Fusion using a weighted blend of white-balanced and contrast-enhanced images
    fused_image = cv2.addWeighted(white_balanced, 0.5, contrast_enhanced, 0.5, 0)
    
    # Step 5: Sharpening using Unsharp Masking
    blurred = cv2.GaussianBlur(fused_image, (0, 0), sigmaX=3, sigmaY=3)
    sharpened = cv2.addWeighted(fused_image, 1.5, blurred, -0.5, 0)
    
    return sharpened

# Function to perform corner detection
def detect_corners(image, max_corners=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=max_corners, qualityLevel=0.01, minDistance=10)
    corners = np.int0(corners)

    # Draw detected corners
    for i in corners:
        x, y = i.ravel()
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Green circles for corners
    return image

# Folder containing images
folder_path = "data 5"
image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('.png', '.jpg', '.jpeg'))]
image_files.sort()  # Optional: Sort files by name

# Check if there are images in the folder
if not image_files:
    print("No images found in the specified folder.")
else:
    print(f"Found {len(image_files)} images.")

# Process each image
for image_path in image_files:
    # Load the original image
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Failed to load image: {image_path}")
        continue

    # Enhance the image
    enhanced_image = enhance_image(original_image.copy())
    
    # Detect corners in both original and enhanced images
    original_with_corners = detect_corners(original_image.copy())
    enhanced_with_corners = detect_corners(enhanced_image.copy())
    
    # Display the results side by side
    combined = np.hstack((original_with_corners, enhanced_with_corners))
    cv2.imshow("Corners in Original and Enhanced Images", combined)
    
    # Press 'q' to quit or any other key to move to the next image
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break

# Release resources
cv2.destroyAllWindows()