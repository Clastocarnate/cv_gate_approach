import cv2
import os
import numpy as np

# Define the image enhancement function based on multi-scale fusion method
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
    
    # Step 4: Fusion using Laplacian and Gaussian pyramids (simple blend)
    # Here, we blend the original white-balanced and contrast-enhanced images
    fused_image = cv2.addWeighted(white_balanced, 0.5, contrast_enhanced, 0.5, 0)
    
    return fused_image

# Folder containing images
folder_path = "data 5"
image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('.png', '.jpg', '.jpeg'))]
image_files.sort()  # Optional: Sort files by name

# Check if there are images in the folder
if not image_files:
    print("No images found in the specified folder.")
else:
    print(f"Found {len(image_files)} images.")

# Process and display each image
for image_path in image_files:
    # Load the image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Failed to load image: {image_path}")
        continue

    # Enhance the image
    enhanced_frame = enhance_image(frame)
    
    # Display the original and enhanced images side by side
    combined = np.hstack((frame, enhanced_frame))
    cv2.imshow("Original and Enhanced Image", combined)
    
    # Press 'q' to quit or any other key to see the next image
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break

# Release resources
cv2.destroyAllWindows()