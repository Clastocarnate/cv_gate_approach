import cv2
import numpy as np
import os

# Function to enhance an image
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
    
    # Step 4: Enhance red color channel
    contrast_enhanced[:, :, 2] = cv2.add(contrast_enhanced[:, :, 2], 50)  # Boost red channel
    
    # Step 5: Fusion using a weighted blend of white-balanced and contrast-enhanced images
    fused_image = cv2.addWeighted(white_balanced, 0.5, contrast_enhanced, 0.5, 0)
    
    # Step 6: Sharpening using Unsharp Masking
    blurred = cv2.GaussianBlur(fused_image, (0, 0), sigmaX=3, sigmaY=3)
    sharpened = cv2.addWeighted(fused_image, 1.5, blurred, -0.5, 0)
    
    return sharpened

# Function to detect corners using Shi-Tomasi method
def detect_corners(image, max_corners=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=max_corners, qualityLevel=0.01, minDistance=10)
    corners = np.int0(corners)
    return np.array([corner.ravel() for corner in corners])  # Flatten the corners

# Function to apply Canny edge detection
def apply_canny_edge(image, threshold1=40, threshold2=50):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1, threshold2)
    return edges

# Function to apply contour detection on the Canny edge output
def apply_contour_detection(image, edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contoured_image = image.copy()
    
    # Draw the center point of the normal bounding box
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        center = (x + w // 2, y + h // 2)
        area = w * h
        if area > 70000:
            cv2.circle(contoured_image, center, 5, (0, 0, 255), -1)  # Draw center point in red
    
    return contoured_image

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
    
    # Apply Canny edge detection
    edges = apply_canny_edge(enhanced_image)

    # Apply contour detection
    contoured_image = apply_contour_detection(enhanced_image, edges)

    # Display the result
    cv2.imshow("Contours and Centers on Enhanced Image", contoured_image)
    
    # Press 'q' to quit or any other key to move to the next image
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break

# Release resources
cv2.destroyAllWindows()
