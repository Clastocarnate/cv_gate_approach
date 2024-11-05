import cv2

vid = "images/dehazed_footage.mp4"
cap = cv2.VideoCapture('images/dehazed_footage.mp4')

if cap.isOpened():
    print("Video Opened")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("No more frames to read or failed to read frame.")
        break

    # Convert to grayscale and apply median blur
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray_frame, 5)
    
    # Apply adaptive thresholding
    thresh1 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 20)
    
    # Find contours
    contours, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Loop through contours
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter out small contours
        if area > 1000 and area<100000:
            # Draw bounding box around contour
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Print area of the contour
            print(f"Contour Area: {area}")

    # Display the processed frame with bounding boxes
    cv2.imshow("Adaptive Threshold Video with Contours", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()