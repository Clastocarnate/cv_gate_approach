import cv2
import numpy as np
import os

class ImageProcessor:
    def __init__(self, folder_path, output_video='prediction.mp4', fps=5):
        self.folder_path = folder_path
        self.image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('.png', '.jpg', '.jpeg'))]
        self.current_index = 0
        self.threshold_min = 180
        self.threshold_max = 255
        self.min_line_length = 100
        self.output_video = output_video
        self.fps = fps
        self.frame_width, self.frame_height = cv2.imread(self.image_files[0]).shape[1], cv2.imread(self.image_files[0]).shape[0]
        self.video_writer = cv2.VideoWriter(self.output_video, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.frame_width, self.frame_height))
        
    def load_image(self):
        # Load current image
        image_path = self.image_files[self.current_index]
        image = cv2.imread(image_path)
        return image
    
    def next_image(self):
        # Move to the next image, cycling back to the first if needed
        self.current_index = (self.current_index + 1) % len(self.image_files)
    
    def enhance_image(self, image):
        # Step 1: Color Balance
        image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        image_yuv[:,:,0] = cv2.equalizeHist(image_yuv[:,:,0])
        image_rgb = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)
        
        # Step 2: Contrast Enhancement using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(9, 9))
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2Lab)
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        image_enhanced = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
        
        # Step 3: Sharpening
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        image_sharpened = cv2.filter2D(image_enhanced, -1, kernel)
        return image_sharpened
    
    def smooth_image(self, image):
        # Apply Gaussian Blur
        blurred_image = cv2.GaussianBlur(image, (11, 11), 0)
        
        # Apply Median Filter
        median_filtered = cv2.medianBlur(blurred_image, 5)
        return median_filtered

    def group_points(self, points, distance_threshold=35):
        # Group points that are close to each other
        if not points:
            return []
        points = sorted(points, key=lambda x: x[0])
        clusters, cluster = [], [points[0]]
        
        for point in points[1:]:
            if np.linalg.norm(np.array(point) - np.array(cluster[-1])) < distance_threshold:
                cluster.append(point)
            else:
                if len(cluster) > 1:
                    clusters.append(cluster)
                cluster = [point]
        if len(cluster) > 1:
            clusters.append(cluster)
        
        centroids = [(int(sum([p[0] for p in cluster]) / len(cluster)), 
                      int(sum([p[1] for p in cluster]) / len(cluster))) 
                     for cluster in clusters]
        return centroids

    def apply_threshold(self, gray_image, original_image):
        _, threshold = cv2.threshold(gray_image, self.threshold_min, self.threshold_max, cv2.THRESH_BINARY)
        
        # Find contours and process them
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        points = []
        
        for cnt in contours:
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            for i in range(len(approx)):
                pt1, pt2 = tuple(approx[i][0]), tuple(approx[(i + 1) % len(approx)][0])
                if np.linalg.norm(np.array(pt1) - np.array(pt2)) >= self.min_line_length:
                    points.append(pt1)
                    points.append(pt2)

        centroids = self.group_points(points)
        for centroid in centroids:
            cv2.circle(original_image, centroid, 5, (0, 255, 0), -1)
            cv2.putText(original_image, str(centroid), centroid, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        return original_image

    def process_and_display(self):
        # Load image, enhance, smooth, apply thresholding, and display
        original_image = self.load_image()
        enhanced_image = self.enhance_image(original_image)
        smoothed_image = self.smooth_image(enhanced_image)
        
        inverted_image = cv2.bitwise_not(smoothed_image)
        gray = cv2.cvtColor(inverted_image, cv2.COLOR_BGR2GRAY)
        
        processed_image = self.apply_threshold(gray, original_image)
        
        # Write frame to video file
        self.video_writer.write(processed_image)
        
        # Display processed image
        cv2.imshow("Processed Image", processed_image)
    
    def adjust_threshold_min(self, val):
        self.threshold_min = val
        self.process_and_display()
    
    def adjust_threshold_max(self, val):
        self.threshold_max = val
        self.process_and_display()

    def create_trackbars(self):
        cv2.namedWindow("Processed Image")
        cv2.createTrackbar("Threshold Min", "Processed Image", self.threshold_min, 255, self.adjust_threshold_min)
        cv2.createTrackbar("Threshold Max", "Processed Image", self.threshold_max, 255, self.adjust_threshold_max)

    def run(self):
        self.create_trackbars()
        while True:
            self.process_and_display()
            key = cv2.waitKey(0)
            
            if key == ord('n'):
                self.next_image()
            elif key == ord('q'):
                break
        cv2.destroyAllWindows()
        self.video_writer.release()

# Example usage
folder_path = '/Users/madhuupadhyay/Downloads/data 5'
processor = ImageProcessor(folder_path)
processor.run()