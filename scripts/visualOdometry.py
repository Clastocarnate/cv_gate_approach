import os
import numpy as np
import cv2
from tqdm import tqdm

class VisualOdometry():
    def __init__(self):
        # Camera calibration parameters for the webcam
        self.K = np.array([[1003.1076720832533, 0.0, 325.5842274588375],
                           [0.0, 1004.8079121262164, 246.67564927792367],
                           [0.0, 0.0, 1.0]])
        self.dist_coef = np.array([0.1886629014531147, 0.0421057002310688, 0.011153911980654914, 0.012946956962024124, 0.0])
        self.orb = cv2.ORB_create(3000)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

    def _form_transf(self, R, t):
        """
        Makes a transformation matrix from the given rotation matrix and translation vector
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def get_matches(self, img1, img2):
        """
        This function detects and computes keypoints and descriptors from two images using the class orb object
        """
        # Find the keypoints and descriptors with ORB
        kp1, des1 = self.orb.detectAndCompute(img1, None)
        kp2, des2 = self.orb.detectAndCompute(img2, None)

        if des1 is None or des2 is None:
            return np.array([]), np.array([])

        # Find matches
        matches = self.flann.knnMatch(des1, des2, k=2)

        # Find the matches that do not have a too high distance
        good = []
        try:
            for m, n in matches:
                if m.distance < 0.8 * n.distance:
                    good.append(m)
        except ValueError:
            pass

        # Get the image points from the good matches
        q1 = np.float32([kp1[m.queryIdx].pt for m in good])
        q2 = np.float32([kp2[m.trainIdx].pt for m in good])
        return q1, q2

    def get_pose(self, q1, q2):
        """
        Calculates the transformation matrix
        """
        # Essential matrix
        E, _ = cv2.findEssentialMat(q1, q2, self.K, threshold=1)

        # Decompose the Essential matrix into R and t
        R, t = self.decomp_essential_mat(E, q1, q2)

        # Get transformation matrix
        transformation_matrix = self._form_transf(R, np.squeeze(t))
        return transformation_matrix

    def decomp_essential_mat(self, E, q1, q2):
        """
        Decompose the Essential matrix
        """
        def sum_z_cal_relative_scale(R, t):
            P1 = np.dot(self.K, np.hstack((np.eye(3), np.zeros((3, 1)))))
            P2 = np.dot(self.K, np.hstack((R, t.reshape(-1, 1))))

            hom_Q1 = cv2.triangulatePoints(P1, P2, q1.T, q2.T)
            hom_Q2 = np.dot(self._form_transf(R, t), hom_Q1)

            uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

            sum_of_pos_z_Q1 = np.sum(uhom_Q1[2, :] > 0)
            sum_of_pos_z_Q2 = np.sum(uhom_Q2[2, :] > 0)

            relative_scale = np.mean(np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=1) / 
                                     np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=1))
            return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale

        # Decompose the essential matrix
        R1, R2, t = cv2.decomposeEssentialMat(E)
        t = np.squeeze(t)

        # Make a list of the different possible pairs
        pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]

        # Check which solution there is the right one
        z_sums = []
        relative_scales = []
        for R, t in pairs:
            z_sum, scale = sum_z_cal_relative_scale(R, t)
            z_sums.append(z_sum)
            relative_scales.append(scale)

        # Select the pair that has the most points with positive z coordinate
        right_pair_idx = np.argmax(z_sums)
        right_pair = pairs[right_pair_idx]
        relative_scale = relative_scales[right_pair_idx]
        R1, t = right_pair
        t = t * relative_scale

        return [R1, t]

def main():
    vo = VisualOdometry()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        return

    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    cur_pose = np.eye(4)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        q1, q2 = vo.get_matches(prev_frame_gray, frame_gray)
        if len(q1) >= 5 and len(q2) >= 5:
            transf = vo.get_pose(q1, q2)
            cur_pose = np.matmul(cur_pose, np.linalg.inv(transf))
            print(f"Current Pose: {cur_pose}")

        prev_frame_gray = frame_gray

        cv2.imshow('Live Visual Odometry', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()