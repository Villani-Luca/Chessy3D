import os
import cv2
import math
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
import pyautogui

import matplotlib.pyplot as plt
import matplotlib.cm as cms


class ChessboardDetector:
    original_image: cv2.typing.MatLike
    processed_image: cv2.typing.MatLike
    edges_image: cv2.typing.MatLike
    lines_image: cv2.typing.MatLike
    clustered_lines_image: cv2.typing.MatLike
    clustered_lines_filtered_image: cv2.typing.MatLike
    intersections_image: cv2.typing.MatLike

    lines_angles = None
    line_segments = None
    lines_labels = None

    filtered_line_segments = None
    filtered_line_labels = None

    def __init__(self, img):
        self.original_image = img

    def process(self):
        self.__apply_image_processing()
        self.__detect_edges()
        self.__detect_lines()
        self.__classify_lines()
        #self.__filter_lines_duplicates()
        self.__find_all_intersections2()

        #print(self.lines_angles)
        #print(self.line_segments)
        #print(self.lines_labels)

        #self.__debug_image_cv2(self.original_image)
        #self.__debug_image_cv2(self.processed_image)
        #self.__debug_image_cv2(self.edges_image)
        #self.__debug_image_cv2(self.lines_image)
        #self.__debug_image_cv2(self.clustered_lines_image)
        #self.__debug_image_cv2(self.clustered_lines_filtered_image)
        self.__debug_image_cv2(self.intersections_image)

        #debug_line_angles(angles_radiants, labels)
        #plot_vectors(filtered_segments_features, labels)

    def __apply_image_processing(self):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(
            gray,
            ksize=(15, 15),  # kernel size -> since the image is big this must be big
            sigmaX=3,  # horizontal blur intensity
            sigmaY=3,  # vertical blur intensity
        )

        self.processed_image = blurred

    def __detect_edges(self):
        self.edges_image = cv2.Canny(
            self.processed_image,
            threshold1=50,  # lower hysteresis threshold
            threshold2=100,  # upper hysteresis threshold,
            apertureSize=3,  # sobel kernel size
            L2gradient=True,  # false = l1 norm (faster), true = l2 norm (more accurate)
        )

    def __detect_lines(self):
        hough_segments = cv2.HoughLinesP(  # Probabilistic Hough Transform (PHT)
            self.edges_image,
            rho=1,  # 1 pixel resolution -> this is usually fine
            theta=math.pi
            / 180,  # 1 degree resolution (in radiants) -> this is usually fine
            threshold=100,  # lower = detect more lines (including noise), higher = keep only clear lines
            minLineLength=100,  # lower = detect short and fragmented lines, higher = discard short and fragmented lines
            maxLineGap=20,  # lower = close lines are considered separated, higher = close lines are considered the same (merged)
        )

        segments = np.array(hough_segments).squeeze()
        segments_vectors = self.__calculate_normalized_vectors_from_segments(segments)

        # these are between [-π, π]
        self.lines_angles = np.arctan2(segments_vectors[:, 1], segments_vectors[:, 0])

        img_height, img_width = self.original_image.shape[:2]
        self.line_segments = self.__extend_segments_size(
            segments, img_height, img_width
        )
        self.lines_image = self.__draw_segments_on_image(
            self.line_segments, self.original_image
        )

    def __detect_lines_alt(self):
        hough_lines = cv2.HoughLines(
            self.edges_image,
            rho=1,  # 1 pixel resolution -> this is usually fine
            theta=math.pi
            / 180,  # 1 degree resolution (in radiants) -> this is usually fine
            threshold=100,  # lower = detect more lines (including noise), higher = keep only clear lines
        )

        lines = np.array(hough_lines).squeeze()

        # these are between [0, π]
        self.lines_angles = lines[:, 1]

        self.line_segments = self.__create_segments_from_lines(lines)
        self.lines_image = self.__draw_segments_on_image(
            self.line_segments, self.original_image
        )

    def __classify_lines(self):
        doubled_line_angles = self.lines_angles * 2

        features = np.stack(
            (np.cos(doubled_line_angles), np.sin(doubled_line_angles)), axis=1
        )

        kmeans = KMeans(n_clusters=2)
        self.lines_labels = kmeans.fit(features).labels_
        self.clustered_lines_image = self.__draw_segments_on_image(
            self.line_segments, self.original_image, self.lines_labels
        )

    def __classify_lines_alt(self):
        doubled_line_angles = self.lines_angles * 2

        features = np.stack(
            (np.cos(doubled_line_angles), np.sin(doubled_line_angles)), axis=1
        )

        model = AgglomerativeClustering(
            n_clusters=2,
            metric="euclidean",
            linkage="ward",  # as alternative also "average" is fine
        )

        self.lines_labels = model.fit_predict(features)
        self.clustered_lines_image = self.__draw_segments_on_image(
            self.line_segments, self.original_image, self.lines_labels
        )

    def __filter_lines_duplicates(self):

        filtered_segments = []
        filtered_labels = []

        for label in self.lines_labels:
            segments = self.line_segments[self.lines_labels == label]

            if len(segments) == 0:
                continue

            dist_matrix = self.__calculate_segments_distance_matrix2(segments)
            dist_matrix = np.nan_to_num(dist_matrix, nan=1e10)

            dbscan = DBSCAN(eps=10, min_samples=1, metric="precomputed")
            clusters = dbscan.fit_predict(dist_matrix)

            for cluster_id in np.unique(clusters):
                if cluster_id == -1:
                    continue

                cluster_mask = clusters == cluster_id
                cluster_segments = segments[cluster_mask]
                cluster_dist_matrix = dist_matrix[cluster_mask][:, cluster_mask]
                medoid_idx = np.argmin(cluster_dist_matrix.sum(axis=1))
                filtered_segments.append(cluster_segments[medoid_idx])
                filtered_labels.append(label)

        self.filtered_line_segments = np.array(filtered_segments)
        self.filtered_line_labels = np.array(filtered_labels)

        self.clustered_lines_filtered_image = self.__draw_segments_on_image(
            self.filtered_line_segments,
            self.original_image,
            self.filtered_line_labels,
        )

    def __find_lines_intersections(self):
        segments_line_equation = self.__calculate_line_equation_from_segments(
            self.line_segments
        )

        N = len(self.line_segments)
        intersection_matrix = np.full((N, N, 2), np.nan)

        self.intersections_image = self.clustered_lines_image.copy()
        img_height, img_width = self.intersections_image.shape[:2]

        for i in range(N):
            for j in range(N):
                if self.lines_labels[i] == self.lines_labels[j]:
                    continue

                A1, B1, C1 = segments_line_equation[i]
                A2, B2, C2 = segments_line_equation[j]

                det = A1 * B2 - A2 * B1

                if abs(det) < 1e-6:
                    continue

                x = (B2 * C1 - B1 * C2) / det
                y = (A1 * C2 - A2 * C1) / det

                x = np.clip(round(x), 0, img_width - 1)
                y = np.clip(round(y), 0, img_height - 1)
                intersection_matrix[i, j] = [x, y]

        #print(intersection_matrix)
        
        for i in range(intersection_matrix.shape[0]):
            for j in range(intersection_matrix.shape[1]):
                if not np.isnan(intersection_matrix[i, j, 0]):
                    x, y = intersection_matrix[i, j]
                    x_pixel = int(x)
                    y_pixel = int(y)
                    cv2.circle(
                        self.intersections_image,
                        (x_pixel, y_pixel),
                        20,
                        (0, 255, 0),
                        -1,
                    )

    def  __find_all_intersections2(self):

        segments = self.line_segments
        labels = self.lines_labels

        #segments = self.filtered_line_segments
        #labels = self.filtered_line_labels

        self.intersections_image = self.clustered_lines_image.copy()
        img_height, img_width = self.intersections_image.shape[:2]
        
        N = len(segments)
        intersections = np.full((N, N, 2), np.nan)
        unique_points = set()
        
        # Precompute Hesse normal forms
        hesse_lines = [self.__segment_to_hesse_normal(seg) for seg in segments]
        
        for i in range(N):
            for j in range(i+1, N):

                if labels[i] == labels[j]:
                    continue

                pt = self.__intersect_hesse_lines(hesse_lines[i], hesse_lines[j], img_height, img_width)
                if pt is not None:
                    x, y = pt
                    intersections[i,j] = [x, y]
                    intersections[j,i] = [x, y]  # Symmetric
                    unique_points.add((x, y))
        
        for i in range(intersections.shape[0]):
            for j in range(intersections.shape[1]):
                if not np.isnan(intersections[i, j, 0]):
                    x, y = intersections[i, j]
                    x_pixel = int(x)
                    y_pixel = int(y)
                    cv2.circle(
                        self.intersections_image,
                        (x_pixel, y_pixel),
                        5,
                        (0, 255, 0),
                        -1,
                    )

    # ----------------------------------------------------------

    def __debug_image_cv2(self, img: cv2.typing.MatLike, max_height=800, max_width=800):

        window_name = "Debug"

        img_height, img_width = img.shape[:2]

        scale_width = max_width / img_width
        scale_height = max_height / img_height

        scale_factor = min(scale_width, scale_height)

        new_width = int(img_width * scale_factor)
        new_height = int(img_height * scale_factor)

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, new_width, new_height)
        cv2.imshow(window_name, img)

        screen_width, screen_height = pyautogui.size()

        pos_x = (screen_width - new_width) // 2
        pos_y = (screen_height - new_height) // 2

        cv2.moveWindow(window_name, pos_x, pos_y)

        cv2.waitKey(0)
        cv2.destroyWindow(window_name)

    def __calculate_normalized_vectors_from_segments(self, segments):
        x1 = segments[:, 0]
        y1 = segments[:, 1]
        x2 = segments[:, 2]
        y2 = segments[:, 3]

        dx = x2 - x1
        dy = y2 - y1

        norm = np.sqrt(dx**2 + dy**2)

        # this is used to avoid divisions by 0
        norm[norm == 0] = 1e-8

        normalized_dx = dx / norm
        normalized_dy = dy / norm

        return np.stack((normalized_dx, normalized_dy), axis=1)

    def __calculate_line_slope_intercept_from_segments(self, segments):
        x1 = segments[:, 0]
        y1 = segments[:, 1]
        x2 = segments[:, 2]
        y2 = segments[:, 3]

        dx = x2 - x1
        dy = y2 - y1

        line_coeffiecients = np.zeros((len(segments), 2))

        vertical_mask = np.isclose(dx, 0, atol=1e-8)
        non_vertical_mask = ~vertical_mask

        if np.any(non_vertical_mask):
            m = dy[non_vertical_mask] / dx[non_vertical_mask]
            q = y1[non_vertical_mask] - m * x1[non_vertical_mask]
            line_coeffiecients[non_vertical_mask, 0] = m
            line_coeffiecients[non_vertical_mask, 1] = q

        if np.any(vertical_mask):
            line_coeffiecients[vertical_mask, 0] = np.inf
            line_coeffiecients[vertical_mask, 1] = x1[vertical_mask]

        return line_coeffiecients

    def __calculate_line_equation_from_segments(self, segments):
        x1 = segments[:, 0]
        y1 = segments[:, 1]
        x2 = segments[:, 2]
        y2 = segments[:, 3]

        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1

        return np.stack((A, B, C), axis=1)

    def __segment_to_hesse_normal(self, segment):

        x1, y1, x2, y2 = segment
        
        # Handle vertical segments (x1 == x2)
        if np.isclose(x1, x2):
            theta = np.pi/2  # 90 degrees (normal to vertical line)
            rho = x1  # Distance from origin along x-axis
        else:
            # Standard line: Ax + By + C = 0
            A = y2 - y1
            B = x1 - x2
            C = x2*y1 - x1*y2
            
            # Convert to Hesse normal form
            norm = np.sqrt(A**2 + B**2)
            rho = -C / norm  # Perpendicular distance from origin
            theta = np.arctan2(B, A)  # Angle of normal vector
        
        return rho, theta

    def __intersect_hesse_lines(self, line1, line2, img_height, img_width):
        rho1, theta1 = line1
        rho2, theta2 = line2
        
        # Check if lines are parallel (no intersection)
        if np.isclose(np.abs(theta1 - theta2), 0, atol=1e-6) or \
        np.isclose(np.abs(theta1 - theta2), np.pi, atol=1e-6):
            return None
        
        # Solve intersection
        A = np.array([
            [np.cos(theta1), np.sin(theta1)],
            [np.cos(theta2), np.sin(theta2)]
        ])
        b = np.array([[rho1], [rho2]])
        
        try:
            x, y = np.linalg.solve(A, b)
            # Clip to image bounds
            x = np.clip(int(np.round(x)), 0, img_width - 1)
            y = np.clip(int(np.round(y)), 0, img_height - 1)
            return (x, y)
        except np.linalg.LinAlgError:
            return None  # Parallel case (should be caught above)


    def __create_segments_from_lines(self, lines, segment_length=10000):
        segments = np.zeros((lines.shape[0], 4), dtype=np.int32)

        for index, line in enumerate(lines):
            rho, theta = line
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho

            x1 = int(x0 + segment_length * (-b))
            y1 = int(y0 + segment_length * (a))
            x2 = int(x0 - segment_length * (-b))
            y2 = int(y0 - segment_length * (a))

            segments[index] = (x1, y1, x2, y2)

        return segments

    def __extend_segments_size(self, segments, max_height, max_width):

        extended_segments = np.zeros((segments.shape[0], 4), dtype=np.int32)

        for index, segment in enumerate(segments):
            x1, y1, x2, y2 = segment

            # Vertical line (m = 0)
            if x1 == x2:
                extended_segments[index] = (x1, 0, x1, max_height)
                continue

            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1

            # Intersections with the left/right image sides
            x_left = 0
            x_right = max_width
            y_left = m * x_left + b
            y_right = m * x_right + b

            # Intersections with the top/bottom image sides
            y_top = 0
            y_bottom = max_height
            x_top = (y_top - b) / m
            x_bottom = (y_bottom - b) / m

            candidates = []

            # check if the left side intersection is in the image
            if y_left >= 0 and y_left <= max_height:
                candidates.append((x_left, y_left))

            # check if the right side intersection is in the image
            if y_right >= 0 and y_right <= max_height:
                candidates.append((x_right, y_right))

            # check if the top side intersection is in the image
            if x_top >= 0 and x_top <= max_width:
                candidates.append((x_top, y_top))

            # check if the bottom side intersection is in the image
            if x_bottom >= 0 and x_bottom <= max_width:
                candidates.append((x_bottom, y_bottom))

            # Normally there should be only 2 candidates (left/right or top/bottom)
            # but for perfectly diagonal line, the intersection points are the two corners of the image
            if len(candidates) >= 2:
                candidates = np.array(candidates)
                candidates = candidates[np.argsort(candidates[:, 0])]
                extended_segments[index] = (*candidates[0], *candidates[-1])
            else:
                # this should never happen
                extended_segments[index] = (x1, y1, x2, y2)

        return extended_segments

    def __draw_segments_on_image(
        self, segments, img, labels=None, use_black_background=False
    ):
        img_height, img_width = img.shape[:2]

        if use_black_background:
            segment_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        else:
            segment_img = img.copy()

        if segments is None:
            return segment_img

        for index, segment in enumerate(segments):
            x1, y1, x2, y2 = segment
            color = (0, 255, 0)

            if labels is not None:
                if labels[index] == 0:
                    color = (255, 0, 0)
                else:
                    color = (0, 0, 255)

            cv2.line(segment_img, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=4)

        return segment_img

    def __calculate_segments_distance_matrix(self, segments):
        n = segments.shape[0]
        distance_matrix = np.zeros((n, n))

        line_equations = self.__calculate_line_equation_from_segments(segments)

        x1 = segments[:, 0]
        x2 = segments[:, 2]
        mx = (x1 + x2) / 2

        y1 = segments[:, 1]
        y2 = segments[:, 3]
        my = (y1 + y2) / 2

        # loop all the lines and calculate the distance between the line and the midpoints of all other segments
        for i in range(n):
            A, B, C = line_equations[i]
            for j in range(i + 1, n):
                distance = np.abs(A * mx[j] + B * my[j] + C) / np.sqrt(A**2 + B**2)
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

        return distance_matrix

    def __calculate_segments_distance_matrix2(self, segments):
        """
        Compute a hybrid distance metric for non-parallel but similarly angled lines.
        Combines:
        - Perpendicular distance between lines at their midpoint.
        - Penalty for angular deviation.
        """
        n = len(segments)
        distance_matrix = np.zeros((n, n))
        line_equations = self.__calculate_line_equation_from_segments(segments)
        
        # Precompute midpoints
        midpoints = (segments[:, :2] + segments[:, 2:]) / 2
        
        for i in range(n):
            A1, B1, C1 = line_equations[i]
            norm1 = np.sqrt(A1**2 + B1**2 + 1e-10)
            
            for j in range(i + 1, n):
                A2, B2, C2 = line_equations[j]
                norm2 = np.sqrt(A2**2 + B2**2 + 1e-10)
                
                # --- 1. Perpendicular Distance ---
                # Distance from line i to midpoint of j
                d_i_to_j = np.abs(A1 * midpoints[j, 0] + B1 * midpoints[j, 1] + C1) / norm1
                # Distance from line j to midpoint of i
                d_j_to_i = np.abs(A2 * midpoints[i, 0] + B2 * midpoints[i, 1] + C2) / norm2
                avg_lateral_dist = (d_i_to_j + d_j_to_i) / 2
                
                # --- 2. Angular Penalty ---
                cos_theta = (A1*A2 + B1*B2) / (norm1 * norm2)
                theta = np.arccos(np.clip(cos_theta, -1, 1))  # Angle in radians
                angle_penalty = np.sin(theta)  # Penalize deviation from parallel
                
                # --- Combine Metrics ---
                hybrid_distance = avg_lateral_dist * (1 + angle_penalty)
                distance_matrix[i, j] = distance_matrix[j, i] = hybrid_distance
        
        return distance_matrix

    def __debug_line_angles(self, angles, labels=None):
        if labels is None:
            colors = None
        else:
            colors = ["red" if label == 0 else "blue" for label in labels]

        plt.scatter(angles, np.zeros_like(angles), c=colors)
        plt.xlabel("Angle (radians)")
        plt.yticks([])
        plt.show()

    def __plot_vectors(self, vectors, labels):
        origin = np.zeros((vectors.shape[0], 2))
        plt.figure(figsize=(6, 6))

        unique_labels = np.unique(labels)

        # Up to 10 distinct colors
        colormap = cm.get_cmap("tab10", len(unique_labels))

        for i, label in enumerate(unique_labels):
            mask = labels == label
            color = colormap(i)

            plt.quiver(
                origin[mask, 0],
                origin[mask, 1],
                vectors[mask, 0],
                vectors[mask, 1],
                angles="xy",
                scale_units="xy",
                scale=1,
                color=color,
                alpha=0.8,
                label=f"Cluster {label}",
            )

        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        plt.gca().set_aspect("equal")
        plt.grid(True)
        plt.legend()
        plt.title("Direction Vectors by Cluster")
        plt.xlabel("dx")
        plt.ylabel("dy")
        plt.show()


# ----------------------------------------------------------

if __name__ == "__main__":
    image_path = os.path.abspath("data/chessred2k/images/0/G000_IMG001.jpg")
    img = cv2.imread(image_path)

    detector = ChessboardDetector(img)
    detector.process()
