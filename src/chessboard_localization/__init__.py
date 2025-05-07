import os
import cv2
import math
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
import pyautogui


class ChessboardDetector:
    original_image: cv2.typing.MatLike
    processed_image: cv2.typing.MatLike
    edges_image: cv2.typing.MatLike
    segments_image: cv2.typing.MatLike
    lines_image: cv2.typing.MatLike
    clustered_lines_image: cv2.typing.MatLike
    clustered_lines_filtered_image: cv2.typing.MatLike
    intersections_image: cv2.typing.MatLike
    filtered_intersections_image: cv2.typing.MatLike

    lines_angles = None
    line_segments = None
    lines_labels = None
    intersections_matrix = None
    filtered_intersection_points = None

    filtered_line_segments = None
    filtered_line_labels = None

    def __init__(self, img):
        self.original_image = img

    # TODO: define image size dependant parameters (multi scale processing)
    # alternatively run the processing many times with different parameters and take the best/merge the results

    def process(self):
        self.__apply_image_processing()
        self.__detect_edges()
        self.__detect_lines()
        self.__classify_lines()
        # self.__filter_lines_duplicates()
        self.__find_line_intersections()
        self.__filter_duplicate_intersection_points()

        # print(self.lines_angles)
        # print(self.line_segments)
        # print(self.lines_labels)

        # self.__debug_image_cv2(self.original_image)
        # self.__debug_image_cv2(self.processed_image)
        # self.__debug_image_cv2(self.edges_image)
        # self.__debug_image_cv2(self.segments_image)
        # self.__debug_image_cv2(self.lines_image)
        # self.__debug_image_cv2(self.clustered_lines_image)
        # self.__debug_image_cv2(self.clustered_lines_filtered_image)
        # self.__debug_image_cv2(self.intersections_image)
        self.__debug_image_cv2(self.filtered_intersections_image)

        # debug_line_angles(angles_radiants, labels)
        # plot_vectors(filtered_segments_features, labels)

    def __apply_image_processing(self):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        gaussian = cv2.GaussianBlur(
            gray,
            ksize=(7, 7),  # kernel size -> since the image is big this must be big
            sigmaX=1.5,  # horizontal blur intensity
            sigmaY=1.5,  # vertical blur intensity
        )

        # bilateral = cv2.bilateralFilter(gray, 9, 75, 75)

        self.processed_image = gaussian

    def __detect_edges(self):
        canny = cv2.Canny(
            self.processed_image,
            threshold1=50,  # lower hysteresis threshold
            threshold2=150,  # upper hysteresis threshold,
            apertureSize=3,  # sobel kernel size
            L2gradient=True,  # false = l1 norm (faster), true = l2 norm (more accurate)
        )

        dilated_edges = cv2.dilate(canny, np.ones((5, 5), np.uint8), iterations=1)

        # This is a dilation followed by an erosion
        # closed_edges = cv2.morphologyEx(
        #    canny, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
        # )

        self.edges_image = dilated_edges

    def __detect_lines(self):
        hough_segments = cv2.HoughLinesP(  # Probabilistic Hough Transform (PHT)
            self.edges_image,
            rho=1,  # 1 pixel resolution -> this is usually fine
            theta=math.pi
            / 180,  # 1 degree resolution (in radiants) -> this is usually fine
            threshold=100,  # lower = detect more lines (including noise), higher = keep only clear lines
            minLineLength=200,  # lower = detect short and fragmented lines, higher = discard short and fragmented lines
            maxLineGap=10,  # lower = close lines are merged, higher = close lines are considered separate
        )

        segments = np.array(hough_segments).squeeze()

        self.segments_image = self.__draw_segments_on_image(
            segments, self.original_image
        )

        x1 = segments[:, 0]
        y1 = segments[:, 1]
        x2 = segments[:, 2]
        y2 = segments[:, 3]

        dx = x2 - x1
        dy = y2 - y1

        # angles between [-π, π]
        self.lines_angles = np.arctan2(dy, dx)

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

        # transformation used to made sure all multiple of π (including 0) are treated as the same angle but still keeping the separation between the other angles
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

        # transformation used to made sure all multiple of π (including 0) are treated as the same angle but still keeping the separation between the other angles
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

    def __find_line_intersections(self):

        segments = self.line_segments
        labels = self.lines_labels

        hesse_normal_lines = self.__segments_to_hesse_normal_lines(segments)

        N = len(segments)

        # Using dense matrix even if this can be optimized with a sparse matrix
        intersections_matrix = np.full((N, N, 2), np.nan)
        img_height, img_width = self.original_image.shape[:2]

        for i in range(N):
            for j in range(i + 1, N):

                # Only check the intersections between lines belonging to different clusters
                # (vertical-ish lines vs horizontal-ish lines)
                if labels[i] == labels[j]:
                    continue

                pt = self.__intersect_hesse_lines(
                    hesse_normal_lines[i], hesse_normal_lines[j], img_height, img_width
                )

                # Store intersection points only when results are found
                if pt is not None:
                    x, y = pt

                    # the matrix is symmetric
                    intersections_matrix[i, j] = [x, y]
                    intersections_matrix[j, i] = [x, y]

        self.intersections_matrix = intersections_matrix

        self.intersections_image = self.__draw_intersection_points(
            intersections_matrix, self.clustered_lines_image
        )

    def __filter_duplicate_intersection_points(self):

        # Flatten the matrix
        intersections_matrix_flattened = self.intersections_matrix.reshape(-1, 2)

        # Remove nans
        nan_filter_mask = ~np.isnan(intersections_matrix_flattened).any(axis=1)
        points = intersections_matrix_flattened[nan_filter_mask]

        # Since the intersection matrix is symmetrical, remove the duplicates
        unique_points = np.unique(points, axis=0)

        print(f"points before filtering: {len(unique_points)}")

        clustering = DBSCAN(eps=1, min_samples=1).fit(unique_points)
        unique_labels = np.unique(clustering.labels_)
        centers = np.array(
            [
                unique_points[clustering.labels_ == label].mean(axis=0)
                for label in unique_labels
            ]
        )

        print(f"points after filtering: {len(centers)}")

        self.filtered_points = centers
        self.filtered_intersections_image = self.__draw_points(
            self.filtered_points, self.original_image
        )

    # ----------------------------------------------------------

    def __debug_image_cv2(
        self, img: cv2.typing.MatLike, max_height=1024, max_width=1024
    ):

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

    def __calculate_line_slope_intercept_from_segments(self, segments):
        x1 = segments[:, 0]
        y1 = segments[:, 1]
        x2 = segments[:, 2]
        y2 = segments[:, 3]

        dx = x2 - x1
        dy = y2 - y1

        line_coefficients = np.zeros((len(segments), 2))

        vertical_mask = np.isclose(x1, x2)
        non_vertical_mask = ~vertical_mask

        if np.any(non_vertical_mask):
            m = dy[non_vertical_mask] / dx[non_vertical_mask]
            q = y1[non_vertical_mask] - m * x1[non_vertical_mask]
            line_coefficients[non_vertical_mask, 0] = m
            line_coefficients[non_vertical_mask, 1] = q

        if np.any(vertical_mask):
            line_coefficients[vertical_mask, 0] = np.inf
            line_coefficients[vertical_mask, 1] = x1[vertical_mask]

        return line_coefficients

    def __calculate_line_equation_from_segments(self, segments):
        x1 = segments[:, 0]
        y1 = segments[:, 1]
        x2 = segments[:, 2]
        y2 = segments[:, 3]

        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1

        return np.stack((A, B, C), axis=1)

    def __segments_to_hesse_normal_lines(self, segments):

        hesse_normal_lines = np.zeros((len(segments), 2))

        x1 = segments[:, 0]
        x2 = segments[:, 2]
        y1 = segments[:, 1]
        y2 = segments[:, 3]

        vertical_mask = np.isclose(x1, x2)
        non_vertical_mask = ~vertical_mask

        if np.any(non_vertical_mask):
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2

            norm = np.sqrt(A**2 + B**2)
            rho = -C / norm
            theta = np.arctan2(B, A)

            hesse_normal_lines[non_vertical_mask, 0] = rho
            hesse_normal_lines[non_vertical_mask, 1] = theta

        if np.any(vertical_mask):
            theta = np.pi / 2
            rho = x1
            hesse_normal_lines[non_vertical_mask, 0] = rho
            hesse_normal_lines[non_vertical_mask, 1] = theta

        return hesse_normal_lines

    def __intersect_hesse_lines(self, line1, line2, img_height, img_width):

        rho1, theta1 = line1
        rho2, theta2 = line2

        cos1, sin1 = np.cos(theta1), np.sin(theta1)
        cos2, sin2 = np.cos(theta2), np.sin(theta2)

        # this is a cross product
        det = cos1 * sin2 - cos2 * sin1

        # Skip if there are no intersection (parallel lines)
        if np.isclose(det, 0, atol=1e-8):
            return None

        # Solving the linear system to find the intersection point
        A = np.array([[cos1, sin1], [cos2, sin2]])
        b = np.array([[rho1], [rho2]])

        try:
            x, y = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None

        x = int(np.round(x.item()))
        y = int(np.round(y.item()))

        # Drop points located outside the image
        if x < 0 or x >= img_width or y < 0 or y >= img_height:
            return None

        return (x, y)

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
            color = (255, 0, 0)

            if labels is not None:
                if labels[index] == 0:
                    color = (0, 0, 255)

            cv2.line(segment_img, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=4)

        return segment_img

    def __draw_intersection_points(self, intersections_matrix, img: cv2.typing.MatLike):

        # Flatten the matrix
        intersections_matrix_flattened = intersections_matrix.reshape(-1, 2)

        # Remove nans
        nan_filter_mask = ~np.isnan(intersections_matrix_flattened).any(axis=1)
        points = intersections_matrix_flattened[nan_filter_mask]

        # Since the intersection matrix is symmetrical, remove the duplicates
        unique_points = np.unique(points, axis=0)

        return self.__draw_points(unique_points, img)

    def __draw_points(self, points, img: cv2.typing.MatLike):

        img_copy = img.copy()

        for p in points:
            x, y = p
            x_pixel = int(x)
            y_pixel = int(y)

            cv2.circle(
                img_copy,
                (x_pixel, y_pixel),
                5,
                (0, 255, 0),
                -1,
            )

        return img_copy

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
                d_i_to_j = (
                    np.abs(A1 * midpoints[j, 0] + B1 * midpoints[j, 1] + C1) / norm1
                )
                # Distance from line j to midpoint of i
                d_j_to_i = (
                    np.abs(A2 * midpoints[i, 0] + B2 * midpoints[i, 1] + C2) / norm2
                )
                avg_lateral_dist = (d_i_to_j + d_j_to_i) / 2

                # --- 2. Angular Penalty ---
                cos_theta = (A1 * A2 + B1 * B2) / (norm1 * norm2)
                theta = np.arccos(np.clip(cos_theta, -1, 1))  # Angle in radians
                angle_penalty = np.sin(theta)  # Penalize deviation from parallel

                # --- Combine Metrics ---
                hybrid_distance = avg_lateral_dist * (1 + angle_penalty)
                distance_matrix[i, j] = distance_matrix[j, i] = hybrid_distance

        return distance_matrix


# ----------------------------------------------------------

if __name__ == "__main__":
    image_path = os.path.abspath("data/chessred2k/images/0/G000_IMG015.jpg")
    img = cv2.imread(image_path)

    detector = ChessboardDetector(img)
    detector.process()
