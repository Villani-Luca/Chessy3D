import os
import cv2
import math
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
import pyautogui
import matplotlib.pyplot as plt

from line_segment_linking import link_and_merge_segments_array 

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
        self.__filter_lines_duplicates()
        #self.__filter_lines_duplicates2()
        self.__find_segment_intersections()
        self.__filter_duplicate_intersection_points()

        # self.__debug_image_cv2(self.original_image)
        # self.__debug_image_cv2(self.processed_image)
        # self.__debug_image_cv2(self.edges_image)
        # self.__debug_image_cv2(self.segments_image)
        # self.__debug_image_cv2(self.lines_image)
        self.__debug_image_cv2(self.clustered_lines_image)
        self.__debug_image_cv2(self.clustered_lines_filtered_image)
        self.__debug_image_cv2(self.intersections_image)
        self.__debug_image_cv2(self.filtered_intersections_image)

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

        # self.line_segments = self.__extend_segments_size(
        #    segments, img_height, img_width
        # )

        self.line_segments = self.__extend_segment_relative(
            segments, img_height, img_width, 400
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

    def __filter_lines_duplicates(self, rho_tol=20):

        lines = self.__segments_to_hesse_normal_lines(self.line_segments)
        merged = []
        merged_labels = []

        for label in np.unique(self.lines_labels):

            group_mask = self.lines_labels == label
            group_lines = lines[group_mask]
            group_segments = self.line_segments[group_mask]

            X = group_lines[:, 0].reshape(-1, 1) / rho_tol
            clustering = DBSCAN(eps=0.5, min_samples=1).fit(X)

            for cluster_id in np.unique(clustering.labels_):

                cluster_mask = clustering.labels_ == cluster_id
                cluster_segments = group_segments[cluster_mask]

                # Skip if only one segment in cluster
                if len(cluster_segments) == 1:
                    merged.append(cluster_segments[0])
                    merged_labels.append(label)
                    continue

                # Merge segments using PCA (weighted by segment length)
                points = cluster_segments.reshape(-1, 2)  # Shape: (2N, 2)
                lengths = np.linalg.norm(
                    cluster_segments[:, 2:] - cluster_segments[:, :2], axis=1
                )
                weights = np.repeat(lengths, 2)  # Weight endpoints by segment length

                # Weighted PCA to find dominant direction
                mean = np.average(points, axis=0, weights=weights)
                centered = points - mean
                _, _, Vt = np.linalg.svd(centered * weights[:, np.newaxis])
                direction = Vt[0]  # Principal direction

                projections = np.dot(centered, direction)
                p1 = mean + np.min(projections) * direction
                p2 = mean + np.max(projections) * direction
                merged.append((int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])))
                merged_labels.append(label)

        self.filtered_line_segments = np.array(merged)
        self.filtered_line_labels = np.array(merged_labels)

        self.clustered_lines_filtered_image = self.__draw_segments_on_image(
            self.filtered_line_segments,
            self.original_image,
            self.filtered_line_labels,
        )

        # x1 = self.filtered_line_segments[:, 0:2]
        # x2 = self.filtered_line_segments[:, 2:4]
        # st = np.concat((x1, x2), axis=0)
        # self.clustered_lines_filtered_image = self.__draw_points(st, self.clustered_lines_filtered_image, size=10)

        print(f"lines before filtering: {len(self.line_segments)}")
        print(f"lines after filtering: {len(self.filtered_line_segments)}")

    def __filter_lines_duplicates2(self):
        img_height, img_width = self.original_image.shape[:2]

        print(f"lines before merging: {len(self.filtered_line_segments)}")

        merged = []
        merged_labels = []

        segments = self.filtered_line_segments
        labels = self.filtered_line_labels

        for label in np.unique(labels):
            group_mask = labels == label
            group_segments = segments[group_mask]
            out = link_and_merge_segments_array(group_segments, img_height*img_width, 0.4)

            for index, item in enumerate(out):
                merged.append(item)
                merged_labels.append(label)

        self.filtered_line_segments = np.array(merged)
        self.filtered_line_labels = np.array(merged_labels)

        self.clustered_lines_filtered_image = self.__draw_segments_on_image(
            self.filtered_line_segments,
            self.original_image,
            self.filtered_line_labels,
        )

        
        print(f"lines after merging: {len(self.filtered_line_segments)}")

    def __find_line_intersections(self):

        segments = self.filtered_line_segments
        labels = self.filtered_line_labels

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
            intersections_matrix, self.clustered_lines_filtered_image
        )

    def __find_segment_intersections(self):

        segments = self.filtered_line_segments
        labels = self.filtered_line_labels

        N = len(segments)

        # Using dense matrix even if this can be optimized with a sparse matrix
        intersections_matrix = np.full((N, N, 2), np.nan)

        for i in range(N):
            for j in range(i + 1, N):

                # Only check the intersections between lines belonging to different clusters
                # (vertical-ish lines vs horizontal-ish lines)
                if labels[i] == labels[j]:
                    continue

                pt = self.__calculate_segment_intersection(segments[i], segments[j])

                # Store intersection points only when results are found
                if pt is not None:
                    x, y = pt

                    # the matrix is symmetric
                    intersections_matrix[i, j] = [x, y]
                    intersections_matrix[j, i] = [x, y]

        self.intersections_matrix = intersections_matrix

        self.intersections_image = self.__draw_intersection_points(
            intersections_matrix, self.clustered_lines_filtered_image
        )

    def __filter_duplicate_intersection_points(self):

        # Flatten the matrix
        intersections_matrix_flattened = self.intersections_matrix.reshape(-1, 2)

        # Remove nans
        nan_filter_mask = ~np.isnan(intersections_matrix_flattened).any(axis=1)
        points = intersections_matrix_flattened[nan_filter_mask]

        # Since the intersection matrix is symmetrical, remove the duplicates
        unique_points = np.unique(points, axis=0)

        clustering = DBSCAN(eps=1, min_samples=1).fit(unique_points)
        unique_labels = np.unique(clustering.labels_)
        centers = np.array(
            [
                unique_points[clustering.labels_ == label].mean(axis=0)
                for label in unique_labels
            ]
        )

        self.filtered_points = centers

        self.filtered_intersections_image = self.__draw_points(
            self.filtered_points, self.original_image
        )

        print(f"points before filtering: {len(unique_points)}")
        print(f"points after filtering: {len(self.filtered_points)}")

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

    def __calculate_segment_intersection(self, segment1, segment2, epsilon=1e-10):
        x1, y1, x2, y2 = segment1
        x3, y3, x4, y4 = segment2

        a1 = y2 - y1
        b1 = x1 - x2
        c1 = a1 * x1 + b1 * y1

        a2 = y4 - y3
        b2 = x3 - x4
        c2 = a2 * x3 + b2 * y3

        det = a1 * b2 - a2 * b1

        # Check if lines are parallel
        if abs(det) < epsilon:
            return None

        # Intersection point
        x = (b2 * c1 - b1 * c2) / det
        y = (a1 * c2 - a2 * c1) / det

        x_in_seg1 = np.isclose(x, np.clip(x, min(x1, x2), max(x1, x2)), atol=epsilon)
        y_in_seg1 = np.isclose(y, np.clip(y, min(y1, y2), max(y1, y2)), atol=epsilon)
        x_in_seg2 = np.isclose(x, np.clip(x, min(x3, x4), max(x3, x4)), atol=epsilon)
        y_in_seg2 = np.isclose(y, np.clip(y, min(y3, y4), max(y3, y4)), atol=epsilon)

        if x_in_seg1 and y_in_seg1 and x_in_seg2 and y_in_seg2:
            return (x, y)

        return None

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

    def __extend_segment_relative(
        self, segments, max_height, max_width, extend_length=250
    ):
        extended_segments = np.zeros((segments.shape[0], 4), dtype=np.int32)

        for index, segment in enumerate(segments):
            x1, y1, x2, y2 = segment

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            dx = x2 - x1
            dy = y2 - y1

            length = np.sqrt(dx**2 + dy**2)

            unit_dx = dx / length
            unit_dy = dy / length

            new_x1 = center_x - unit_dx * (length / 2 + extend_length)
            new_y1 = center_y - unit_dy * (length / 2 + extend_length)
            new_x2 = center_x + unit_dx * (length / 2 + extend_length)
            new_y2 = center_y + unit_dy * (length / 2 + extend_length)

            # clip segments to the image size
            clipped_x1, clipped_y1, clipped_x2, clipped_y2 = self.__clip_segment(
                new_x1, new_y1, new_x2, new_y2, max_height, max_width
            )

            extended_segments[index] = (
                int(np.round(clipped_x1)),
                int(np.round(clipped_y1)),
                int(np.round(clipped_x2)),
                int(np.round(clipped_y2)),
            )

        return extended_segments

    def __clip_segment(self, x1, y1, x2, y2, height, width):
        # Parametric line equations: x = x1 + t*(x2-x1), y = y1 + t*(y2-y1)
        t_values = [0.0, 1.0]  # Start with original segment

        # Calculate intersection with image boundaries
        for boundary in [0, width]:
            if (x2 - x1) != 0:  # Avoid division by zero
                t = (boundary - x1) / (x2 - x1)
                y = y1 + t * (y2 - y1)
                if 0 <= y <= height:
                    t_values.append(t)

        for boundary in [0, height]:
            if (y2 - y1) != 0:  # Avoid division by zero
                t = (boundary - y1) / (y2 - y1)
                x = x1 + t * (x2 - x1)
                if 0 <= x <= width:
                    t_values.append(t)

        # Find the valid t-range that stays within image
        t_min = max(min(t_values), 0.0)
        t_max = min(max(t_values), 1.0)

        # Return clipped coordinates
        return (
            x1 + t_min * (x2 - x1),
            y1 + t_min * (y2 - y1),
            x1 + t_max * (x2 - x1),
            y1 + t_max * (y2 - y1),
        )

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

    def __draw_points(self, points, img: cv2.typing.MatLike, size=5):

        img_copy = img.copy()

        for p in points:
            x, y = p
            x_pixel = int(x)
            y_pixel = int(y)

            cv2.circle(
                img_copy,
                (x_pixel, y_pixel),
                size,
                (0, 255, 0),
                -1,
            )

        return img_copy


# ----------------------------------------------------------


def debug_image_cv2(img: cv2.typing.MatLike, max_height=1024, max_width=1024):

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


def get_corner_harris(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst > 0.01 * dst.max()] = [0, 0, 255]

    debug_image_cv2(img)


if __name__ == "__main__":
    image_path = os.path.abspath("data/chessred2k/images/0/G000_IMG015.jpg")
    img = cv2.imread(image_path)

    detector = ChessboardDetector(img)
    detector.process()

    # get_corner_harris(image_path)
