import os
import cv2
import math
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN

import drawing as dw
import line_math as lm
import image_processing as ip
import quad_finder as qf

from line_segment_linking import link_and_merge_segments_array


class ChessboardDetector:

    original_image: cv2.typing.MatLike
    scaled_image: cv2.typing.MatLike
    processed_image: cv2.typing.MatLike

    edges_image: cv2.typing.MatLike
    segments_image: cv2.typing.MatLike
    lines_image: cv2.typing.MatLike

    clustered_lines_image: cv2.typing.MatLike
    clustered_lines_filtered_image: cv2.typing.MatLike
    clustered_lines_merged_image: cv2.typing.MatLike

    intersections_image: cv2.typing.MatLike
    filtered_intersections_image: cv2.typing.MatLike

    quad_image: cv2.typing.MatLike
    quad_image_black: cv2.typing.MatLike

    lines_angles = None
    line_segments = None
    lines_labels = None
    intersections_matrix = None
    filtered_intersection_points = None

    filtered_line_segments = None
    filtered_line_labels = None
    merged_line_segments = None
    merged_line_labels = None

    def __init__(self, img):
        self.original_image = img

    def process(self):
        print(
            f"original image size: {self.original_image.shape[:2][0]}x{self.original_image.shape[:2][1]}"
        )
        # dw.display_image_cv2(self.original_image)

        self.scaled_image, self.processed_image = ip.apply_image_processing(
            self.original_image
        )
        print(
            f"scaled image size: {self.scaled_image.shape[:2][0]}x{self.scaled_image.shape[:2][1]}"
        )
        # dw.display_image_cv2(self.processed_image)

        self.__detect_edges()
        # dw.display_image_cv2(self.edges_image)

        self.__detect_lines()
        # dw.display_image_cv2(self.segments_image)
        # dw.display_image_cv2(self.lines_image)

        self.__classify_lines()
        # dw.display_image_cv2(self.clustered_lines_image)

        self.__filter_lines_duplicates()

        print(f"segments before filtering: {len(self.line_segments)}")
        print(f"segments after filtering: {len(self.filtered_line_segments)}")
        # dw.display_image_cv2(self.clustered_lines_filtered_image)

        self.__merge_close_segments()

        print(f"segments after merging: {len(self.merged_line_segments)}")
        # dw.display_image_cv2(self.clustered_lines_merged_image)

        self.__find_segment_intersections()
        dw.display_image_cv2(self.intersections_image)

        # self.__filter_duplicate_intersection_points()
        # dw.display_image_cv2(self.filtered_intersections_image)

        self.__get_squares()
        dw.display_image_cv2(self.quad_image)
        dw.display_image_cv2(self.quad_image_black)

    def __detect_edges(self):
        canny = cv2.Canny(
            self.processed_image,
            threshold1=50,  # lower hysteresis threshold
            threshold2=150,  # upper hysteresis threshold,
            apertureSize=3,  # sobel kernel size
            L2gradient=True,  # false = l1 norm (faster), true = l2 norm (more accurate)
        )

        dilated_edges = cv2.dilate(canny, np.ones((3, 3), np.uint8), iterations=1)

        # This is a dilation followed by an erosion
        # closed_edges = cv2.morphologyEx(
        #    canny, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
        # )

        self.edges_image = dilated_edges

    def __detect_lines(self, expansion=200):
        hough_segments = cv2.HoughLinesP(  # Probabilistic Hough Transform (PHT)
            self.edges_image,
            rho=1,  # 1 pixel resolution -> this is usually fine
            theta=math.pi
            / 180,  # 1 degree resolution (in radiants) -> this is usually fine
            threshold=100,  # lower = detect more lines (including noise), higher = keep only clear lines
            minLineLength=200,  # lower = detect short and fragmented lines, higher = discard short and fragmented lines
            maxLineGap=15,  # lower = close lines are merged, higher = close lines are considered separate
        )

        segments = np.array(hough_segments).squeeze()

        self.segments_image = dw.create_image_with_segments(segments, self.scaled_image)

        x1 = segments[:, 0]
        y1 = segments[:, 1]
        x2 = segments[:, 2]
        y2 = segments[:, 3]

        dx = x2 - x1
        dy = y2 - y1

        # angles between [-π, π]
        self.lines_angles = np.arctan2(dy, dx)

        img_height, img_width = self.scaled_image.shape[:2]

        # self.line_segments = lm.extend_segments_size(
        #    segments, img_height, img_width
        # )

        self.line_segments = lm.expand_segments_relative(
            segments, expansion, img_height, img_width
        )

        self.lines_image = dw.create_image_with_segments(
            self.line_segments, self.scaled_image
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

        self.line_segments = lm.convert_hesse_normal_form_lines_to_segments(lines)
        self.lines_image = dw.create_image_with_segments(
            self.line_segments, self.scaled_image
        )

    def __classify_lines(self):

        # transformation used to made sure all multiple of π (including 0) are treated as the same angle but still keeping the separation between the other angles
        doubled_line_angles = self.lines_angles * 2

        features = np.stack(
            (np.cos(doubled_line_angles), np.sin(doubled_line_angles)), axis=1
        )

        kmeans = KMeans(n_clusters=2)
        self.lines_labels = kmeans.fit(features).labels_
        self.clustered_lines_image = dw.create_image_with_segments(
            self.line_segments, self.scaled_image, segments_labels=self.lines_labels
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
        self.clustered_lines_image = dw.create_image_with_segments(
            self.line_segments, self.scaled_image, segments_labels=self.lines_labels
        )

    def __filter_lines_duplicates(self, rho_tol=10, eps=0.02):

        lines = lm.convert_segments_to_hesse_normal_form_lines(self.line_segments)
        merged = []
        merged_labels = []

        for label in np.unique(self.lines_labels):

            group_mask = self.lines_labels == label
            group_lines = lines[group_mask]
            group_segments = self.line_segments[group_mask]

            X = group_lines[:, 0].reshape(-1, 1) / rho_tol
            clustering = DBSCAN(eps=eps, min_samples=1).fit(X)

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

        self.clustered_lines_filtered_image = dw.create_image_with_segments(
            self.filtered_line_segments,
            self.scaled_image,
            segments_labels=self.filtered_line_labels,
        )

    def __merge_close_segments(self, p=0.9):
        img_height, img_width = self.scaled_image.shape[:2]

        merged = []
        merged_labels = []

        segments = self.filtered_line_segments
        labels = self.filtered_line_labels

        for label in np.unique(labels):
            group_mask = labels == label
            group_segments = segments[group_mask]
            out = link_and_merge_segments_array(
                group_segments, img_height * img_width, p
            )

            for index, item in enumerate(out):
                merged.append(item)
                merged_labels.append(label)

        self.merged_line_segments = np.array(merged)
        self.merged_line_labels = np.array(merged_labels)

        self.clustered_lines_merged_image = dw.create_image_with_segments(
            self.merged_line_segments,
            self.scaled_image,
            segments_labels=self.merged_line_labels,
        )

    def __find_line_intersections(self):

        segments = self.filtered_line_segments
        labels = self.filtered_line_labels

        hesse_normal_lines = lm.convert_segments_to_hesse_normal_form_lines(segments)

        N = len(segments)

        # Using dense matrix even if this can be optimized with a sparse matrix
        intersections_matrix = np.full((N, N, 2), np.nan)
        img_height, img_width = self.scaled_image.shape[:2]

        for i in range(N):
            for j in range(i + 1, N):

                # Only check the intersections between lines belonging to different clusters
                # (vertical-ish lines vs horizontal-ish lines)
                if labels[i] == labels[j]:
                    continue

                pt = lm.calculate_hesse_normal_form_lines_intersection_point(
                    hesse_normal_lines[i], hesse_normal_lines[j], img_height, img_width
                )

                # Store intersection points only when results are found
                if pt is not None:
                    x, y = pt

                    # the matrix is symmetric
                    intersections_matrix[i, j] = [x, y]
                    intersections_matrix[j, i] = [x, y]

        self.intersections_matrix = intersections_matrix

        # Flatten the matrix
        intersections_matrix_flattened = intersections_matrix.reshape(-1, 2)

        # Remove nans
        nan_filter_mask = ~np.isnan(intersections_matrix_flattened).any(axis=1)
        points = intersections_matrix_flattened[nan_filter_mask]

        # Since the intersection matrix is symmetrical, remove the duplicates
        unique_points = np.unique(points, axis=0)

        self.intersections_image = dw.create_image_with_points(
            unique_points, self.clustered_lines_filtered_image
        )

    def __find_segment_intersections(self):

        segments = self.merged_line_segments
        labels = self.merged_line_labels

        N = len(segments)

        # Using dense matrix even if this can be optimized with a sparse matrix
        intersections_matrix = np.full((N, N, 2), np.nan)

        for i in range(N):
            for j in range(i + 1, N):

                # Only check the intersections between lines belonging to different clusters
                # (vertical-ish lines vs horizontal-ish lines)
                if labels[i] == labels[j]:
                    continue

                pt = lm.calculate_segments_intersection_point(segments[i], segments[j])

                # Store intersection points only when results are found
                if pt is not None:
                    x, y = pt

                    # the matrix is symmetric
                    intersections_matrix[i, j] = [x, y]
                    intersections_matrix[j, i] = [x, y]

        nan_mask = np.isnan(intersections_matrix)
        converted_intersection_mask = np.full_like(
            intersections_matrix, -1, dtype=np.int32
        )
        converted_intersection_mask[~nan_mask] = intersections_matrix[~nan_mask].astype(
            np.int32
        )

        self.intersections_matrix = converted_intersection_mask

        # Flatten the matrix
        intersections_matrix_flattened = self.intersections_matrix.reshape(-1, 2)

        # Remove nans
        nan_filter_mask = ~np.isnan(intersections_matrix_flattened).any(axis=1)
        points = intersections_matrix_flattened[nan_filter_mask]

        # Since the intersection matrix is symmetrical, remove the duplicates
        unique_points = np.unique(points, axis=0)

        self.intersections_image = dw.create_image_with_points(
            unique_points, self.clustered_lines_merged_image
        )

    def __filter_duplicate_intersection_points(self, eps=1):

        # Flatten the matrix
        intersections_matrix_flattened = self.intersections_matrix.reshape(-1, 2)

        # Remove nans
        nan_filter_mask = ~np.isnan(intersections_matrix_flattened).any(axis=1)
        points = intersections_matrix_flattened[nan_filter_mask]

        # Since the intersection matrix is symmetrical, remove the duplicates
        unique_points = np.unique(points, axis=0)

        clustering = DBSCAN(eps=eps, min_samples=1).fit(unique_points)
        unique_labels = np.unique(clustering.labels_)
        centers = np.array(
            [
                unique_points[clustering.labels_ == label].mean(axis=0)
                for label in unique_labels
            ]
        )

        self.filtered_points = centers

        self.filtered_intersections_image = dw.create_image_with_points(
            self.filtered_points, self.scaled_image
        )

        print(f"points before filtering: {len(unique_points)}")
        print(f"points after filtering: {len(self.filtered_points)}")

    def __get_squares(self):
        quads = qf.find_quads(self.intersections_matrix)
        squares = qf.filter_squares_fast(quads, rtol=0.2, atol=20)
        self.quad_image = dw.create_image_with_quads(squares, self.scaled_image)

        img_height, img_width = self.scaled_image.shape[:2]

        black_image = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        self.quad_image_black = dw.create_image_with_quads(squares, black_image)


# ----------------------------------------------------------


def get_corner_harris(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst > 0.01 * dst.max()] = [0, 0, 255]

    dw.display_image_cv2(img)


if __name__ == "__main__":
    image_path = os.path.abspath("data/chessred2k/images/0/G000_IMG015.jpg")
    img = cv2.imread(image_path)

    detector = ChessboardDetector(img)
    detector.process()

    # get_corner_harris(image_path)
