import cv2
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans, DBSCAN

import drawing as dw
import line_math as lm
from line_segment_linking import link_and_merge_segments_array
import quad_finder as qf


def get_lines(original_image, file_name):

    #dw.display_image_cv2(original_image, window_name=f"{file_name} -> original")

    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    #dw.display_image_cv2(grayscale_image, window_name=f"{file_name} -> grayscale")

    blurred_bilateral_grayscale_image = cv2.bilateralFilter(
        grayscale_image, d=15, sigmaColor=75, sigmaSpace=75
    )

    #dw.display_image_cv2(blurred_bilateral_grayscale_image, window_name=f"{file_name} -> bilateral")

    threshold_value, thresholded_bilateral_image = cv2.threshold(
        blurred_bilateral_grayscale_image,
        thresh=0,
        maxval=255,
        type=cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    #dw.display_image_cv2(thresholded_bilateral_image, window_name=f"{file_name} -> otsu")

    canny_image = cv2.Canny(
        thresholded_bilateral_image,
        threshold1=20,  # lower hysteresis threshold
        threshold2=255,  # upper hysteresis threshold,
        apertureSize=3,  # sobel kernel size
        L2gradient=True,  # false = l1 norm (faster), true = l2 norm (more accurate)
    )

    #dw.display_image_cv2(canny_image, window_name="canny")

    dilated_image = cv2.dilate(canny_image, np.ones((7, 7), np.uint8), iterations=1)

    #dw.display_image_cv2(dilated_image, window_name=f"{file_name} -> dilation")

    hough_lines = cv2.HoughLinesP(
        dilated_image,
        rho=1,
        theta=np.pi / 180,
        threshold=500,
        minLineLength=100,
        maxLineGap=150,
    )

    hough_lines = np.array(hough_lines).squeeze()

    lines_image = np.zeros(grayscale_image.shape, np.uint8)

    for line in hough_lines:
        x1, y1, x2, y2 = line
        cv2.line(
            lines_image, pt1=(x1, y1), pt2=(x2, y2), color=(255, 255, 255), thickness=4
        )

    #dw.display_image_cv2(lines_image, window_name=f"{file_name} -> lines")

    return hough_lines, lines_image

def cluster_lines(hough_lines):
    x1 = hough_lines[:, 0]
    y1 = hough_lines[:, 1]
    x2 = hough_lines[:, 2]
    y2 = hough_lines[:, 3]
    dx = x2 - x1
    dy = y2 - y1
    angles = np.arctan2(dy, dx)
    doubled_angles = angles * 2

    features = np.stack((np.cos(doubled_angles), np.sin(doubled_angles)), axis=1)
    kmeans = KMeans(n_clusters=2)
    kmean_res = kmeans.fit(features)
    labels = kmean_res.labels_

    return labels


def filter_lines_duplicates(segments, labels, rho_tol=10, eps=0.02):

    lines = lm.convert_segments_to_hesse_normal_form_lines(segments)
    merged = []
    merged_labels = []

    for label in np.unique(labels):

        group_mask = labels == label
        group_lines = lines[group_mask]
        group_segments = segments[group_mask]

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

    filtered_segments = np.array(merged)
    filtered_line_labels = np.array(merged_labels)

    return filtered_segments, filtered_line_labels


def merge_close_segments(segments, labels, img_height, img_width, p=0.9):
    merged = []
    merged_labels = []

    for label in np.unique(labels):
        group_mask = labels == label
        group_segments = segments[group_mask]
        out = link_and_merge_segments_array(group_segments, img_height * img_width, p)

        for index, item in enumerate(out):
            merged.append(item)
            merged_labels.append(label)

    merged_segments = np.array(merged)
    merged_segment_labels = np.array(merged_labels)

    return merged_segments, merged_segment_labels


def find_segment_intersections(segments, labels):

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
    converted_intersection_mask = np.full_like(intersections_matrix, -1, dtype=np.int32)
    converted_intersection_mask[~nan_mask] = intersections_matrix[~nan_mask].astype(
        np.int32
    )

    # Flatten the matrix
    intersections_matrix_flattened = converted_intersection_mask.reshape(-1, 2)

    # Remove nans
    nan_filter_mask = ~np.isnan(intersections_matrix_flattened).any(axis=1)
    points = intersections_matrix_flattened[nan_filter_mask]

    # Since the intersection matrix is symmetrical, remove the duplicates
    unique_points = np.unique(points, axis=0)

    return converted_intersection_mask, unique_points


def get_squares(intersections_matrix, rtol=0.2, atol=20):
    quads = qf.find_quads(intersections_matrix)
    squares = qf.filter_squares_fast(quads, rtol=rtol, atol=atol)
    return squares, quads