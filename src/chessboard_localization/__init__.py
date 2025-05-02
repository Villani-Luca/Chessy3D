import os
import cv2
import math
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
import pyautogui

import matplotlib.pyplot as plt
import matplotlib.cm as cm


def debug_image_cv2(img, max_width=800, max_height=800):

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


def draw_lines_on_image(
    lines, img, labels=None, use_original_image=False, line_length=10000
):

    img_height, img_width = img.shape[:2]

    if use_original_image:
        line_img = img.copy()
    else:
        line_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    if lines is None:
        return line_img

    for index, line in enumerate(lines):
        rho, theta = line
        color = (0, 255, 0)

        if labels is not None:
            if labels[index] == 0:
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)

        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho

        pt1 = (int(x0 + line_length * (-b)), int(y0 + line_length * (a)))
        pt2 = (int(x0 - line_length * (-b)), int(y0 - line_length * (a)))

        cv2.line(line_img, pt1=pt1, pt2=pt2, color=color, thickness=4)

    return line_img


def draw_segments_on_image(segments, img, labels=None, use_original_image=False):

    img_height, img_width = img.shape[:2]

    if use_original_image:
        segment_img = img.copy()
    else:
        segment_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

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


def draw_segments_extended_on_image(
    segments, img, labels=None, use_original_image=False
):
    if segments is not None:
        img_height, img_width = img.shape[:2]

        if use_original_image:
            segment_img = img.copy()
        else:
            segment_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

        for index, segment in enumerate(segments):
            x1, y1, x2, y2 = segment

            if x1 != x2:
                m = (y2 - y1) / (x2 - x1)
                b = y1 - m * x1

                x1_ext = 0
                y1_ext = int(m * x1_ext + b)

                x2_ext = img_width
                y2_ext = int(m * x2_ext + b)
            else:
                x1_ext, y1_ext = x1, 0
                x2_ext, y2_ext = x1, img_height

            color = (0, 255, 0)

            if labels is not None:
                if labels[index] == 0:
                    color = (255, 0, 0)
                else:
                    color = (0, 0, 255)

            cv2.line(
                segment_img,
                pt1=(x1_ext, y1_ext),
                pt2=(x2_ext, y2_ext),
                color=color,
                thickness=4,
            )

        return segment_img


def debug_line_angles(angles, labels=None):
    if labels is None:
        colors = None
    else:
        colors = ["red" if label == 0 else "blue" for label in labels]

    plt.scatter(angles, np.zeros_like(angles), c=colors)
    plt.xlabel("Angle (radians)")
    plt.yticks([])
    plt.show()


def calculate_line_coefficients_from_segments(segments):
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


def calculate_normalized_segment_vectors(segments):
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


def plot_vectors(vectors, labels):
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


def cluster_lines_kmeans(features):
    kmeans = KMeans(n_clusters=2)
    return kmeans.fit(features).labels_


def cluster_lines_hierarchical_agglomerative(features):
    model = AgglomerativeClustering(
        n_clusters=2,
        metric="euclidean",
        linkage="ward",  # as alternative also "average" is fine
    )

    return model.fit_predict(features)


def compute_with_hough_p(img):
    # debug_image_cv2(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # debug_image_cv2(gray)

    blurred = cv2.GaussianBlur(
        gray,
        ksize=(15, 15),  # kernel size -> since the image is big this must be big
        sigmaX=3,  # horizontal blur intensity
        sigmaY=3,  # vertical blur intensity
    )
    # debug_image_cv2(blurred)

    edges = cv2.Canny(
        blurred,
        threshold1=100,  # lower hysteresis threshold
        threshold2=150,  # upper hysteresis threshold,
        apertureSize=3,  # sobel kernel size
        L2gradient=True,  # false = l1 norm (faster), true = l2 norm (more accurate)
    )
    # debug_image_cv2(edges)

    hough_segments = cv2.HoughLinesP(  # Probabilistic Hough Transform (PHT)
        edges,
        rho=1,  # 1 pixel resolution -> this is usually fine like this
        theta=math.pi
        / 180,  # 1 degree resolution (in radiants) -> this is usually fine like this
        threshold=100,  # lower = detect more lines (including noise), higher = keep only clear lines
        minLineLength=100,  # lower = detect short and fragmented lines, higher = discard short and fragmented lines
        maxLineGap=10,  # lower = close lines are considered separated, higher = close lines are considered the same (merged)
    )

    segments = np.array(hough_segments).squeeze()
    segments_vectors = calculate_normalized_segment_vectors(segments)
    angles_radiants = np.arctan2(segments_vectors[:, 1], segments_vectors[:, 0])

    doubled_angles_radiants = angles_radiants * 2
    features = np.stack(
        (np.cos(doubled_angles_radiants), np.sin(doubled_angles_radiants)), axis=1
    )
    labels = cluster_lines_kmeans(features)

    # debug_line_angles(angles_radiants, labels)
    # plot_vectors(segments_vectors, labels)

    image_with_segments = draw_segments_extended_on_image(segments, img, labels, True)
    debug_image_cv2(image_with_segments)


def compute_with_hough(img):
    # debug_image_cv2(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # debug_image_cv2(gray)

    blurred = cv2.GaussianBlur(
        gray,
        ksize=(15, 15),  # kernel size -> since the image is big this must be big
        sigmaX=3,  # horizontal blur intensity
        sigmaY=3,  # vertical blur intensity
    )
    # debug_image_cv2(blurred)

    edges = cv2.Canny(
        blurred,
        threshold1=100,  # lower hysteresis threshold
        threshold2=150,  # upper hysteresis threshold,
        apertureSize=3,  # sobel kernel size
        L2gradient=True,  # false = l1 norm (faster), true = l2 norm (more accurate)
    )
    # debug_image_cv2(edges)

    hough_lines = cv2.HoughLines(
        edges,
        rho=1,  # 1 pixel resolution -> this is usually fine like this
        theta=math.pi
        / 180,  # 1 degree resolution (in radiants) -> this is usually fine like this
        threshold=100,  # lower = detect more lines (including noise), higher = keep only clear lines
    )

    lines = np.array(hough_lines).squeeze()

    angles_radiants = lines[:, 1]
    doubled_angles_radiants = angles_radiants * 2
    features = np.stack(
        (np.cos(doubled_angles_radiants), np.sin(doubled_angles_radiants)), axis=1
    )

    labels = cluster_lines_kmeans(features)
    # debug_line_angles(angles_radiants, labels)

    lines_on_image = draw_lines_on_image(lines, img, labels, True)
    debug_image_cv2(lines_on_image)


if __name__ == "__main__":
    image_path = os.path.abspath("data/chessred2k/images/0/G000_IMG005.jpg")
    img = cv2.imread(image_path)
    compute_with_hough_p(img)
    # compute_with_hough(img)
