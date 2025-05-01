import os
import cv2
import math
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import pyautogui

import matplotlib.pyplot as plt

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

def draw_lines_on_image(lines, img, labels=None, use_original_image=False, line_length=10000):

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

def draw_segments_extended_on_image(segments, img, labels=None, use_original_image=False):
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

def debug_line_angles(angles, labels = None):
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
  
    line_coefficients = calculate_line_coefficients_from_segments(segments)

    model = AgglomerativeClustering(
        n_clusters=2,
        metric="cosine",
        linkage="complete",  # as alternative also "average" is fine
    )

    # m = tan(angle) -> angle = arctan(m)
    # these angles are between [-π, π]
    angles_radiants = np.arctan(line_coefficients[:,0]) 

    # angles folded between [0, π]
    #folding_mask = angles_radiants < 0
    #angles_radiants[folding_mask] = math.pi + angles_radiants[folding_mask]

    #debug_line_angles(angles_radiants)

    # If we don't consider the direction but only the actual angle, then lines with angle π are equivalent to lines with angle 0.
    # To make sure these two numbers are close, to help the clustering classify the lines correctly, all lines with angle
    # above a certain threshold (140 degress) are flipped to the left quadrant (π-angle).
    # This gives also a bit of margin in the classification
    #equivalence_mask = angles_radiants > math.radians(140)
    #angles_radiants[equivalence_mask] = math.pi - angles_radiants[equivalence_mask]

    # debug_line_angles(angles_radiants)

    labels = model.fit_predict(angles_radiants.reshape(-1, 1))

    # TODO: try this on the entire dataset to see if the clustering is consistent
    # debug_clustered_lines(angles_radiants, labels)

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

    model = AgglomerativeClustering(
        n_clusters=2,
        metric="euclidean",
        linkage="ward",  # as alternative also "average" is fine
    )
   
    # These angles are between [0, π]
    # The copy is necessary because of the alterations of the angle are done just for the clustering.
    angles_radiants = lines[:, 1].copy() 

    # debug_line_angles(angles_radiants)

    # If we don't consider the direction but only the actual angle, then lines with angle π are equivalent to lines with angle 0.
    # To make sure these two numbers are close, to help the clustering classify the lines correctly, all lines with angle
    # above a certain threshold (140 degress) are flipped to the left quadrant (π-angle).
    # This is an heuristic and may not work in all cases.
    equivalence_mask = angles_radiants > math.radians(140)
    angles_radiants[equivalence_mask] = math.pi - angles_radiants[equivalence_mask]

    #debug_line_angles(angles_radiants)

    labels = model.fit_predict(angles_radiants.reshape(-1, 1))
    lines_on_image = draw_lines_on_image(lines, img, labels, True)
    debug_image_cv2(lines_on_image)

if __name__ == "__main__":
    image_path = os.path.abspath("data/chessred2k/images/0/G000_IMG020.jpg")
    img = cv2.imread(image_path)
    #compute_with_hough_p(img)
    compute_with_hough(img)

# G000_IMG001
# G000_IMG005
# G000_IMG006
# G000_IMG013
# G000_IMG015