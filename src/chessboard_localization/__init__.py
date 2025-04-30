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

def debug_clustered_lines(angles, labels):
    colors = ["red" if label == 0 else "blue" for label in labels]
    plt.scatter(angles, np.zeros_like(angles), c=colors)
    plt.xlabel("Absolute Angle (radians)")
    plt.yticks([])
    plt.show()

# draw the lines as they are (segments)
def draw_line_segments(lines, img, use_original_image=False):
    if lines is not None:
        img_height, img_width = img.shape[:2]

        if use_original_image:
            line_img = img.copy()
        else:
            line_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(
                line_img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 0, 255), thickness=4
            )

        debug_image_cv2(line_img)

# calculates the slope and the intercepts of each line then extend them to the boundaries of the image
def draw_line_segments_extended(lines, img, labels=None, use_original_image=False):
    if lines is not None:
        img_height, img_width = img.shape[:2]

        if use_original_image:
            line_img = img.copy()
        else:
            line_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

        for index, line in enumerate(lines):
            x1, y1, x2, y2 = line

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
                line_img,
                pt1=(x1_ext, y1_ext),
                pt2=(x2_ext, y2_ext),
                color=color,
                thickness=4,
            )

        debug_image_cv2(line_img)

# gets lines from an image
def get_lines(img):
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
        threshold2=200,  # upper hysteresis threshold,
        apertureSize=3,  # sobel kernel size
        L2gradient=True,  # false = l1 norm (faster), true = l2 norm (more accurate)
    )
    # debug_image_cv2(edges)

    lines = cv2.HoughLinesP(  # Probabilistic Hough Transform (PHT)
        edges,
        rho=1,  # 1 pixel resolution -> this is usually fine like this
        theta=math.pi
        / 180,  # 1 degree resolution (in radiants) -> this is usually fine like this
        threshold=100,  # lower = detect more lines (including noise), higher = keep only clear lines
        minLineLength=100,  # lower = detect short and fragmented lines, higher = discard short and fragmented lines
        maxLineGap=10,  # lower = close lines are considered separated, higher = close lines are considered the same (merged)
    )

    return np.array(lines).squeeze()

# History:
#   First attempt: cv2.findChessboardCorners
#   Second attempt: cv2.Canny + cv2.HoughLinesP + Hierarchical clustering (AgglomerativeClustering) using slopes as features
#   Third attempt: cv2.Canny + cv2.HoughLinesP + Hierarchical clustering (AgglomerativeClustering) using abs of arctan angles as features

if __name__ == "__main__":
    image_path = os.path.abspath("data/chessred2k/images/0/G000_IMG000.jpg")

    img = cv2.imread(image_path)
    lines = get_lines(img)

    #print(f"Detected {len(lines)} lines")

    # draw_line_segments_extended(lines, img, None, True)

    model = AgglomerativeClustering(
        n_clusters=2,
        metric="euclidean",
        linkage="ward",  # as alternative also "average" is fine
    )

    # x1, y1, x2, y2  -> (y2 - y1) / (x2 - x1) -> slope
    y_diff = lines[:, 3] - lines[:, 1]
    x_diff = lines[:, 2] - lines[:, 0]

    # arctan is used as a surrogate of the slope to avoid division by infinity
    angles = np.abs(np.arctan2(y_diff, x_diff)).reshape(-1, 1)
    labels = model.fit_predict(angles)

    # TODO: try this on the entire dataset to see if the clustering is consistent
    # debug_clustered_lines(angles, labels)

    draw_line_segments_extended(lines, img, labels, True)

