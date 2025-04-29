import os
import cv2
import math
import numpy as np
from visualization import debug_image_cv2

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
def draw_line_segments_extended(lines, img, use_original_image=False):
    if lines is not None:
        img_height, img_width = img.shape[:2]

        if use_original_image:
            line_img = img.copy()
        else:
            line_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

        for line in lines:
            x1, y1, x2, y2 = line[0]

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

            cv2.line(
                line_img,
                pt1=(x1_ext, y1_ext),
                pt2=(x2_ext, y2_ext),
                color=(0, 0, 255),
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

    return lines

if __name__ == "__main__":
    image_path = os.path.abspath("data/chessred2k/images/0/G000_IMG001.jpg")
    
    img = cv2.imread(image_path)
    lines = get_lines(img)
    print(f"Detected {len(lines)} lines")

    draw_line_segments_extended(lines, img, True)