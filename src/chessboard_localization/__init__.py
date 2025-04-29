import os
import cv2
import math
import pyautogui
from matplotlib import pyplot as plt


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


def debug_image_matplot(img, color=False):

    plt.figure(figsize=(20, 10))
    plt.title("Debug")
    plt.axis("off")

    if color:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb)
    else:
        plt.imshow(img, cmap="gray")

    plt.show()


def get_lines(image_path, canny_low_th=100, canny_high_th=200):
    img = cv2.imread(image_path)
    # debug_image_cv2(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # debug_image_cv2(gray)

    blurred = cv2.GaussianBlur(gray, ksize=(15, 15), sigmaX=3, sigmaY=3)
    # debug_image_cv2(blurred)

    edges = cv2.Canny(blurred, threshold1=canny_low_th, threshold2=canny_high_th)
    # debug_image_cv2(edges)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=math.pi / 180,
        threshold=100,
        minLineLength=100,
        maxLineGap=10,
    )

    if lines is not None:

        line_img = img.copy()

        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(
                line_img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=5
            )

        debug_image_cv2(line_img)

    return lines


if __name__ == "__main__":
    image_path = os.path.abspath("data/chessred2k/images/0/G000_IMG001.jpg")
    lines = get_lines(image_path)
    print(lines)
