import pyautogui
from matplotlib import pyplot as plt
import cv2

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

