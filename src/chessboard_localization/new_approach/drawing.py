import cv2
import pyautogui

COLOR_PALETTE = [
    (255, 0, 0),  # Blue
    (0, 0, 255),  # Red
    (0, 255, 0),  # Green
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 255),  # Purple
    (0, 128, 255),  # Orange-ish
    (255, 128, 0),  # Light Orange
    (0, 255, 128),  # Mint
]

def display_image_cv2(image: cv2.typing.MatLike, window_name, max_height=1024, max_width=1024):
    img_height, img_width = image.shape[:2]

    scale_width = max_width / img_width
    scale_height = max_height / img_height

    scale_factor = min(scale_width, scale_height)

    new_width = int(img_width * scale_factor)
    new_height = int(img_height * scale_factor)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, new_width, new_height)
    cv2.imshow(window_name, image)

    screen_width, screen_height = pyautogui.size()

    pos_x = (screen_width - new_width) // 2
    pos_y = (screen_height - new_height) // 2

    cv2.moveWindow(window_name, pos_x, pos_y)

    cv2.waitKey(0)
    cv2.destroyWindow(window_name)

def create_image_with_points(
    points, source_image, thickness=2, color=(0, 255, 0)
):
    dest_image = source_image.copy()

    for p in points:
        x, y = p
        x_pixel = int(x)
        y_pixel = int(y)

        cv2.circle(
            dest_image,
            (x_pixel, y_pixel),
            thickness,
            color,
            -1,
        )

    return dest_image

def create_image_with_segments(
    segments, source_image, thickness=2, segments_labels=None
):
    dest_image = source_image.copy()

    if segments is None:
        return dest_image

    color = COLOR_PALETTE[0]

    for index, segment in enumerate(segments):
        x1, y1, x2, y2 = segment

        if segments_labels is not None:
            label = segments_labels[index]
            color = COLOR_PALETTE[label % len(COLOR_PALETTE)]

        cv2.line(
            dest_image, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=thickness
        )

    return dest_image

def create_image_with_quads(quads, source_image, thickness=1, color=(0, 255, 0)):

    dest_image = source_image.copy()

    for quad in quads:
        quad_pts = quad.reshape((-1, 1, 2))
        cv2.polylines(
            dest_image, [quad_pts], isClosed=True, color=color, thickness=thickness
        )

    return dest_image