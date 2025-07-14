import time
from pathlib import Path
import json
from typing import Callable

import numpy as np
import cv2
import pyautogui
import itertools
from ultralytics import YOLO
from chessboard_localization_temp.localization import (
    find_chessboard,
    find_chessboard_squares,
)
import chessboard_localization_temp.main as local_temp
import concurrent.futures

####################


def display_image_cv2(
    image: cv2.typing.MatLike, window_name, max_height=1024, max_width=1024
):
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


###################

# <filename>: corners
parsed: dict[str, tuple[str, str, np.typing.ArrayLike]]|None = {}
def load_real_chessboard_corners(annotation_path: str):
    with open(annotation_path) as file:
        parsed_file = json.load(file)

        # filename, path, id
        images_info = {}

        for img in parsed_file["images"]:
            images_info[img["id"]] = (img["id"], img["file_name"], img["path"])

        for img in parsed_file["annotations"]["corners"]:
            if img["image_id"] in images_info:
                info = images_info[img["image_id"]]
                img_corner_info = img["corners"]

                parsed[info[0]] = (
                    info[1],
                    info[2],
                    np.array([
                        img_corner_info["top_left"],
                        img_corner_info["top_right"],
                        img_corner_info["bottom_right"],
                        img_corner_info["bottom_left"]
                        ], dtype=np.float32
                    )
                )

        print(f"Loaded {len(parsed)} images")
        return parsed


def get_real_chessboard_corners(imgid: str):
    return parsed[imgid][2]

## METHODS

def get_chessboard_corners1(image):
    img_height, img_width = image.shape[:2]
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    upsize_factor = 1
    parameters_original_size = 1000
    scale_factor = img_width / parameters_original_size

    canny_threshold_1 = [25, 50, 100, 200, 300, 400]
    canny_threshold_2 = [50, 100, 200, 300, 400, 500]
    params = [x for x in itertools.product(canny_threshold_1, canny_threshold_2)]
    corners_list, _, _, _, _ = find_chessboard(gray_image, params, upsize_factor, scale_factor)
    return corners_list


# def get_chessboard_corners2(image):
#
#     model = YOLO("models/board_localization.pt")
#     res = model.predict(
#         image,
#         imgsz=640,
#     )
#
#     return res[0].keypoints.xy.squeeze().cpu().numpy()

def get_chessboard_corners_topdown(image):
    resized_image = cv2.resize(image, (1500, 1500))
    _, corners_list, _, _, _, _, _, _, _, _ = local_temp.auto_chessboard_localization_alt(image, resized_image)

    # riporta in scala originale
    if corners_list is None or len(corners_list) == 0:
        return None

    corners_list[:, 0] = (image.shape[0] / resized_image.shape[0]) * corners_list[:, 0]
    corners_list[:, 1] = (image.shape[1] / resized_image.shape[1]) * corners_list[:, 1]
    return corners_list


def order_corners_clockwise(corners):
    centroid = np.mean(corners, axis=0)
    vectors = corners - centroid
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    sorted_indices = np.argsort(-angles)
    return corners[sorted_indices]


def execute_pipeline(file_path: str, imgid: str, func: Callable[[cv2.Mat], np.typing.ArrayLike], debug=False, th=30):
    image = cv2.imread(file_path)

    corners = get_real_chessboard_corners(imgid)
    sorted_corners = order_corners_clockwise(corners)
    # sorted_corners = local_temp.sort_quadrilateral_approx(corners)
    # sorted_corners = np.array(sorted_corners)

    predicted_corners = func(image)
    if predicted_corners is None or len(predicted_corners) == 0:
        predicted_corners = np.zeros((4,2), dtype=float)
    sorted_predicted_corners = order_corners_clockwise(predicted_corners)
    # sorted_predicted_corners = local_temp.sort_quadrilateral_approx(predicted_corners)
    # sorted_predicted_corners = np.array(sorted_predicted_corners)

    distances = np.linalg.norm(sorted_corners - sorted_predicted_corners, axis=1)
    all_match = np.all(distances <= th) # entro 30 pixel é comunque funzionale al nostro scopo

    if debug:
        for p in sorted_corners:
            cv2.circle(image, (int(p[0]), int(p[1])), 30, (0, 255, 0), -1)

        for p in sorted_predicted_corners:
            cv2.circle(image, (int(p[0]), int(p[1])), 30, (0, 0, 255), -1)
        display_image_cv2(image, file_path)

    return distances, all_match

def get_next_image_in_folder(folder_path):
    rootdir = Path(folder_path)
    file_list = [f for f in rootdir.glob("**/*") if f.is_file()]
    for file_path in file_list:
        yield file_path


def main():
    p = load_real_chessboard_corners(r"E:\projects\uni\Chessy3D\data\chessred\annotations.json")
    base_path = r"E:/projects/uni/Chessy3D/data/chessred/"
    debug = False
    th = 30

    errors = 0
    start = time.time()
    with open("result.txt", "wt") as f:
        for index, [imgid, info] in enumerate(p.items()):
            start_iter = time.time()
            path = base_path + info[1]
            # distances, all_match = execute_pipeline(path, imgid, get_chessboard_corners1)
            distances, all_match = execute_pipeline(path, imgid, get_chessboard_corners_topdown, debug=debug, th=th)

            end_iter = time.time()
            f.write(f"image path: {info[0]}, distances: {distances}, match: {all_match}, time: {end_iter - start_iter}\n")
            if not all_match:
                errors = errors + 1
            print(f"STEP {index} image path: {info[0]}, distances: {distances}, match: {all_match}, error: {errors/(index+1)}. time: {end_iter - start_iter}")

        end = time.time()
        f.write(f"completed in {end - start} seconds")

    count = len(p)
    print(f"Completed: {errors} errors over {count} images -> accuracy: {count-errors/count} - time: {end - start} seconds avg {(end - start)/count}")


if __name__ == "__main__":
    main()
