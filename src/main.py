from pathlib import Path
import json
import numpy as np
import cv2
import pyautogui
import itertools
from ultralytics import YOLO
from chessboard_localization_temp.localization import (
    find_chessboard,
    find_chessboard_squares,
)

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


def get_real_chessboard_corners(file_path):
    file_name = Path(file_path).name

    with open("data/chessred2k/annotations.json") as file:
        parsed_file = json.load(file)

    img_info = next(
        img for img in parsed_file["images"] if img["file_name"] == file_name
    )
    img_id = img_info["id"]
    corner_annotations = [
        a for a in parsed_file["annotations"]["corners"] if a["image_id"] == img_id
    ]

    if len(corner_annotations) == 0:
        return None

    img_corner_info = corner_annotations[0]["corners"]

    corners = np.zeros((4, 2), dtype=np.float32)

    corners[0] = img_corner_info["top_left"]
    corners[1] = img_corner_info["top_right"]
    corners[2] = img_corner_info["bottom_right"]
    corners[3] = img_corner_info["bottom_left"]

    return corners


def get_chessboard_corners1(image):
    img_height, img_width = image.shape[:2]
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    upsize_factor = 1
    parameters_original_size = 1000
    scale_factor = img_width / parameters_original_size

    canny_threshold_1 = [25, 50, 100, 200, 300, 400]
    canny_threshold_2 = [50, 100, 200, 300, 400, 500]
    params = [x for x in itertools.product(canny_threshold_1, canny_threshold_2)]
    corners_list = find_chessboard(gray_image, params, upsize_factor, scale_factor)
    return corners_list


def get_chessboard_corners2(image):

    model = YOLO("models/board_localization.pt")
    res = model.predict(
        image,
        imgsz=640,
    )

    return res[0].keypoints.xy.squeeze().cpu().numpy()

def get_chessboard_cells(image, corners):

    # change corners to top-left, top-right, bottom-left, bottom-right
    src = np.float32([corners[0], corners[1], corners[3], corners[2]])

    dest_width = 1200
    dest_height = 1200
    dest = np.float32(
        [[0, 0], [dest_width, 0], [0, dest_height], [dest_width, dest_height]]
    )

    M = cv2.getPerspectiveTransform(src, dest)
    warped_image = cv2.warpPerspective(image, M, (dest_width + 2, dest_height + 2))

    # display_image_cv2(warped_image, "warped")

    rows = 8
    cols = 8
    square_width = dest_width // cols
    square_height = dest_height // rows

    for i in range(rows):
        for j in range(cols):
            top_left = (j * square_width, i * square_height)
            bottom_right = ((j + 1) * square_width, (i + 1) * square_height)
            cv2.rectangle(warped_image, top_left, bottom_right, (0, 255, 0), 4)

    display_image_cv2(warped_image, "grid")

    # M_inv = cv2.invert(M)[1]

    # TODO: return cells bounding boxes and center coordinates


def detect_chess_pieces(file_path):
    pass


def find_chess_piece_cell(pieces, cells):
    pass


def find_similar_games(mapping):
    pass


def order_corners_clockwise(corners):
    centroid = np.mean(corners, axis=0)
    vectors = corners - centroid
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    sorted_indices = np.argsort(-angles)
    return corners[sorted_indices]


def execute_pipeline(file_path):

    image = cv2.imread(file_path)
    # display_image_cv2(image, "image")

    corners = get_real_chessboard_corners(file_path)
    sorted_corners = order_corners_clockwise(corners)

    #predicted_corners = get_chessboard_corners1(image)
    predicted_corners = get_chessboard_corners2(image)

    sorted_predicted_corners = order_corners_clockwise(predicted_corners)

    distances = np.linalg.norm(sorted_corners - sorted_predicted_corners, axis=1)
    all_match = np.all(distances <= 10.0)

    for p in sorted_corners:
        cv2.circle(image, (int(p[0]), int(p[1])), 30, (0, 255, 0), -1)

    for p in sorted_predicted_corners:
        cv2.circle(image, (int(p[0]), int(p[1])), 30, (0, 0, 255), -1)

    display_image_cv2(image, "corners")

    return distances, all_match

    # cells = get_chessboard_cells(image, corners)

    # pieces = detect_chess_pieces(image)

    # mapping = find_chess_piece_cell(pieces, cells)

    # similar_games = find_similar_games(mapping)

    # TODO: show in UI


def get_next_image_in_folder(folder_path):
    rootdir = Path(folder_path)
    file_list = [f for f in rootdir.glob("**/*") if f.is_file()]
    for file_path in file_list:
        yield file_path


def main():
    paths = get_next_image_in_folder("data/chessred2k/images")
    count = 0
    errors = 0

    with open("results2.txt", "w") as f:
        for index, next_path in enumerate(paths):
            distances, all_match = execute_pipeline(str(next_path))
            
            '''
            count = count + 1
            f.write(
                f"image path: {next_path}, distances: {distances}, match: {all_match}\n"
            )

            if not all_match:
                errors = errors + 1

            print(f"Step {index}")

             '''

    print(f"Completed: {errors} errors over {count} images -> accuracy: {count-errors/count}")


if __name__ == "__main__":
    main()
