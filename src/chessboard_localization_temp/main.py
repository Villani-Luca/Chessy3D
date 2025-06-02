import cv2
import itertools
from src.chessboard_localization_temp.localization import draw_chessboard_squares, find_chessboard, \
    find_chessboard_squares, canny_multiple, hough_multiple, find_best_squares_hough, draw_chessboard_corners


# Path of Image that you want to convert
def chessboard_localization_resize(image_path):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (3000, 3000))

    return  image, resized_image

def chessboard_best_hough(image, resized_image):
    original_size = image.shape[0]
    resized_size = 3000
    parameters_original_size = 1000
    upsize_factor = resized_size / original_size
    scale_factor = resized_size / parameters_original_size

    canny_threshold_1 = [25, 50, 100, 200, 300, 400]
    canny_threshold_2 = [50, 100, 200, 300, 400, 500]
    canny_params = [x for x in itertools.product(canny_threshold_1, canny_threshold_2)]

    canny_images = canny_multiple(resized_image, canny_params, upsize_factor)
    hough_images = hough_multiple(canny_images)
    best_valid_square_image, _, _ = find_best_squares_hough(hough_images, scale_factor)
    return best_valid_square_image

def auto_chessboard_localization(image, resized_image):
    gray_image=cv2.cvtColor(resized_image,cv2.COLOR_BGR2GRAY)

    # parameters used for tuning other parameters based on image resolution
    original_size = image.shape[0]
    resized_size = 3000
    parameters_original_size = 1000
    upsize_factor = resized_size / original_size
    scale_factor = resized_size / parameters_original_size 

    canny_threshold_1 = [25, 50, 100, 200, 300, 400]
    canny_threshold_2 = [50, 100, 200, 300, 400, 500]
    params = [x for x in itertools.product(canny_threshold_1, canny_threshold_2)]

    corners_list, best_canny, best_hough, polygons_image, best_idx = find_chessboard(gray_image, params, upsize_factor, scale_factor)
    squares_data_original = find_chessboard_squares(corners_list)
    polygons_image = draw_chessboard_corners(corners_list, cv2.cvtColor(polygons_image, cv2.COLOR_GRAY2BGR))
    print("best params: ", params[best_idx])

    # Display the result
    resized_image_copy = resized_image.copy()
    draw_chessboard_squares(resized_image, squares_data_original, corners_list)
    return resized_image, corners_list, squares_data_original, resized_image_copy, best_canny, best_hough, polygons_image