import math
import cv2
import itertools

import numpy as np
from src.chessboard_localization_temp.localization import draw_chessboard_squares, find_best_chessboard_polygon, find_best_fitting_enclosing_square, find_chessboard, \
    find_chessboard_squares, canny_multiple, hough_multiple, find_best_squares_hough, draw_chessboard_corners, process_contours, sort_quadrilateral_approx


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


def auto_chessboard_localization_alt(image, resized_image):
    gray_image=cv2.cvtColor(resized_image,cv2.COLOR_BGR2GRAY)

    # parameters used for tuning other parameters based on image resolution
    original_size = image.shape[0]
    resized_size = 3000
    parameters_original_size = 1000
    upsize_factor = resized_size / original_size
    scale_factor = resized_size / parameters_original_size 

    blurred_image = cv2.bilateralFilter(gray_image,9,75,75)

    thresh, _ = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    canny_image = cv2.Canny(blurred_image, thresh, thresh * 0.5)
    canny_image = cv2.dilate(canny_image, np.ones((5,5)), iterations=2)

    ############## find candidates contours ####################
    board_contours, hierarchy = cv2.findContours(canny_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    no_parent_contours = [x for x in enumerate(hierarchy[0]) if x[1][3] == -1]
    candidates = []
    for i, _ in no_parent_contours:
        output = process_contours(i, hierarchy[0], board_contours, min=50_000*scale_factor, max=1_400_000*scale_factor)

        for i,cnt in output:
            approx = cv2.approxPolyN(cnt, 4, True)
            if approx.shape[0] != 1:
                continue

            approx = sort_quadrilateral_approx(approx[0])
            is_duplicate = False
            for candidate in candidates:
                is_duplicate = (
                    np.allclose(approx[0], candidate[0])
                    and np.allclose(approx[1], candidate[1])
                    and np.allclose(approx[2], candidate[2])
                    and np.allclose(approx[3], candidate[3])
                )
                if is_duplicate:
                    break
            
            if is_duplicate:
                continue
            
            candidates.append(approx)


    ########## FIND CORNERS ###############
    corners_list = [np.float32([candidate[2], candidate[1], candidate[3], candidate[0]]) for candidate in candidates]
    threshold = 0
    pwidth, pheight = (1200, 1200) # projection width and height
    dst_pts = np.float32([
        [threshold, threshold], 
        [pwidth + threshold, threshold], 
        [threshold, pheight + threshold], 
        [pwidth + threshold, pheight + threshold]
    ])
    hough_angle_tolerance = np.pi / 18

    final_candidates = []
    for candidate_idx, corners in enumerate(corners_list):
        # Compute the perspective transform matrix
        M = cv2.getPerspectiveTransform(corners, dst_pts)
        M_inv = cv2.invert(M)[1]  # Get the inverse of the perspective matrix

        working_image = cv2.warpPerspective(canny_image, M, (pwidth, pheight), flags=cv2.INTER_LINEAR)
        lines = cv2.HoughLinesP(working_image, 1, np.pi / 180, threshold=500, minLineLength=pwidth*0.1, maxLineGap=pwidth*0.5)
        hough_image = np.zeros_like(working_image)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # draw only lines to the "black_image"
                angle_radians = np.arctan2(y2 - y1, x2 - x1)  
                normalized_angle = abs(angle_radians % np.pi)  # Normalize angle to [0, π]
                if (normalized_angle <= hough_angle_tolerance or abs(normalized_angle - np.pi) <= hough_angle_tolerance) or abs(normalized_angle - np.pi / 2) <= hough_angle_tolerance:
                    cv2.line(hough_image, (x1, y1), (x2, y2), 255, 3)

        path_scale_factor = hough_image.shape[0] / parameters_original_size
        best_valid_square_image, best_squares_number,_,_ = find_best_squares_hough([hough_image], path_scale_factor)
        corners = find_best_fitting_enclosing_square(best_valid_square_image)
        if corners is not None:
            #chessboard_approx = cv2.approxPolyN(max_external_contours, 4, False, ensure_convex=True)
            corners = np.array([(corners[2], corners[1], corners[3], corners[0])], dtype=np.float32)
            M_inv = cv2.invert(M)[1]
            corners_original = cv2.perspectiveTransform(corners, M_inv)

            # print(corners_original.shape, corners_original[0].shape, corners_original)
            # corners_original_draw = corners_original.astype(int)
            # corners_original_draw = [corners_original_draw[0][0], corners_original_draw[0][1], corners_original_draw[0][2], corners_original_draw[0][3]]

            final_candidates.append((corners_original, best_squares_number))

    if len(final_candidates) > 0:
        corners_list = find_best_chessboard_polygon([(x[0].reshape(4, 2), x[1]) for x in final_candidates], resized_image.shape[1], resized_image.shape[0])
        squares_data = find_chessboard_squares(corners_list.reshape(1,4,2))

        # Display the result
        output_with_corners = resized_image.copy()
        draw_chessboard_squares(output_with_corners, squares_data, corners_list.astype(int))

        return output_with_corners, corners_list, squares_data, resized_image, canny_image, resized_image, resized_image
    else:
        return resized_image, [], [], resized_image, canny_image, resized_image, resized_image