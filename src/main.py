import time
from pathlib import Path
import json
from typing import Callable

import numpy as np
import cv2
import pyautogui
import itertools
from ultralytics import YOLO
import math

#from chessboard_localization_temp.localization import (
#    find_chessboard,
#    find_chessboard_squares,
#)
#import chessboard_localization_temp.main as local_temp

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


def get_chessboard_corners2(image):
    model = YOLO("models/board_localization.pt")
    res = model.predict(
        image,
        imgsz=640,
    )
    return res[0].keypoints.xy.squeeze().cpu().numpy()

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


def contour_based(image):
    # read image and convert it to different color spaces 
    gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    original_size = 1000
    scale_factor = image.shape[0] / original_size

    ## Processing Image  -->  OTSU Threshold , Canny edge detection , dilate , HoughLinesP 

    # OTSU threshold
    ret, otsu_binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Canny edge detection
    canny_image = cv2.Canny(otsu_binary, 20, 255)

    # Dilation
    kernel = np.ones((7, 7), np.uint8)  
    dilation_image = cv2.dilate(canny_image, kernel, iterations=1)

    # Hough Lines
    lines = cv2.HoughLinesP(dilation_image, 1, np.pi / 180, threshold=500, minLineLength=150, maxLineGap=100)


    # Create an image that contains only black pixels
    black_image = np.zeros_like(dilation_image)

    # Draw only lines that are output of HoughLinesP function to the "black_image"
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # draw only lines to the "black_image"
            cv2.line(black_image, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # Dilation
    kernel = np.ones((3, 3), np.uint8)
    black_image = cv2.dilate(black_image, kernel, iterations=1)

    # Look for valid squares and check if squares are inside of board

    # find contours
    board_contours, hierarchy = cv2.findContours(black_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # blank image for displaying all contours
    #all_contours_image= np.zeros_like(black_image)

    # Copy blank image for displaying all squares 
    #squares_image = np.copy(image) 

    # blank image for displaying valid contours (squares)
    valid_squares_image = np.zeros_like(black_image)

    

    # loop through contours and filter them by deciding if they are potential squares
    for contour in board_contours:
        if scale_factor * 2000 < cv2.contourArea(contour) < scale_factor * 20000:

            # Approximate the contour to a simpler shape
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # if polygon has 4 vertices
            if len(approx) == 4:

                # 4 points of polygon
                pts = [pt[0].tolist() for pt in approx]

                # create same pattern for points , bottomright(1) , topright(2) , topleft(3) , bottomleft(4)
                index_sorted = sorted(pts, key=lambda x: x[0], reverse=True)

                #  Y values
                if index_sorted[0][1]< index_sorted[1][1]:
                    cur=index_sorted[0]
                    index_sorted[0] =  index_sorted[1]
                    index_sorted[1] = cur

                if index_sorted[2][1]> index_sorted[3][1]:
                    cur=index_sorted[2]
                    index_sorted[2] =  index_sorted[3]
                    index_sorted[3] = cur

                # bottomright(1) , topright(2) , topleft(3) , bottomleft(4)
                pt1=index_sorted[0]
                pt2=index_sorted[1]
                pt3=index_sorted[2]
                pt4=index_sorted[3]

                # find rectangle that fits 4 point 
                x, y, w, h = cv2.boundingRect(contour)
                # find center of rectangle 
                center_x=(x+(x+w))/2
                center_y=(y+(y+h))/2

                

                # calculate length of 4 side of rectangle
                l1 = math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
                l2 = math.sqrt((pt2[0] - pt3[0])**2 + (pt2[1] - pt3[1])**2)
                l3 = math.sqrt((pt3[0] - pt4[0])**2 + (pt3[1] - pt4[1])**2)
                l4 = math.sqrt((pt1[0] - pt4[0])**2 + (pt1[1] - pt4[1])**2)
    
    
                # Create a list of lengths
                lengths = [l1, l2, l3, l4]
                
                # Get the maximum and minimum lengths
                max_length = max(lengths)
                min_length = min(lengths)

                # Check if this length values are suitable for a square , this threshold value plays crucial role for squares ,  
                if (max_length - min_length) <= scale_factor * 35 : # 20 for smaller boards  , 50 for bigger , 35 works most of the time 
                    valid_square=True
                else:
                    valid_square=False
    
                if valid_square:

                    # Draw the lines between the points
                    #cv2.line(squares_image, pt1, pt2, (255, 255, 0), 7)
                    #cv2.line(squares_image, pt2, pt3, (255, 255, 0), 7)
                    #cv2.line(squares_image, pt3, pt4, (255, 255, 0), 7)
                    #cv2.line(squares_image, pt1, pt4, (255, 255, 0), 7)

                    # Draw only valid squares to "valid_squares_image"
                    cv2.line(valid_squares_image, pt1, pt2, (255, 255, 0), 7)
                    cv2.line(valid_squares_image, pt2, pt3, (255, 255, 0), 7)
                    cv2.line(valid_squares_image, pt3, pt4, (255, 255, 0), 7)
                    cv2.line(valid_squares_image, pt1, pt4, (255, 255, 0), 7)
                
                # Draw only valid squares to "valid_squares_image"
                #cv2.line(all_contours_image, pt1, pt2, (255, 255, 0), 7)
                #cv2.line(all_contours_image, pt2, pt3, (255, 255, 0), 7)
                #cv2.line(all_contours_image, pt3, pt4, (255, 255, 0), 7)
                #cv2.line(all_contours_image, pt1, pt4, (255, 255, 0), 7)
            

    #### Dilation to the image that contains only valid squares (gemoetrically valid)

    # Apply dilation to the valid_squares_image
    kernel = np.ones((7, 7), np.uint8)
    dilated_valid_squares_image = cv2.dilate(valid_squares_image, kernel, iterations=1)


    #### Find biggest contour of image 

    # Find contours of dilated_valid_squares_image
    contours, _ = cv2.findContours(dilated_valid_squares_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # take biggest contour 
    largest_contour = max(contours, key=cv2.contourArea)

    # create black image
    #biggest_area_image = np.zeros_like(dilated_valid_squares_image)

    # draw biggest contour to the image
    #cv2.drawContours(biggest_area_image,largest_contour,-1,(255,255,255),10)

    #### Find 4 extreme point of chess board

    # Initialize variables to store extreme points
    top_left = None
    top_right = None
    bottom_left = None
    bottom_right = None

    # Loop through the contour to find extreme points
    for point in largest_contour[:, 0]:
        x, y = point

        if top_left is None or (x + y < top_left[0] + top_left[1]):
            top_left = (x, y)

        if top_right is None or (x - y > top_right[0] - top_right[1]):
            top_right = (x, y)

        if bottom_left is None or (x - y < bottom_left[0] - bottom_left[1]):
            bottom_left = (x, y)

        if bottom_right is None or (x + y > bottom_right[0] + bottom_right[1]):
            bottom_right = (x, y)

    # Draw the contour and the extreme points
    #extreme_points_image = np.zeros_like(dilated_valid_squares_image, dtype=np.uint8)
    #extreme_points_image = image
    #cv2.drawContours(extreme_points_image, [largest_contour], -1, (255, 255, 255), thickness=2)

    # Mark the extreme points
    # Mark the extreme points
    #cv2.circle(extreme_points_image, top_left, 15, (255, 0, 255), -1)  # red for top-left
    #cv2.circle(extreme_points_image, top_right, 15, (255, 0, 255), -1)  # green for top-right
    #cv2.circle(extreme_points_image, bottom_left, 15, (255, 0,255), -1)  # blue for bottom-left
    #cv2.circle(extreme_points_image, bottom_right, 15, (255, 0, 255), -1)  # yellow for bottom-right

    #display_image_cv2(extreme_points_image, window_name="test")

    return np.array([top_left, top_right, bottom_right, bottom_left])

def main():

    p = load_real_chessboard_corners(r"D:\workspace\Chessy3D\data\chessred2k\annotations.json")
    base_path = r"D:\\workspace\\Chessy3D\\data\\chessred2k\\"
    debug = False
    th = 30

    errors = 0
    start = time.time()
    with open("deep_learning_method.txt", "wt") as f:
        for index, [imgid, info] in enumerate(p.items()):
            start_iter = time.time()
            path = base_path + info[1]
            distances, all_match = execute_pipeline(path, imgid, contour_based)
            # distances, all_match = execute_pipeline(path, imgid, get_chessboard_corners1)
            #distances, all_match = execute_pipeline(path, imgid, get_chessboard_corners_topdown, debug=debug, th=th)

            end_iter = time.time()
            f.write(f"image path: {info[0]}, distances: {distances}, match: {all_match}, time: {end_iter - start_iter}\n")
            if not all_match:
                errors = errors + 1
            print(f"STEP {index} image path: {info[0]}, distances: {distances}, match: {all_match}, error: {errors/(index+1)}. time: {end_iter - start_iter}")

        end = time.time()
        f.write(f"completed in {end - start} seconds")

    count = len(p)
    print(f"Completed: {errors} errors over {count} images -> accuracy: {(count-errors)/count} - time: {end - start} seconds avg {(end - start)/count}")


if __name__ == "__main__":
    main()
