import cv2
import numpy as np
import math
from scipy.stats import zscore
import concurrent.futures

canny_dilation_kernel = np.ones((7, 7), np.uint8)
def canny_multiple(img: cv2.typing.MatLike, params: list[tuple[int, int]], upsize_factor: float):
    canny_images = []

    blur_sigma = math.sqrt((upsize_factor // 2)*(5**2))
    blurred_image = cv2.GaussianBlur(img, (5,5),blur_sigma)
    for idx, (t1, t2) in enumerate(params):
        can = cv2.Canny(blurred_image, t1, t2) # é stato modificato questo input per creare le varie immagini
        can = cv2.dilate(can, canny_dilation_kernel, iterations=1)
        canny_images.append(can)

    return canny_images

def __hough_single(img: cv2.typing.MatLike):
    hough_dilation_kernel = np.ones((3, 3), np.uint8)
    lines = cv2.HoughLinesP(img, 1, np.pi / 180, threshold=500, minLineLength=150, maxLineGap=100)

    # Create an image that contains only black pixels
    black_image = np.zeros_like(img)

    # Draw only lines that are output of HoughLinesP function to the "black_image"
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # draw only lines to the "black_image"
            cv2.line(black_image, (x1, y1), (x2, y2), (255, 255, 255), 3)

    # Dilation
    cv2.dilate(black_image, hough_dilation_kernel, iterations=1)
    return black_image

def hough_multiple(images: list[cv2.typing.MatLike]):
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        return [x for x in executor.map(__hough_single, images, chunksize=10)]

# TEMP
# def hough_multiple(images: list[cv2.typing.MatLike]):
#     hough_images = []
#     for can in images:
#         # Hough Lines
#         lines = cv2.HoughLinesP(can, 1, np.pi / 180, threshold=500, minLineLength=150, maxLineGap=100)
#
#         # Create an image that contains only black pixels
#         black_image = np.zeros_like(can)
#
#         # Draw only lines that are output of HoughLinesP function to the "black_image"
#         if lines is not None:
#             for line in lines:
#                 x1, y1, x2, y2 = line[0]
#                 # draw only lines to the "black_image"
#                 cv2.line(black_image, (x1, y1), (x2, y2), (255, 255, 255), 3)
#
#         # Dilation
#         black_image = cv2.dilate(black_image, hough_dilation_kernel, iterations=1)
#         hough_images.append(black_image)
#
#     return hough_images

def sort_quadrilateral_approx(approx):
    '''
    returns (bottomright(1), topright(2) , topleft(3) , bottomleft(4))
    '''

    # create same pattern for points , bottomright(1) , topright(2) , topleft(3) , bottomleft(4)
    index_sorted = sorted(approx, key=lambda x: x[0], reverse=True)
    #  Y values
    if index_sorted[0][1]< index_sorted[1][1]:
        index_sorted[0], index_sorted[1] = index_sorted[1], index_sorted[0]

    if index_sorted[2][1]> index_sorted[3][1]:
        index_sorted[2], index_sorted[3] = index_sorted[3], index_sorted[2]

    # bottomright(1) , topright(2) , topleft(3) , bottomleft(4)
    return (index_sorted[0], index_sorted[1], index_sorted[2], index_sorted[3])

def find_best_squares_hough(images: list[cv2.typing.MatLike], scale_factor: float):
    best_valid_square_image = None
    best_valid_square_image_square_count = 0
    best_square_approx = []


    for black_image in images:
        current_valid_square_count = 0

        # Look for valid squares and check if squares are inside of board
        board_contours = []
        square_approx = []
        valid_squares_image = np.zeros_like(black_image)

        # find contours
        board_contours, _ = cv2.findContours(black_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # loop through contours and filter them by deciding if they are potential squares
        for contour in board_contours:
            if 4000*scale_factor < cv2.contourArea(contour) < 20000*scale_factor:
                # Approximate the contour to a simpler shape
                epsilon = 0.01 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                if len(approx) == 4:
                    # 4 points of polygon
                    pts = [pt[0].tolist() for pt in approx]
                    (pt1, pt2, pt3, pt4) = sort_quadrilateral_approx(pts)

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
                    if (max_length - min_length) <= 50 * scale_factor: # 20 for smaller boards  , 50 for bigger , 35 works most of the time 
                        valid_square=True
                    else:
                        valid_square=False

                    if valid_square:
                        current_valid_square_count += 1 

                        square_approx.append(approx)

                        # Draw only valid squares to "valid_squares_image"
                        cv2.line(valid_squares_image, pt1, pt2, (255, 255, 0), 7)
                        cv2.line(valid_squares_image, pt2, pt3, (255, 255, 0), 7)
                        cv2.line(valid_squares_image, pt3, pt4, (255, 255, 0), 7)
                        cv2.line(valid_squares_image, pt1, pt4, (255, 255, 0), 7)

        if current_valid_square_count >= best_valid_square_image_square_count:
            best_valid_square_image = valid_squares_image
            best_valid_square_image_square_count = current_valid_square_count
            best_square_approx = square_approx

    return (
        best_valid_square_image,
        best_valid_square_image_square_count,
        best_square_approx
    )

def extract_features(approx):
        # Centroid
        M = cv2.moments(approx)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Size (area or bounding box)
        area = cv2.contourArea(approx)
        rect = cv2.minAreaRect(approx)
        width, height = rect[1]
        angle = rect[2]  # Angle of rotation

        # Normalize angle if necessary
        if width < height:
            angle = angle + 90

        return {
            "centroid": (cx, cy),
            "area": area,
            "angle": angle,
            "size": max(width, height) * min(width, height)  # approximate area
        }

def filter_valid_polygons(approx):
    # features is a list of dicts from extract_features()
    features = [extract_features(x) for x in approx]
    centroids = np.array([f["centroid"] for f in features])
    areas = np.array([f["area"] for f in features])
    angles = np.array([f["angle"] for f in features])
    sizes = np.array([f["size"] for f in features])

    # Use z-score for outlier detection
    area_z = np.abs(zscore(areas))
    angle_z = np.abs(zscore(angles))
    size_z = np.abs(zscore(sizes))

    # Distance of centroids from cluster center
    center = np.mean(centroids, axis=0)
    dists = np.linalg.norm(centroids - center, axis=1)
    dist_z = np.abs(zscore(dists))

    # Define threshold (commonly 2 or 3)
    threshold = 3

    # valid_indices = np.where((area_z < threshold) &
    #                          (angle_z < threshold) &
    #                          (size_z < threshold) &
    #                          (dist_z < threshold))[0]

    valid_indices = np.where((area_z < threshold) &
                            (size_z < threshold) &
                            (dist_z < threshold))[0]


    valid_polygons = [approx[i] for i in valid_indices]
    return valid_polygons

def draw_quadrilateral_approx(image, quadrilaterals, color = 255, thickness=7):
    for approx in quadrilaterals:
        pts = [pt[0].tolist() for pt in approx]
        # pts = approx[0]

        # create same pattern for points , bottomright(1) , topright(2) , topleft(3) , bottomleft(4)
        (pt1, pt2, pt3, pt4) = sort_quadrilateral_approx(pts)

        # Draw only valid squares to "valid_squares_image"
        cv2.line(image, pt1, pt2, color, thickness)
        cv2.line(image, pt2, pt3, color, thickness)
        cv2.line(image, pt3, pt4, color, thickness)
        cv2.line(image, pt1, pt4, color, thickness)

    return image

def find_best_fitting_enclosing_square(image: cv2.typing.MatLike):
    closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, np.ones((5,5)), iterations=3)
    contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    best_fitting_square = cv2.approxPolyN(cnt, 4)
    approx = cv2.approxPolyN(best_fitting_square, 4)

    # Initialize variables to store extreme points
    return sort_quadrilateral_approx(best_fitting_square[0])

def chessboard_best_hough(image, canny_params, upsize_factor, scale_factor):
    canny_images = canny_multiple(image, canny_params, upsize_factor)
    hough_images = hough_multiple(canny_images)
    _, _, best_square_approx = find_best_squares_hough(hough_images, scale_factor)

def find_chessboard(image: cv2.typing.MatLike, canny_params, upsize_factor, scale_factor):
    canny_images = canny_multiple(image, canny_params, upsize_factor)
    hough_images = hough_multiple(canny_images)
    _, _, best_square_approx = find_best_squares_hough(hough_images, scale_factor)

    if best_square_approx is None:
        raise ValueError("No best square image found")

    valid_polygons = filter_valid_polygons(best_square_approx)
    valid_polygons_image = draw_quadrilateral_approx(np.zeros_like(image), valid_polygons)
    bottom_right, top_right, top_left, bottom_left = find_best_fitting_enclosing_square(valid_polygons_image)

    # Define the four source points (replace with actual coordinates)
    corners_list = np.float32([top_left, top_right, bottom_left, bottom_right])
    return corners_list


chessboard_rows, chessboard_cols = 8, 8  # 8x8 chessboard
def find_chessboard_squares(corners, threshold = 0, size = (1200, 1200)):
    '''
    :param corners (top_left, top_right, bottom_left, bottom_right)
    :param threshold extra space on all sides
    '''
    width, height = size 

    # Define the destination points (shifted by 'threshold' on all sides)
    dst_pts = np.float32([
        [threshold, threshold], 
        [width + threshold, threshold], 
        [threshold, height + threshold], 
        [width + threshold, height + threshold]
    ])

    # Compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(corners, dst_pts)
    M_inv = cv2.invert(M)[1]  # Get the inverse of the perspective matrix

    # Calculate the width and height of each square in the warped image
    square_width = width // chessboard_cols
    square_height = height // chessboard_rows

    # List to store squares' data in the correct order (bottom-left first)
    squares_data_warped = []
    for i in range(chessboard_rows - 1, -1, -1):  # Start from bottom row and move up
        for j in range(chessboard_cols):  # Left to right order
            # Define the 4 corners of each square
            top_left = (j * square_width, i * square_height)
            top_right = ((j + 1) * square_width, i * square_height)
            bottom_left = (j * square_width, (i + 1) * square_height)
            bottom_right = ((j + 1) * square_width, (i + 1) * square_height)

            # Calculate center of the square
            x_center = (top_left[0] + bottom_right[0]) // 2
            y_center = (top_left[1] + bottom_right[1]) // 2

            # Append to list in the correct order
            squares_data_warped.append([
                (x_center, y_center),
                bottom_right,
                top_right,
                top_left,
                bottom_left
            ])

    # Convert to numpy array for transformation
    squares_data_warped_np = np.array(squares_data_warped, dtype=np.float32).reshape(-1, 1, 2)

    # Transform all points back to the original image
    squares_data_original_np = cv2.perspectiveTransform(squares_data_warped_np, M_inv)

    # Reshape back to list format
    squares_data_original = squares_data_original_np.reshape(-1, 5, 2)  # (num_squares, 5 points, x/y)
    return squares_data_original

def draw_chessboard_squares(image: cv2.typing.MatLike, squares, corners, color=(0,255,0), text_color = (0, 0, 0), thickness=8):
    for idx, square in enumerate(squares):
        x_center, y_center = tuple(map(int, square[0]))  # Convert to int

        bottom_right = tuple(map(int, square[1]))
        top_right = tuple(map(int, square[2]))
        top_left = tuple(map(int, square[3]))
        bottom_left = tuple(map(int, square[4]))

        # Draw necessary lines only (to form grid)
        cv2.line(image, top_left, top_right, (0, 255, 0), 6)  # Top line
        cv2.line(image, top_left, bottom_left, (0, 255, 0), 6)  # Left line

        # Draw bottom and right lines only for last row/column
        i = idx // chessboard_rows
        j = idx % chessboard_cols

        if idx % chessboard_cols == chessboard_cols - 1:
            cv2.line(image, top_right, bottom_right, (0, 255, 0), 8)  # Right line
        if i == 0:
            cv2.line(image, bottom_left, bottom_right, (0, 255, 0), 8)  # Bottom line

    # Mark the extreme points
    corners = corners.astype(int)

    cv2.circle(image, corners[0], 15, (0, 0, 255), -1)  # blue for top-left
    cv2.circle(image, corners[1], 15, (0, 255, 0), -1)  # green for top-right
    cv2.circle(image, corners[2], 15, (255, 0,0), -1)  # red for bottom-left
    cv2.circle(image, corners[3], 15, (255, 255, 0), -1)  # yellow for bottom-right

    cv2.putText(image, "TOP LEFT", corners[0], cv2.FONT_HERSHEY_SIMPLEX, 3, text_color, 2, cv2.LINE_AA)
    cv2.putText(image, "TOP RIGHT", corners[1], cv2.FONT_HERSHEY_SIMPLEX, 3, text_color, 2, cv2.LINE_AA)
    cv2.putText(image, "BOTTOM LEFT", corners[2], cv2.FONT_HERSHEY_SIMPLEX, 3, text_color, 2, cv2.LINE_AA)
    cv2.putText(image, "BOTTOM RIGHT", corners[3], cv2.FONT_HERSHEY_SIMPLEX, 3, text_color, 2, cv2.LINE_AA)
