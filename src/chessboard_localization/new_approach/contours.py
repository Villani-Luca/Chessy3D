import cv2
import numpy as np
import drawing as dw

def get_countours(hough_lines, lines_image):

    board_contours, hierarchy = cv2.findContours(
        lines_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    approximated_quads_contours = []

    for contour in board_contours:

        contour_area = cv2.contourArea(contour)

        if contour_area < 20000:
            continue

        if contour_area > 100000:
            continue

        contour_perimeter = cv2.arcLength(contour, closed=True)
        # how much the contour is going to be approximated
        # high epsilon = course approximation, low epsilon = more precise approximation
        epsilon = 0.02 * contour_perimeter  # 2% of the perimeter
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)

        if len(approx) == 4:
            ordered_points = np.array(cv2.convexHull(approx), dtype=np.int32)

            if len(ordered_points) != 4:
                continue

            p1, p2, p3, p4 = ordered_points

            d12 = np.linalg.norm(p1 - p2)
            d34 = np.linalg.norm(p3 - p4)

            d23 = np.linalg.norm(p2 - p3)
            d31 = np.linalg.norm(p4 - p1)

            lengths = np.array([d12, d34, d23, d31])

            max_length = lengths.max()
            min_length = lengths.min()

            if (max_length - min_length) > 200:
                continue

            approximated_quads_contours.append(approx)

    contours_image = np.zeros(lines_image.shape, np.uint8)
    cv2.drawContours(contours_image, approximated_quads_contours, -1, (255, 255, 255), 3)
    dw.display_image_cv2(contours_image, window_name=f"contours")