from pathlib import Path
import lines
import cv2
import drawing as dw
import line_math as lm
import numpy as np

def process_lines(file_path):
    original_image = cv2.imread(file_path)
    img_height, img_width = original_image.shape[:2]
    black_image = np.zeros(original_image.shape, np.uint8)

    file_name = Path(file_path).name
    #dw.display_image_cv2(original_image, window_name=f"{file_name} -> original")

    segments, lines_image = lines.get_lines(original_image, file_name)
    print(f"segments before filtering: {len(segments)}")
    #dw.display_image_cv2(lines_image, window_name=f"{file_name} -> segments")
   
    #expanded_segments = lm.expand_segments_relative(segments, 100, img_height, img_width)
    #expanded_segments_image = dw.create_image_with_segments(expanded_segments, original_image, 2)
    #dw.display_image_cv2(expanded_segments_image, window_name=f"{file_name} -> expanded")

    labels = lines.cluster_lines(segments)
    cluster_image = dw.create_image_with_segments(segments, original_image, 2, labels)
    dw.display_image_cv2(cluster_image, window_name=f"{file_name} -> cluster")

    filtered_segments, filtered_labels = lines.filter_lines_duplicates(segments, labels)
    filtered_segments_image = dw.create_image_with_segments(filtered_segments, original_image, 2, filtered_labels)
    print(f"segments after filtering: {len(filtered_segments)}")
    # dw.display_image_cv2(filtered_segments_image, window_name=f"{file_name} -> filtered segments")
    

    merged_segments, merged_labels = lines.merge_close_segments(filtered_segments, filtered_labels, img_height, img_width)
    merged_segments_image = dw.create_image_with_segments(merged_segments, original_image, 2, merged_labels)
    print(f"segments after merging: {len(merged_segments)}")
    #dw.display_image_cv2(merged_segments_image, window_name=f"{file_name} -> merged segments")
    

    intersection_matrix, unique_points = lines.find_segment_intersections(merged_segments, merged_labels)
    intersections_image = dw.create_image_with_points(unique_points,black_image, 5)
    print(f"unique points: {len(unique_points)}")
    #dw.display_image_cv2(intersections_image, window_name=f"{file_name} -> intersections")

    squares, quads = lines.get_squares(intersection_matrix)
    squares_image = dw.create_image_with_quads(squares, black_image, 4)
    print(f"quads: {len(quads)}")
    print(f"squares: {len(squares)}")
    dw.display_image_cv2(squares_image, window_name=f"{file_name} -> squares")

def process_contours(file_path):
    original_image = cv2.imread(file_path)
    img_height, img_width = original_image.shape[:2]
    black_image = np.zeros(original_image.shape, np.uint8)

    file_name = Path(file_path).name
    #dw.display_image_cv2(original_image, window_name=f"{file_name} -> original")

    segments, lines_image = lines.get_lines(original_image, file_name)
    #dw.display_image_cv2(lines_image, window_name=f"{file_name} -> segments")

    board_contours, hierarchy = cv2.findContours(
        lines_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    contours_image = np.zeros(lines_image.shape, np.uint8)
    cv2.drawContours(contours_image, board_contours, -1, (255, 255, 255), 3)
    dw.display_image_cv2(contours_image, window_name=f"{file_name} -> contours")


def process_all_images():
    rootdir = Path("data/chessred2k/images")
    file_list = [f for f in rootdir.glob("**/*") if f.is_file()]

    for file_path in file_list:
       process_contours(file_path)

def process_single_image():
    file_path = Path("data/chessred2k/images/0/G000_IMG011.jpg")
    process_contours(file_path)

if __name__ == "__main__":
    process_all_images()
    #process_single_image()
