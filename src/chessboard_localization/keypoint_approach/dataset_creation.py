import json
import shutil
from pathlib import Path

def __calculate_bbox_for_pieces(bbox, img_w, img_h):
    yolo_bbox = [
        bbox[0] / img_w,   # x_center
        bbox[1] / img_h,   # y_center
        bbox[2] / img_w,   # width
        bbox[3] / img_h    # height
    ]
    return yolo_bbox

def convert_chessred2k_dataset_pieces(src_dataset_path, dest_dataset_path):

    src_base_dir = Path(src_dataset_path)
    dest_base_dir = Path(dest_dataset_path)
    dest_base_dir.mkdir(exist_ok=True)

    with open(src_base_dir / "annotations.json") as f:
        parsed_file = json.load(f)

    # Initialize splits
    splits = ["train", "val", "test"]

    for split in splits:
        (dest_base_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (dest_base_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    for split in splits:

        # Use the chessred2k instead of the full splits
        image_ids = parsed_file["splits"]["chessred2k"][split]["image_ids"]

        for img_id in image_ids:

            # Find image metadata
            img_info = next(img for img in parsed_file["images"] if img["id"] == img_id)

            img_file = img_info["file_name"]
            img_path = img_info["path"]
            img_w = img_info["width"]
            img_h = img_info["height"]

            # Copy image
            src_img_path = src_base_dir / img_path
            dst_img_path = dest_base_dir / split / "images" / img_file
            shutil.copy(src_img_path, dst_img_path)

            # Get pieces annotation
            pieces_annotations = [
                a for a in parsed_file["annotations"]["pieces"] if a["image_id"] == img_id
            ]

            label_path = dest_base_dir / split / "labels" / f"{Path(img_file).stem}.txt"

            with open(label_path, "w") as f_txt:
                for annotation in pieces_annotations:

                    piece_class = annotation["category_id"]

                    bbox_norm = __calculate_bbox_for_pieces(
                        annotation["bbox"], img_w, img_h
                    )

                    line = [piece_class] + bbox_norm
                    f_txt.write(" ".join(map(str, line)) + "\n")

def __calculate_bbox_from_corners(corners_dict, img_w, img_h):

    x_coords = [
        corners_dict["top_left"][0],
        corners_dict["top_right"][0],
        corners_dict["bottom_left"][0],
        corners_dict["bottom_right"][0],
    ]

    y_coords = [
        corners_dict["top_left"][1],
        corners_dict["top_right"][1],
        corners_dict["bottom_left"][1],
        corners_dict["bottom_right"][1],
    ]

    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)

    width = x_max - x_min
    height = y_max - y_min

    x_center = (x_min + width / 2) / img_w
    y_center = (y_min + height / 2) / img_h

    # Normalize the bbox size
    width_norm = width / img_w
    height_norm = height / img_h

    # BBox format: [x_center, y_center, width, height]
    return [x_center, y_center, width_norm, height_norm]

def convert_chessred2k_dataset_corners(src_dataset_path, dest_dataset_path):

    src_base_dir = Path(src_dataset_path)
    dest_base_dir = Path(dest_dataset_path)
    dest_base_dir.mkdir(exist_ok=True)

    with open(src_base_dir / "annotations.json") as f:
        parsed_file = json.load(f)

    # Initialize splits
    splits = ["valid", "val", "test"]

    for split in splits:
        (dest_base_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (dest_base_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    for split in splits:

        # Use the chessred2k instead of the full splits
        image_ids = parsed_file["splits"]["chessred2k"][split]["image_ids"]

        for img_id in image_ids:

            # Find image metadata
            img_info = next(img for img in parsed_file["images"] if img["id"] == img_id)

            img_file = img_info["file_name"]
            img_path = img_info["path"]
            img_w = img_info["width"]
            img_h = img_info["height"]

            # Copy image
            src_img_path = src_base_dir / img_path
            dst_img_path = dest_base_dir / split / "images" / img_file
            shutil.copy(src_img_path, dst_img_path)

            # Get corner annotation
            corner_annotations = [
                a for a in parsed_file["annotations"]["corners"] if a["image_id"] == img_id
            ]

            label_path = dest_base_dir / split / "labels" / f"{Path(img_file).stem}.txt"

            with open(label_path, "w") as f_txt:
                for annotation in corner_annotations:
                    bbox_norm = __calculate_bbox_from_corners(
                        annotation["corners"], img_w, img_h
                    )

                    kpts = [
                        annotation["corners"]["top_left"][0] / img_w,  # x1
                        annotation["corners"]["top_left"][1] / img_h,  # y1
                        1,  # v1 (visible)
                        annotation["corners"]["top_right"][0] / img_w,  # x2
                        annotation["corners"]["top_right"][1] / img_h,  # y2
                        1,  # v2
                        annotation["corners"]["bottom_right"][0] / img_w,  # x3
                        annotation["corners"]["bottom_right"][1] / img_h,  # y3
                        1,  # v3
                        annotation["corners"]["bottom_left"][0] / img_w,  # x4
                        annotation["corners"]["bottom_left"][1] / img_h,  # y4
                        1,  # v4
                    ]

                    # Write to file (class 0 = chessboard)
                    line = [0] + bbox_norm + kpts
                    f_txt.write(" ".join(map(str, line)) + "\n")

if __name__ == "__main__":
    chessred2k_path = "data/chessred2k"
    board_localization_path = "data/board_localization"
    #convert_chessred2k_dataset_corners(chessred2k_path, board_localization_path)

    convert_chessred2k_dataset_pieces("data/chessred2k", "data/pieces_detection")
