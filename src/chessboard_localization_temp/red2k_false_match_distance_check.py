import math

import cv2

import src.chessboard_localization_temp.main as chess_localization
import re
import pathlib
import concurrent.futures
import itertools

root = pathlib.Path(r"D:\Projects\Uni\Chessy3D")
folder = pathlib.Path(r"C:\Users\Alessandro\Downloads")
text_file = folder / "result_false.txt"
output_text_file = folder / "result_false_50.txt"

# regex to capture the image path and all the distances
pattern = r"image path: ([^,]+), distances: \[([^\]]+)\]"



write_lines = []
with text_file.open("r") as f:
    for line in f.readlines():
        for image_path, distances_str in re.findall(pattern, line):
            # parse distances to floats
            distances = [float(d) for d in distances_str.split()]
            # check if any distance is greater than 50
            has_large_distance = any(d > 50 for d in distances)

            if has_large_distance:
                write_lines.append(line)

with output_text_file.open("w") as o:
    o.writelines(write_lines)
