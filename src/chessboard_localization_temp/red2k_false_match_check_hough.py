import math

import cv2

import src.chessboard_localization_temp.main as chess_localization
import re
import pathlib
import concurrent.futures
import itertools

root = pathlib.Path(r"D:\Projects\Uni\Chessy3D")
folder = pathlib.Path(r"C:\Users\Alessandro\Downloads")
text_file = folder / "result_false_50_puliti_manualmente.txt"
output_folder = folder / "result_false_check_hough/"
pattern = r"(?<=image path: )[^,]+"

def work(chunk: list[tuple[str, int]]):
    image_path, chunk = chunk

    for img_path in image_path:
        img, resized = chess_localization.chessboard_localization_resize((root / img_path).as_posix())
        output_image = chess_localization.chessboard_best_hough(img, resized)
        output_image_path = output_folder / img_path.split("\\")[-1]
        cv2.imwrite(output_image_path.as_posix(), output_image)

    print(chunk)

if __name__ == "__main__":
    files: list[str] = []
    with text_file.open('r') as f:
        for line in f.readlines():
            files.append(re.findall(pattern, line)[0])

    if not output_folder.exists():
        output_folder.mkdir()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        n = math.ceil(len(files) / 10)

        for chunk in zip(itertools.batched(files, 10), range(n)):
            executor.submit(work, chunk)

