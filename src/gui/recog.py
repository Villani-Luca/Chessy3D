import cv2
import ultralytics
from PySide6.QtCore import Signal, QThreadPool

import src.chessboard_localization_temp.main as chess_localization
from src.gui.worker import Worker, WorkerSignals

class RecognitionJobSignals(WorkerSignals):
    update_image = Signal(object)

class RecognitionJob(Worker):
    def __init__(self,
                 img: cv2.typing.MatLike,
                 resized_image: cv2.typing.MatLike,
                 recog_yolo_path: str):
        super().__init__(RecognitionJobSignals())

        # Store constructor arguments (re-used for processing)
        self.img = img
        self.resized_image = resized_image
        self.recog_yolo_path = recog_yolo_path

        self.signals = RecognitionJobSignals()

    def execute(self):
        _, squares_data_original, img, rgb_image = self.__chessboard_localization()
        self.signals.update_image.emit(rgb_image)

        game_list, result_plot = self.__piece_recognition(img, squares_data_original)
        return (game_list, result_plot)

    def __chessboard_localization(self):
        (
            rgb_image,
            corners_list,
            squares_data_original,
            img
        ) = chess_localization.auto_chessboard_localization(
            self.img,
            self.resized_image
        )

        return corners_list, squares_data_original, img, rgb_image

    def __piece_recognition(self, image, squares_data_original):
        model = ultralytics.YOLO(self.recog_yolo_path)

        # make prediction
        results = model(image)  # path to test image

        coord_dict = {}
        for cell, coordinate in enumerate(squares_data_original, start=1):
            center, bottom_right, top_right, top_left, bottom_left = coordinate
            coord_dict[cell] = [bottom_right, top_right,  top_left, bottom_left]

        game_list = []
        for result in results:  # results is model's prediction
            for id, box in enumerate(result.boxes.xyxy):  # box with xyxy format, (N, 4)

                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])  # take coordinates

                # find middle of bounding boxes for x and y
                x_mid = int((x1 + x2) / 2)
                # add padding to y values
                # TODO: al posto di 75 prendere l'ultimo quarto / quinto della bounding box, dipende dalla risoluzione
                y_mid = int((y1 + y2) / 2) + 75

                for cell_value, coordinates in coord_dict.items():
                    x_values = [point[0] for point in coordinates]
                    y_values = [point[1] for point in coordinates]

                    if (min(x_values) <= x_mid <= max(x_values)) and (min(y_values) <= y_mid <= max(y_values)):
                        a = int(result.boxes.cls[id])

                        #print(f" cell :  {cell_value} --> {a} ")
                        # add cell values and piece cell_value(class value
                        game_list.append([cell_value, a])
                        break

        return game_list, results[0].plot()
