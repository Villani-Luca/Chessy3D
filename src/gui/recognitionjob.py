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
                 yolo: ultralytics.YOLO):
        super().__init__(RecognitionJobSignals())

        # Store constructor arguments (re-used for processing)
        self.img = img
        self.resized_image = resized_image
        self.yolo = yolo

        self.signals = RecognitionJobSignals()

    def execute(self):
        _, squares_data_original, img, rgb_image = self.__chessboard_localization()
        self.signals.update_image.emit(rgb_image)

        game_list, result_plot = self.__piece_recognition(img, rgb_image, squares_data_original)
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

    class_colors = {
        9: (200, 200, 200),  # white-pawn
        3: (80, 80, 80),  # black-pawn
        8: (0, 165, 255),  # white-knight
        2: (0, 85, 170),  # black-knight
        0: (0, 255, 0),  # white-bishop
        6: (0, 100, 0),  # black-bishop
        11: (255, 0, 0),  # white-rook
        5: (139, 0, 0),  # black-rook
        10: (128, 0, 128),  # white-queen
        4: (64, 0, 64),  # black-queen
        7: (0, 0, 255),  # white-king
        1: (0, 0, 139),  # black-king
    }

    def __piece_recognition(self, image, outimage, squares_data_original):
        # make prediction
        results = self.yolo(image)  # path to test image

        coord_dict = {}
        for cell, coordinate in enumerate(squares_data_original, start=1):
            center, bottom_right, top_right, top_left, bottom_left = coordinate
            coord_dict[cell] = [bottom_right, top_right,  top_left, bottom_left]

        game_list = []
        for result in results:  # results is model's prediction
            for idx, box in enumerate(result.boxes.xyxy):  # box with xyxy format, (N, 4)

                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])  # take coordinates
                class_id = int(result.boxes.cls[idx])

                # find middle of bounding boxes for x and y
                x_mid = int((x1 + x2) / 2)
                # add padding to y values
                # TODO: al posto di 75 prendere l'ultimo quarto / quinto della bounding box, dipende dalla risoluzione
                y_distance = abs(y2 - y1) // 10
                # y_mid = int((y1 + y2) / 2) + 75
                y_mid = y2 - y_distance*3

                for cell_value, coordinates in coord_dict.items():
                    x_values = [point[0] for point in coordinates]
                    y_values = [point[1] for point in coordinates]

                    if (min(x_values) <= x_mid <= max(x_values)) and (min(y_values) <= y_mid <= max(y_values)):


                        #print(f" cell :  {cell_value} --> {a} ")
                        # add cell values and piece cell_value(class value
                        game_list.append([cell_value, class_id])
                        break

                # custom draw yolo result on image
                color = self.class_colors.get(class_id, (255, 255, 255))  # Default to white if class not in mapping
                cv2.rectangle(outimage, (x1, y1), (x2, y2), color, 8)
                # cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


        return game_list, outimage