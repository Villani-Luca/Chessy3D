import sys

import chess
import cv2.typing
import ultralytics
from PySide6.QtCore import QThreadPool, Slot
from PySide6.QtWidgets import (QApplication, QWidget, QGridLayout)

from src.gui.chessboard import ChessBoard, ChessBoardWidget
from src.gui.datagrid import DataGrid
from src.gui.fileuploader import FileUploader
from src.gui.recognitionjob import RecognitionJob
from src.gui.retrivaljob import RetrievalJob
from src.retrieval.src.model.pgsql import Connection, PgGamesRepository
import src.chessboard_localization_temp.main as chess_localization
from src.retrieval.src.position_embeddings import NaivePositionEmbedder

piece_mapping = {
    9: chess.PAWN,  # white-pawn
    3: chess.PAWN,  # black-pawn
    8: chess.KNIGHT,  # white-knight
    2: chess.KNIGHT,  # black-knight
    0: chess.BISHOP,  # white-bishop
    6: chess.BISHOP,  # black-bishop
    11: chess.ROOK,  # white-rook
    5: chess.ROOK,  # black-rook
    10: chess.QUEEN,  # white-queen
    4: chess.QUEEN,  # black-queen
    7: chess.KING,  # white-king
    1: chess.KING,  # black-king
}

class MainWindow(QWidget):
    def __init__(self, args: dict):
        super().__init__()

        # CONST
        pg_conn = args['pgconn']
        yolo_path = args['object_detection_yolo']
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

        self.setWindowTitle("Chessy 3D")
        self.threadpool = QThreadPool()

        # globals
        print("Setting up main window...")
        pgconn = Connection(pg_conn)
        yolo = ultralytics.YOLO(yolo_path)

        print("Setting up services...")
        games_repo = PgGamesRepository(pgconn)
        position_embedder = NaivePositionEmbedder()

        # Create widgets
        print("Setting up widgets...")

        self.chess_widget = ChessBoardWidget(ChessBoard(chess.Board(None)))

        def refresh_datagrid():
            embedding = self.chess_widget.retrieve_embedding(position_embedder)
            self.retrieval_job = RetrievalJob(embedding, games_repo)
            self.retrieval_job.signals.start.connect(lambda: self.data_grid.set_enable_refreshbutton(False))
            self.retrieval_job.signals.result.connect(lambda r: self.data_grid.set_data(r))
            self.retrieval_job.signals.finished.connect(lambda: self.data_grid.set_enable_refreshbutton(True))

            self.threadpool.start(self.retrieval_job)

        def on_file_upload_callback(uploader: FileUploader, filename: str):
            img, resized = chess_localization.chessboard_localization_resize(filename)

            self.chess_widget.draw_board(chess.Board(None))
            rgb_image, corners_list, squares_data_original, img, best_canny, best_hough, polygons_image = chess_localization.auto_chessboard_localization(img,resized)

            # make prediction
            results = yolo(img)  # path to test image

            coord_dict = {}
            for cell, coordinate in enumerate(squares_data_original, start=1):
                center, bottom_right, top_right, top_left, bottom_left = coordinate
                coord_dict[cell] = [bottom_right, top_right, top_left, bottom_left]

            game_list = []
            for result in results:  # results is model's prediction
                for idx, box in enumerate(result.boxes.xyxy):  # box with xyxy format, (N, 4)

                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])  # take coordinates
                    class_id = int(result.boxes.cls[idx])

                    # find middle of bounding boxes for x and y
                    x_mid = int((x1 + x2) / 2)
                    # add padding to y values
                    y_mid = y2 - 30

                    for cell_value, coordinates in coord_dict.items():
                        x_values = [point[0] for point in coordinates]
                        y_values = [point[1] for point in coordinates]

                        if (min(x_values) <= x_mid <= max(x_values)) and (min(y_values) <= y_mid <= max(y_values)):
                            # add cell values and piece cell_value(class value
                            game_list.append([cell_value, class_id])
                            break

                    # custom draw yolo result on image
                    color = class_colors.get(class_id, (255, 255, 255))  # Default to white if class not in mapping
                    cv2.rectangle(rgb_image, (x1, y1), (x2, y2), color, 8)

            self.file_uploader.set_opencv_image(best_canny, FileUploader.Tabs.CANNY)
            self.file_uploader.set_opencv_image(best_hough, FileUploader.Tabs.HOUGH)
            self.file_uploader.set_opencv_image(polygons_image, FileUploader.Tabs.SQUARES)
            self.file_uploader.set_opencv_image(rgb_image, FileUploader.Tabs.FINAL)
            new_board = chess.Board(None)
            for (cell, detected_class) in game_list:
                piece = chess.Piece(piece_mapping[detected_class], chess.WHITE if detected_class < 6 else chess.BLACK)
                new_board.set_piece_at(cell - 1, piece)

            self.chess_widget.draw_board(new_board)

        self.file_uploader = FileUploader(callback=on_file_upload_callback)
        self.data_grid = DataGrid(on_refresh_button=refresh_datagrid)

        # Main layout with QGridLayout
        main_layout =  QGridLayout()

        # Add widgets to the grid
        main_layout.addWidget(self.file_uploader, 0, 0)
        main_layout.addWidget(self.chess_widget, 0, 1)
        main_layout.addWidget(self.data_grid, 1, 0, 1, 2)  # Span across two columns

        # Set row stretch to make the first row twice as large as the second row
        main_layout.setRowStretch(0, 2)
        main_layout.setRowStretch(1, 1)

        # Set column stretch for equal resizing of columns in the first row
        main_layout.setColumnStretch(0, 1)
        main_layout.setColumnStretch(1, 1)

        self.setLayout(main_layout)

if __name__ == "__main__":
    args = {
        'pgconn': r"host=localhost user=postgres password=password dbname=chessy",
        #"milvus_url": r"http://localhost:19530",
        #"milvus_collection": NAIVE_COLLECTION_NAME,
        #"object_detection_yolo": r"D:\Projects\Uni\Chessy3D\src\object_detection_yolo\best_yolo_e200_small.pt",
        "object_detection_yolo": r"D:\Projects\Uni\Chessy3D\src\object_detection_yolo\chess-model-yolov8m.pt"
    }


    app = QApplication(sys.argv)
    window = MainWindow(args)
    window.resize(1200, 900)
    window.show()
    sys.exit(app.exec())
