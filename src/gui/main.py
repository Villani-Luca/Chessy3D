import re
import sys

import chess
import chess.polyglot
import cv2.typing
import numpy as np
import ultralytics
from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import (QApplication, QWidget, QGridLayout)
from psycopg.rows import Row

from src.gui.chessboard import ChessBoard, ChessBoardWidget
from src.gui.datagrid import DataGrid
from src.gui.fileuploader import FileUploader
from src.gui.retrivaljob import RetrievalJob
from src.retrieval.src.commons import zobrist_to_s64
from src.retrieval.src.pgsql import Connection, PgGamesRepository
import src.chessboard_localization_temp.main as chess_localization
from src.retrieval.src.position_embeddings import NaivePositionEmbedder

# yolo chess-model-yolov8m
piece_mapping_yolo1 = {
    0:  (chess.BISHOP,  chess.BLACK, (0, 100, 0)    ),  # white-bishop
    1:  (chess.KING,    chess.BLACK, (0, 0, 139)    ),  # white-king
    2:  (chess.KNIGHT,  chess.BLACK, (0, 85, 170)   ),  # white-knight
    3:  (chess.PAWN,    chess.BLACK, (80, 80, 80)   ),  # white-pawn
    4:  (chess.QUEEN,   chess.BLACK, (64, 0, 64)    ),  # white-queen
    5:  (chess.ROOK,    chess.BLACK, (139, 0, 0)    ),  # white-rook
    6:  (chess.BISHOP,  chess.WHITE, (0, 255, 0)    ),  # black-bishop
    7:  (chess.KING,    chess.WHITE, (0, 0, 255)    ),  # black-king
    8:  (chess.KNIGHT,  chess.WHITE, (0, 165, 255)  ),  # black-knight
    9:  (chess.PAWN,    chess.WHITE, (200, 200, 200)),  # black-pawn
    10: (chess.QUEEN,   chess.WHITE, (128, 0, 128)  ),  # black-queen
    11: (chess.ROOK,    chess.WHITE, (255, 0, 0)    ),  # black-rook
}

piece_mapping_yolo2 = {
    0:  (chess.PAWN,    chess.BLACK, (80, 80, 80)   ),  # black-pawn
    1:  (chess.ROOK,    chess.BLACK, (139, 0, 0)    ),  # black-rook
    2:  (chess.KNIGHT,  chess.BLACK, (0, 85, 170)   ),  # black-knight
    3:  (chess.BISHOP,  chess.BLACK, (0, 100, 0)    ),  # black-bishop
    4:  (chess.QUEEN,   chess.BLACK, (64, 0, 64)    ),  # black-queen
    5:  (chess.KING,    chess.BLACK, (0, 0, 139)    ),  # black-king
    6:  (chess.PAWN,    chess.WHITE, (200, 200, 200)),  # white-pawn
    7:  (chess.ROOK,    chess.WHITE, (255, 0, 0)    ),  # white-rook
    8:  (chess.KNIGHT,  chess.WHITE, (0, 165, 255)  ),  # white-knight
    9:  (chess.BISHOP,  chess.WHITE, (0, 255, 0)    ),  # white-bishop
    10: (chess.QUEEN,   chess.WHITE, (128, 0, 128)  ),  # white-queen
    11: (chess.KING,    chess.WHITE, (0, 0, 255)    ),  # white-king
}

SAN_MOVE_REGEX = r'(?:\d+\.)?([KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?|O-O(?:-O)?)[+#]?'

class MainWindow(QWidget):
    def __init__(self, args: dict):
        super().__init__()

        # CONST
        pg_conn = args['pgconn']
        yolo_path = args['object_detection_yolo']
        piece_mapping = args['piece_mapping']
        debug = args["debug"]

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

        self.chess_widget = ChessBoardWidget(ChessBoard(chess.Board(None), debug=debug))

        def __on_retrieval_result(r: list[Row]):
            self.retrieval_result = r
            self.data_grid.set_data([x[:-2] for x in r])

        def refresh_datagrid():
            embedding = self.chess_widget.retrieve_embedding(position_embedder)
            self.retrieval_job = RetrievalJob(embedding, games_repo)
            self.retrieval_job.signals.start.connect(lambda: self.data_grid.set_enable_refreshbutton(False))
            self.retrieval_job.signals.result.connect(__on_retrieval_result)
            self.retrieval_job.signals.finished.connect(lambda: self.data_grid.set_enable_refreshbutton(True))

            self.threadpool.start(self.retrieval_job)

        def on_file_upload_callback(uploader: FileUploader, filename: str):
            img, resized = chess_localization.chessboard_localization_resize(filename)

            self.chess_widget.draw_board(chess.Board(None))
            # rgb_image, corners_list, squares_data_original, img, best_canny, best_hough, polygons_image = chess_localization.auto_chessboard_localization(img,resized)
            final_image, corners_list, squares_data_original, img, best_canny, best_hough, polygons_image, upsize_factor = chess_localization.auto_chessboard_localization_alt(img,resized)

            # make prediction
            results = yolo(img)  # resized image

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

                        if cv2.pointPolygonTest(np.array(coordinates).reshape(-1, 1, 2), (x_mid, y_mid), False) >= 0:
                            # add cell values and piece cell_value(class value
                            game_list.append([cell_value, class_id])
                            break

                    # custom draw yolo result on image
                    color = piece_mapping.get(class_id, (255, 255, 255))[2]  # Default to white if class not in mapping
                    cv2.rectangle(final_image, (x1, y1), (x2, y2), color, 4)

            ######
            mesh_img = np.zeros_like(final_image)
            for result in results:  # results is model's prediction
                for idx, box in enumerate(result.boxes.xyxy):  # box with xyxy format, (N, 4)
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])  # take coordinates
                    x_mid = int((x1 + x2) / 2)
                    y_mid = y2 - 30
                    class_id = int(result.boxes.cls[idx])
                    color = piece_mapping.get(class_id, (255, 255, 255))[2]
                    cv2.circle(mesh_img, (x_mid, y_mid), 5, color, -1)

            for cell, coordinates in coord_dict.items():
                cv2.polylines(mesh_img, [np.array(coordinates, dtype=np.int32).reshape(-1, 1, 2)], True, (255, 255, 255), 2)
            #######

            self.file_uploader.set_opencv_image(best_canny, FileUploader.Tabs.CANNY)
            self.file_uploader.set_opencv_image(best_hough, FileUploader.Tabs.HOUGH)
            self.file_uploader.set_opencv_image(polygons_image, FileUploader.Tabs.SQUARES)
            self.file_uploader.set_opencv_image(mesh_img, FileUploader.Tabs.MESH)
            self.file_uploader.set_opencv_image(final_image, FileUploader.Tabs.FINAL)
            new_board = chess.Board(None)
            for (cell, detected_class) in game_list:
                piece = chess.Piece(piece_mapping[detected_class][0], piece_mapping[detected_class][1])
                new_board.set_piece_at(cell - 1, piece)

            self.chess_widget.draw_board(new_board)
            self.file_uploader.open_tab(FileUploader.Tabs.FINAL)

        def __on_row_dblclick(row_data):
            print(row_data)
            if self.retrieval_result is None:
                return

            full_row = next((x for x in self.retrieval_result if x[0] == int(row_data[0])), None)
            if full_row is None:
                return

            position_hash, moves = full_row[-2], full_row[-1]
            similar_board = chess.Board()
            for san_move in re.findall(SAN_MOVE_REGEX, moves):
                similar_board.push_san(san_move)
                if position_hash == zobrist_to_s64(chess.polyglot.zobrist_hash(similar_board)):
                    break

            self.chess_widget.show_relative_board(similar_board, full_row[0], full_row[3], full_row[5])


        self.file_uploader = FileUploader(callback=on_file_upload_callback)
        self.data_grid = DataGrid(on_refresh_button=refresh_datagrid, on_row_double_click=__on_row_dblclick)

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
        "object_detection_yolo": r"E:\projects\uni\Chessy3D\src\object_detection_yolo\chess-model-yolov8m.pt",
        # "object_detection_yolo": r"E:\projects\uni\Chessy3D\src\object_detection_yolo\20250623_finetune_50.pt",
        "piece_mapping": piece_mapping_yolo1,
        # "object_detection_yolo": r"E:\projects\uni\Chessy3D\src\object_detection_yolo\best_final.pt",
        # "piece_mapping": piece_mapping_yolo2,
        "debug": False,
    }


    app = QApplication(sys.argv)
    window = MainWindow(args)
    window.resize(1200, 900)
    window.show()
    sys.exit(app.exec())
