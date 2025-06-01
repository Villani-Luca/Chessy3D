import sys

import chess
import cv2.typing
import ultralytics
from PySide6.QtCore import QThreadPool
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

        self.chess_widget = ChessBoardWidget(ChessBoard(chess.Board()))

        def refresh_datagrid():
            embedding = self.chess_widget.retrieve_embedding(position_embedder)
            self.retrieval_job = RetrievalJob(embedding, games_repo)
            self.retrieval_job.signals.start.connect(lambda: self.data_grid.set_enable_refreshbutton(False))
            self.retrieval_job.signals.result.connect(lambda r: self.data_grid.set_data(r))
            self.retrieval_job.signals.finished.connect(lambda: self.data_grid.set_enable_refreshbutton(True))

            self.threadpool.start(self.retrieval_job)

        def on_file_upload_callback(uploader: FileUploader, filename: str):
            img, resized = chess_localization.chessboard_localization_resize(filename)

            def start_callback():
                self.chess_widget.draw_board(chess.Board())

            def progress_callback(progress: float):
                print(progress)

            def finish_callback(result: tuple[list[tuple[int, int]], cv2.typing.MatLike]):
                try:
                    game_list, result_img = result
                    self.file_uploader.setOpencvImage(result_img)

                    new_board = chess.Board()
                    for (cell, detected_class) in game_list:
                        piece = chess.Piece(piece_mapping[detected_class], chess.WHITE if detected_class < 6 else chess.BLACK)
                        new_board.set_piece_at(cell - 1, piece)

                    self.chess_widget.draw_board(new_board)
                except Exception as e:
                    print("ERROR finish_callback: ", e)
                #refresh_datagrid()

            def update_image_callback(image: cv2.typing.MatLike):
                try:
                    self.file_uploader.setOpencvImage(image)
                except Exception as e:
                    print("ERROR update_image_callback: ", e)

            recognition_job = RecognitionJob(img, resized, yolo)
            recognition_job.signals.update_image.connect(update_image_callback)
            recognition_job.signals.start.connect(start_callback)
            recognition_job.signals.progress.connect(progress_callback)
            recognition_job.signals.result.connect(finish_callback)
            recognition_job.signals.error.connect(lambda x: print(x))

            self.threadpool.start(recognition_job)

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
