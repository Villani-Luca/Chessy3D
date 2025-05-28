import sys
import time

import chess
import ultralytics
from PySide6.QtCore import QThreadPool
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (QApplication, QWidget, QGridLayout)

from src.gui.chessboard import ChessBoard, ChessBoardWidget
from src.gui.datagrid import DataGrid
from src.gui.fileuploader import FileUploader
from src.gui.recog import RecognitionJob
from src.gui.worker import Worker, WorkerSignals
from src.retrieval.src.model.pgsql import Connection, PgGamesRepository
import src.chessboard_localization_temp.main as chess_localization

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
        self.setWindowTitle("Chessy 3D")

        self.threadpool = QThreadPool()

        # CONST
        PG_CONN = args['pgconn']

        # globals
        board = chess.Board()
        pgconn = Connection(PG_CONN)
        games_repo = PgGamesRepository(pgconn)

        # Create widgets
        # self.chess_board = ChessBoard(board)
        self.chess_widget = ChessBoardWidget(ChessBoard(board))

        def on_file_upload_callback(uploader: FileUploader, filename: str):
            img, resized = chess_localization.chessboard_localization_resize(filename)

            def start_callback():
                board.clear()
                self.chess_widget.draw_board()

            def progress_callback(progress: float):
                print(progress)

            def finish_callback(result):
                game_list, img = result

                self.file_uploader.setOpencvImage(img)

                board.clear()
                for (cell, detected_class) in game_list:
                    piece = chess.Piece(piece_mapping[detected_class], chess.WHITE if detected_class < 6 else chess.BLACK)
                    board.set_piece_at(cell - 1, piece)
                self.chess_widget.draw_board()

            recogjob = RecognitionJob(
                img,
                resized,
                r"D:\Projects\Uni\Chessy3D\src\object_detection_yolo\chess-model-yolov8m.pt"
            )
            recogjob.signals.update_image.connect(self.file_uploader.setOpencvImage)
            recogjob.signals.start.connect(start_callback)
            recogjob.signals.progress.connect(progress_callback)
            recogjob.signals.result.connect(finish_callback)
            recogjob.signals.error.connect(lambda x: print(x))

            self.threadpool.start(recogjob)



        self.file_uploader = FileUploader(callback=on_file_upload_callback)
        self.data_grid = DataGrid()

        # Main layout with QGridLayout
        main_layout = QGridLayout()

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
        'pgconn': r"host=localhost user=postgres password=password dbname=chessy"
    }


    app = QApplication(sys.argv)
    window = MainWindow(args)
    window.resize(1200, 900)
    window.show()
    sys.exit(app.exec())
