import sys
from typing import Callable
import pathlib

import chess
import ultralytics
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QColor, QBrush, QPixmap, QImage
from PySide6.QtWidgets import (QApplication, QLabel, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget, QGridLayout, QGraphicsView, QGraphicsScene,
                               QGraphicsRectItem, QGraphicsPixmapItem)

from src.retrieval.src.model.pgsql import Connection, PgGamesRepository
import src.chessboard_localization_temp.main as chess_localization


class FileUploader(QLabel):
    def __init__(self, callback: Callable[[object, str], None] | None = None):
        super().__init__("Drag and Drop a File Here")

        self.file_upload_callback = callback

        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 2px dashed #aaa;")
        self.setAcceptDrops(True)


    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if file_path.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
                pixmap = QPixmap(file_path)
                self.setPixmap(pixmap.scaled(self.size()))

                if self.file_upload_callback is not None:
                    self.file_upload_callback(self, file_path)

            elif file_path.lower().endswith(('mp4', 'avi', 'mov', 'mkv')):
                self.setText("Video support is pending!")

class ChessBoard(QGraphicsView):
    loaded_pixmap: dict[int, QPixmap]
    cell_size: float

    def __init__(self, board: chess.Board):
        super().__init__()

        self.loaded_pixmap = {}

        self.setScene(QGraphicsScene(self))
        self.board_size = 8
        self.board = board
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.draw_board()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.draw_board()

    def draw_board(self):
        scene = self.scene()
        scene.clear()

        size = min(self.width(), self.height())
        self.cell_size = size / self.board_size

        white = QColor("white")
        gray = QColor("gray")
        for row in range(self.board_size):
            for col in range(self.board_size):
                rect = QRectF(col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size)
                color = white if (row + col) % 2 == 0 else gray
                square = QGraphicsRectItem(rect)
                square.setBrush(QBrush(color))
                scene.addItem(square)

                piece = self.board.piece_at(row * 8 + col)
                if piece is not None:
                    self.draw_piece(row, col, piece)

    def draw_piece(self, row: int, col: int, piece: chess.Piece, confidence: float | None = None):
        scene = self.scene()

        piece_id = piece.piece_type + (6 if piece.color else 0)
        if piece_id not in self.loaded_pixmap:
            piece_image = pathlib.Path.cwd() / rf"assets/pieces/{piece_id}.png"
            self.loaded_pixmap[piece_id] = QPixmap(piece_image.as_posix())

        pixmap = self.loaded_pixmap[piece_id].scaled(32, 32)  if self.cell_size <= 60 else self.loaded_pixmap[piece_id]
        qpixmap = QGraphicsPixmapItem(pixmap)
        offset = (self.cell_size - pixmap.width()) / 2
        qpixmap.setPos(col * self.cell_size + offset, row * self.cell_size + offset)
        scene.addItem(qpixmap)

class DataGrid(QTableWidget):
    def __init__(self):
        super().__init__()  # Example: 5 rows, 3 columns

        self.setHorizontalHeaderLabels(["Column 1", "Column 2", "Column 3"])

        # Set up the label to display when the table is empty
        self.empty_label = QLabel("Waiting for image upload")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setStyleSheet("font-size: 20px; font-weight: bold; color: gray;")
        self.empty_label.setVisible(True)  # Initially visible

        # Set up a layout to hold the table and the label
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.empty_label)
        self.setLayout(self.layout)

        # Fill data
        self.update_empty_label()

    def update_empty_label(self):
        # If the table has data, hide the "waiting" label
        if self.rowCount() > 0 and self.columnCount() > 0:
            self.empty_label.setVisible(False)
        else:
            self.empty_label.setVisible(True)

    def set_data(self, data):
        # Populate the table with data, you can modify this method based on how you want to add data
        for row in range(len(data)):
            for col in range(len(data[row])):
                self.setItem(row, col, QTableWidgetItem(data[row][col]))

        self.update_empty_label()

class MainWindow(QWidget):
    @staticmethod
    def on_file_upload(uploader: FileUploader, filename: str):
        (
            rgb_image, 
            corners_list, 
            squares_data_original,
            img
        ) = chess_localization.auto_chessboard_localization(filename)

        height, width, channels = rgb_image.shape
        q_image = QImage(rgb_image.data, width, height, channels * width, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(q_image).scaled(uploader.size())
        uploader.setPixmap(pixmap)

        return corners_list, squares_data_original, img

    piece_mapping = {
        9: chess.PAWN, # white-pawn
        3: chess.PAWN, # black-pawn
        8: chess.KNIGHT, # white-knight
        2: chess.KNIGHT, # black-knight
        0: chess.BISHOP, # white-bishop
        6: chess.BISHOP, # black-bishop
        11: chess.ROOK, # white-rook
        5: chess.ROOK, # black-rook
        10: chess.QUEEN, # white-queen
        4: chess.QUEEN, # black-queen
        7: chess.KING, # white-king
        1: chess.KING, # black-king
    }

    @staticmethod
    def recog(image, squares_data_original):
        model = ultralytics.YOLO(r"D:\Projects\Uni\Chessy3D\src\object_detection_yolo\chess-model-yolov8m.pt")

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
                y_mid = int((y1 + y2) / 2) + 25

                for cell_value, coordinates in coord_dict.items():
                    x_values = [point[0] for point in coordinates]
                    y_values = [point[1] for point in coordinates]

                    if (min(x_values) <= x_mid <= max(x_values)) and (min(y_values) <= y_mid <= max(y_values)):
                        a = int(result.boxes.cls[id])

                        print(f" cell :  {cell_value} --> {a} ")
                        # add cell values and piece cell_value(class value
                        game_list.append([cell_value, a])
                        break

        return game_list, results[0].plot()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chessy 3D")

        # CONST
        PG_CONN = r"host=localhost user=postgres password=password dbname=chessy"

        # globals
        board = chess.Board()
        pgconn = Connection(PG_CONN)
        games_repo = PgGamesRepository(pgconn)

        # Create widgets
        self.chess_board = ChessBoard(board)

        def on_file_upload_callback(uploader: FileUploader, filename: str):
            _, squares_data_original, img = MainWindow.on_file_upload(uploader, filename)
            game_list, result_plot = MainWindow.recog(img, squares_data_original)

            board.clear()
            for (cell, detected_class) in game_list:
                piece = chess.Piece(self.piece_mapping[detected_class], chess.WHITE if detected_class < 6 else chess.BLACK)
                board.set_piece_at(cell - 1, piece)

            height, width, channels = result_plot.shape
            q_image = QImage(result_plot.data, width, height, channels * width, QImage.Format.Format_BGR888)
            pixmap = QPixmap.fromImage(q_image).scaled(uploader.size())
            uploader.setPixmap(pixmap)

            self.chess_board.draw_board()


        self.file_uploader = FileUploader(callback=on_file_upload_callback)
        self.data_grid = DataGrid()

        # Main layout with QGridLayout
        main_layout = QGridLayout()

        # Add widgets to the grid
        main_layout.addWidget(self.file_uploader, 0, 0)
        main_layout.addWidget(self.chess_board, 0, 1)
        main_layout.addWidget(self.data_grid, 1, 0, 1, 2)  # Span across two columns

        # Set row stretch to make the first row twice as large as the second row
        main_layout.setRowStretch(0, 2)
        main_layout.setRowStretch(1, 1)

        # Set column stretch for equal resizing of columns in the first row
        main_layout.setColumnStretch(0, 1)
        main_layout.setColumnStretch(1, 1)

        self.setLayout(main_layout)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1200, 900)
    window.show()
    sys.exit(app.exec())
