import pathlib

import chess
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QPixmap, QColor, QBrush
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsPixmapItem, QWidget, \
    QVBoxLayout, QHBoxLayout, QPushButton


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

                position = (7 - row) * 8 + col
                piece = self.board.piece_at(position)
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

    def rotate_board_left(self):
        new_board = chess.Board(None)
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                row, col = divmod(square, 8)
                new_row, new_col = 7 - col, row
                new_square = new_row * 8 + new_col
                new_board.set_piece_at(new_square, piece)
        self.board = new_board
        self.draw_board()

    def rotate_board_right(self):
        new_board = chess.Board(None)
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                row, col = divmod(square, 8)
                new_row, new_col = col, 7 - row
                new_square = new_row * 8 + new_col
                new_board.set_piece_at(new_square, piece)
        self.board = new_board
        self.draw_board()

class ChessBoardWidget(QWidget):
    def __init__(self, board: ChessBoard):
        super().__init__()

        layout = QVBoxLayout(self)
        self.chess_view = board
        layout.addWidget(self.chess_view)

        button_layout = QHBoxLayout()
        self.rotate_left_btn = QPushButton("⟲")
        self.rotate_right_btn = QPushButton("⟳")
        self.rotate_left_btn.clicked.connect(self.chess_view.rotate_board_left)
        self.rotate_right_btn.clicked.connect(self.chess_view.rotate_board_right)

        button_layout.addWidget(self.rotate_left_btn)
        button_layout.addWidget(self.rotate_right_btn)
        layout.addLayout(button_layout)

    def draw_board(self):
        self.chess_view.draw_board()