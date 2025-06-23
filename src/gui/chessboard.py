import pathlib

import chess
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QPixmap, QColor, QBrush
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsPixmapItem, QWidget, \
    QVBoxLayout, QHBoxLayout, QPushButton, QGraphicsSimpleTextItem, QLineEdit, QApplication, QStyle, QLabel

from src.retrieval.src.position_embeddings import PositionEmbedder


class ChessBoard(QGraphicsView):
    loaded_pixmap: dict[int, QPixmap]
    cell_size: float

    def __init__(self, board: chess.Board, debug = False):
        super().__init__()

        self.debug = debug
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

    def draw_board(self, board = None):
        if board is not None:
            self.board = board

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

                if self.debug:
                    label = f"{position + 1}"
                    text_item = QGraphicsSimpleTextItem(label)
                    text_item.setBrush(QBrush("black"))
                    text_item.setZValue(1)  # make sure it's on top of the board
                    # Position below the last row
                    text_item.setPos(col * self.cell_size + 2, row * self.cell_size + 2)
                    scene.addItem(text_item)

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
        self.draw_board(new_board)

    def rotate_board_right(self):
        new_board = chess.Board(None)
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                row, col = divmod(square, 8)
                new_row, new_col = col, 7 - row
                new_square = new_row * 8 + new_col
                new_board.set_piece_at(new_square, piece)
        self.draw_board(new_board)

    def retrieve_embedding(self, embedder: PositionEmbedder):
        return embedder.embedding(self.board)

    def get_fen(self):
        return self.board.fen()

class ChessBoardWidget(QWidget):
    def __init__(self, board: ChessBoard):
        super().__init__()

        self.board = board
        self.relative_board = board

        layout = QVBoxLayout(self)

        self.info_label = QLabel()
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        #self.__reset_relative_board()
        layout.addWidget(self.info_label)

        self.chess_view = board
        layout.addWidget(self.chess_view)

        button_layout = QHBoxLayout()
        self.rotate_left_btn = QPushButton("⟲")
        self.rotate_right_btn = QPushButton("⟳")
        self.rotate_left_btn.clicked.connect(self.chess_view.rotate_board_left)
        self.rotate_right_btn.clicked.connect(self.chess_view.rotate_board_right)

        self.fen_textbox = QLineEdit()
        self.fen_textbox.setReadOnly(True)
        self.update_fen_textbox()

        self.copy_fen_btn = QPushButton()
        self.copy_fen_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogStart))
        self.copy_fen_btn.clicked.connect(self.copy_fen_to_clipboard)

        self.reset_relative_board_btn = QPushButton()
        self.reset_relative_board_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowBack))
        self.reset_relative_board_btn.clicked.connect(self.__reset_relative_board)

        button_layout.addWidget(self.rotate_left_btn)
        button_layout.addWidget(self.rotate_right_btn)
        button_layout.addWidget(self.fen_textbox)
        button_layout.addWidget(self.copy_fen_btn)
        button_layout.addWidget(self.reset_relative_board_btn)
        layout.addLayout(button_layout)

    def draw_board(self, board = None):
        self.board = board
        self.chess_view.draw_board(self.board)
        self.update_fen_textbox()

    def retrieve_embedding(self, embedder: PositionEmbedder):
        return self.chess_view.retrieve_embedding(embedder)

    def update_fen_textbox(self):
        self.fen_textbox.setText(self.chess_view.get_fen())

    def copy_fen_to_clipboard(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.fen_textbox.text())

    def __reset_relative_board(self):
        self.show_relative_board(None, "Unknown", "Unknown", "Unknown")

    def show_relative_board(self, board: chess.Board|None, event: str, white_player: str, black_player: str):
        # Update the informational label
        visible = board is not None

        self.relative_board = board
        self.reset_relative_board_btn.setVisible(visible)
        if visible:
            self.info_label.setText(f"""
                <h3>{event}</h3>
                <p>{white_player} vs {black_player}</p>
            """)
        else:
            self.info_label.setText(f"")