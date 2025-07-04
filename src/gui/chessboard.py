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

    def draw_board(self, board = None, differences = None, confidences = None):
        differences = differences if differences is not None else []

        if board is not None:
            self.board = board

        scene = self.scene()
        scene.clear()
        # QApplication.processEvents()

        size = min(self.width(), self.height())
        self.cell_size = size / self.board_size

        white = QColor("white")
        gray = QColor("gray")
        highlight_brush = QBrush(QColor(255, 255, 0, 128))
        for row in range(self.board_size):
            for col in range(self.board_size):
                rect = QRectF(col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size)
                color = white if (row + col) % 2 == 0 else gray
                square = QGraphicsRectItem(rect)
                square.setBrush(QBrush(color))
                scene.addItem(square)

                position = (7 - row) * 8 + col
                if position in differences:
                    highlight = QGraphicsRectItem(rect)
                    highlight.setBrush(highlight_brush)
                    highlight.setPen(Qt.PenStyle.NoPen)  # Remove the border for the highlight
                    highlight.setZValue(1)  # Ensure it overlays the base square
                    scene.addItem(highlight)

                if self.debug:
                    label = f"{position + 1}"
                    text_item = QGraphicsSimpleTextItem(label)
                    text_item.setBrush(QBrush("black"))
                    text_item.setZValue(2)  # make sure it's on top of the board
                    # Position below the last row
                    text_item.setPos(col * self.cell_size + 2, row * self.cell_size + 2)
                    scene.addItem(text_item)

                piece = self.board.piece_at(position)
                if piece is not None:
                    self.draw_piece(row, col, piece)

                    if confidences is not None:
                        confidence = confidences[position]
                        confidence_text = f"{confidence:.2f}"
                        text_item = QGraphicsSimpleTextItem(confidence_text)
                        text_item.setBrush(QBrush("red"))
                        text_item.setZValue(2)
                        # Position near the right edge of the cell
                        text_item.setPos(
                            col * self.cell_size + self.cell_size - 25,  # 25 px offset from the right
                            row * self.cell_size + 2  # small top padding
                        )
                        scene.addItem(text_item)




    def draw_piece(self, row: int, col: int, piece: chess.Piece):
        scene = self.scene()

        piece_id = piece.piece_type + (0 if piece.color else 6)
        if piece_id not in self.loaded_pixmap:
            piece_image = pathlib.Path.cwd() / rf"assets/pieces/{piece_id}.png"
            self.loaded_pixmap[piece_id] = QPixmap(piece_image.as_posix())

        pixmap = self.loaded_pixmap[piece_id].scaled(32, 32)  if self.cell_size <= 60 else self.loaded_pixmap[piece_id]
        qpixmap = QGraphicsPixmapItem(pixmap)
        offset = (self.cell_size - pixmap.width()) / 2
        qpixmap.setPos(col * self.cell_size + offset, row * self.cell_size + offset)
        qpixmap.setZValue(2)
        scene.addItem(qpixmap)

class ChessBoardWidget(QWidget):
    def __init__(self, board: chess.Board, debug=False):
        super().__init__()

        self.board = board
        self.saved_board = None
        self.confidences = None
        self.show_confidence = False

        layout = QVBoxLayout(self)

        self.info_label = QLabel()
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.info_label)

        self.chess_view = ChessBoard(board, debug=debug)
        layout.addWidget(self.chess_view)

        button_layout = QHBoxLayout()
        self.rotate_left_btn = QPushButton("⟲")
        self.rotate_right_btn = QPushButton("⟳")
        self.rotate_left_btn.clicked.connect(self.rotate_board_left)
        self.rotate_right_btn.clicked.connect(self.rotate_board_right)

        self.fen_textbox = QLineEdit()
        self.fen_textbox.setReadOnly(True)
        self.update_fen_textbox()

        self.copy_fen_btn = QPushButton()
        self.copy_fen_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogStart))
        self.copy_fen_btn.clicked.connect(self.copy_fen_to_clipboard)

        self.reset_relative_board_btn = QPushButton()
        self.reset_relative_board_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowBack))
        self.reset_relative_board_btn.clicked.connect(self.reset_relative_board)
        self.reset_relative_board_btn.setVisible(False)

        self.toggle_confidence_btn = QPushButton("Show Confidence")
        self.toggle_confidence_btn.setCheckable(True)
        self.toggle_confidence_btn.toggled.connect(self.toggle_confidence_display)

        button_layout.addWidget(self.rotate_left_btn)
        button_layout.addWidget(self.rotate_right_btn)
        button_layout.addWidget(self.fen_textbox)
        button_layout.addWidget(self.copy_fen_btn)
        button_layout.addWidget(self.reset_relative_board_btn)
        button_layout.addWidget(self.toggle_confidence_btn)

        layout.addLayout(button_layout)

    def toggle_confidence_display(self, checked):
        self.show_confidence = checked
        self.toggle_confidence_btn.setText("Hide Confidence" if checked else "Show Confidence")
        self.chess_view.draw_board(
            self.board,
            confidences=self.confidences if self.show_confidence else None
        )

    def __rotate_board(self, right=False):
        new_board = chess.Board(None)
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                row, col = divmod(square, 8)
                new_row, new_col = (col, 7 - row) if right else (7 - col, row)
                new_square = new_row * 8 + new_col
                new_board.set_piece_at(new_square, piece)
        return new_board

    def rotate_board_left(self):
        self.board = self.__rotate_board(right=False)
        self.chess_view.draw_board(self.board)
        self.update_fen_textbox()

    def rotate_board_right(self):
        self.board = self.__rotate_board(right=True)
        self.chess_view.draw_board(self.board)
        self.update_fen_textbox()

    def draw_board(self, board = None, confidences = None):
        self.board = board or self.board
        self.confidences = confidences
        self.update_fen_textbox()
        self.reset_relative_board()
        self.chess_view.draw_board(self.board, confidences=self.confidences if self.show_confidence else None)

    def retrieve_embedding(self, embedder: PositionEmbedder):
        return embedder.embedding(self.board)

    def update_fen_textbox(self):
        self.fen_textbox.setText(self.board.fen())

    def copy_fen_to_clipboard(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.fen_textbox.text())

    def reset_relative_board(self):
        if self.saved_board is None:
            return

        self.board = self.saved_board
        self.saved_board = None
        self.reset_relative_board_btn.setVisible(False)
        self.info_label.setText(f"")

        self.update_fen_textbox()
        self.rotate_left_btn.setVisible(True)
        self.rotate_right_btn.setVisible(True)
        self.toggle_confidence_btn.setVisible(True)
        self.chess_view.draw_board(self.board)

    def show_relative_board(self, board: chess.Board, event: str, white_player: str, black_player: str):
        # Update the informational label
        if self.saved_board is None:
            self.saved_board = self.board

        self.board = board
        self.reset_relative_board_btn.setVisible(True)
        self.info_label.setText(f"""
            <h3>{event}</h3>
            <p>{white_player} vs {black_player}</p>
        """)

        differences = [x for x in chess.SQUARES if self.saved_board.piece_at(x) != self.board.piece_at(x)]

        # sinceramente questa é magia nera, non ho capito veramente come pyside6 funziona
        # ma a quanto pare alcune cose sono aggiornate in momenti diversi rispetto a quello che dice il codice
        # aspettare gli eventi in questo punto permette alla griglia ( con highlights ) di essere renderizzata correttamente
        QApplication.processEvents()

        print("Differences at squares", differences)
        self.chess_view.draw_board(self.board, differences)

        self.update_fen_textbox()
        self.rotate_left_btn.setVisible(False)
        self.rotate_right_btn.setVisible(False)
        self.toggle_confidence_btn.setVisible(False)