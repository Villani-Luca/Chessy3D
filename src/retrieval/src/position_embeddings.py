import math
import re
from abc import ABC
import chess
import numpy as np

##### Interface #####
class PositionEmbedder(ABC):
    EMBEDDING_SIZE: int

    def embedding(self, board: chess.Board) -> np.ndarray:
        pass

    def embedding_move(self, board: chess.Board, move: chess.Move, array: np.ndarray) -> np.ndarray:
        pass

    def create_empty_embedding(self) -> np.ndarray:
        pass

##### Implementations #####

# CONS
PIECE_TO_INDEX = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5
}

class NaivePositionEmbedder(PositionEmbedder):
    def __init__(self):
        self.EMBEDDING_SIZE = 768

    def embedding(self, board: chess.Board) -> np.ndarray:
        arr = self.create_empty_embedding()
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                computed_index = self._compute_index(square, piece)
                arr[computed_index] = 1

        return arr

    @staticmethod
    def _compute_index(square: chess.Square, piece: chess.Piece):
        return square * 6 + PIECE_TO_INDEX[piece.piece_type]

    def embedding_move(self, board: chess.Board, move: chess.Move, array: np.ndarray) -> np.ndarray:
        if array.shape[0] != self.EMBEDDING_SIZE and array.dtype != np.uint8:
            raise ValueError('array should be array of 768 of uint8')

        # ogni 6 posti sono una posizione della scacchiera, ordinati da basso sinistra a alto destra
        piece = board.piece_at(move.to_square)
        from_index = self._compute_index(move.from_square, piece)
        to_index = self._compute_index(move.to_square, piece)

        array[from_index] = 0
        array[to_index] = 1

        return array

    def create_empty_embedding(self):
        return np.zeros(self.EMBEDDING_SIZE, dtype=np.float32)


class AutoencoderPositionEmbedder(PositionEmbedder):
    pass

