from abc import ABC
import chess
import torch

##### Interface #####
class PositionEmbedder(ABC):
    def embedding(self, board: chess.Board) -> list[int]:
        pass

##### Implementations #####

# CONST
PIECE_TO_INDEX = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5
}

class NaivePositionEmbedder(PositionEmbedder):
    def embedding(self, board: chess.Board) -> list[int]:
        tensor = torch.zeros(8, 8, 12)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                piece_index = PIECE_TO_INDEX[piece.piece_type] + 6 if piece.color == chess.BLACK else 0
                tensor[rank, file, piece_index] = 1

        # TODO: capire in che modo viene flattened ( cosa si trova dove )
        return tensor.flatten().tolist()

class AutoencoderPositionEmbedder(PositionEmbedder):
    pass

