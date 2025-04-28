import chess.pgn
import torch

PIECE_TO_INDEX = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5
}

def board_to_embedding(board: chess.Board) -> list[int]:
    tensor = torch.zeros(8, 8, 6, 2)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            piece_index = PIECE_TO_INDEX[piece.piece_type]
            color_index = 0 if piece.color == chess.WHITE else 1
            tensor[rank, file, piece_index, color_index] = 1

    # TODO: capire in che modo viene flattened ( cosa si trova dove )
    return tensor.flatten().tolist()