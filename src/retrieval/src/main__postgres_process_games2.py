import math
import multiprocessing
import pathlib
import queue
import re
import time
from time import sleep

import chess.pgn
import chess.polyglot
import numpy as np
from numpy import ndarray
from pgvector import Bit
import sys
sys.path.extend(r"E:\projects\uni\Chessy3D\src")

from src.retrieval.src.model.pgsql import Connection, PgGamesRepository, PgMovesRepository
from src.retrieval.src.position_embeddings import PositionEmbedder, NaivePositionEmbedder


ROOT = pathlib.Path.cwd().parent.parent.parent
PG_CONN = r"host=localhost user=postgres password=password dbname=chessy"

SAN_MOVE_REGEX = r'(?:\d+\.)?([KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?|O-O(?:-O)?)[+#]?'
STARTING_ID = 0
N_RECORDS = 1_000_000
PRODUCERS_NUMBER = 6
CONSUMERS_NUMBER = 5
DISABLE_SQL = False

board = chess.Board()
embedder = NaivePositionEmbedder()
arr = embedder.embedding(board)
for san_move in re.findall(SAN_MOVE_REGEX, r"  1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 4.Ba4 Nf6 5.O-O Be7 6.Re1 O-O 7.Bxc6 dxc6 8. Nxe5 Bc5 9.c3 Re8 10.d4 Bd6 11.f4 c5 12.h3 cxd4 13.cxd4 c5 14.Nc3 cxd4 15. Qxd4 Qe7 16.Nd3 Bb8 17.Be3 Rd8 18.Qc5 Qe8 19.Rad1 Ba7 20.Qe5 Bxe3+ 21.Rxe3 Be6 22.Ree1 Bc4 23.Qxe8+ Rxe8 24.Ne5 Be6 25.Nf3 Rac8 26.a3 Bb3 27.Rd6 Nh5 28.Rb6 Nxf4 29.Rxb3 b5 30.e5 Nd3 31.Re3 Red8 32.Ne4 Rc1+ 33.Kh2 Nc5 34. Nxc5 Rxc5 35.Rbc3 Rcd5 36.Rc7 h6 37.Ra7 Rd3 38.Rxd3 Rxd3 39.Rxa6 Rb3 40. Ra8+ Kh7 41.Rb8 Kg6 42.Nd4 Rxb2 43.Rxb5 Ra2 44.Ra5 h5 45.a4 Kh6 46.Nf3 g5 47.Ra6+ Kg7 48.Nxg5 Re2 49.Ra7 Rxe5 50.Nxf7 Re2 51.a5 Kf6 52.a6 Rb2 53.Nd6 Ra2 54.Nb5 Ke6 55.Ra8 Kd7 56.a7 Kc6 57.Rb8 Ra6 58.a8=Q+ Rxa8 59.Rxa8 Kxb5 60.Rh8 h4 61.Rh5+ Kc4 62.Rxh4+ Kd5 63.g4 Ke6 64.Kg3 Kf6 65.Rh5 Kg6 66.Rf5 Kg7 67.h4 Kg6 68.h5+ Kg7 69.Kh4 1-0"):
    move = board.push_san(san_move)

    # find if position has been reached before
    id = chess.polyglot.zobrist_hash(board)
    arr = embedder.embedding_move(board, move, arr)
    unembed = embedder.unembedding(arr)
    print(id if id <= 9223372036854775807 else id - 18446744073709551616)
    print(f"{''.join(arr.astype(int).astype(str))}")
    print()
    print(board)
    print(str(board) == str(unembed))
    print(unembed)
    print()
    print()
