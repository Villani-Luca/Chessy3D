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
for san_move in re.findall(SAN_MOVE_REGEX, r" 1.e4 d6 2.d4 Nf6 3.Nc3 Nbd7 4.Be3 e5 5.Nf3 Be7 6.Be2 O-O 7.O-O c6 8.h3 exd4 9.Qxd4 Nc5 10.Rad1 a5 11.e5 dxe5 12.Qxe5 Ncd7 13.Qh2 Bb4 14.Na4 Nd5 15.Bg5 Qe8 16.Bc4 b5 17.Bxd5 cxd5 18.Nc3 Bxc3 19.bxc3 Bb7 20.Rfe1 Qc8 21. Re7 Re8 22.Rde1 Rxe7 23.Rxe7 h6 24.Qf4 Qxc3 25.Qxf7+ Kh7 26.Rxd7 Bc6 27. Re7 hxg5 28.Nxg5+ Kh8 29.Qh5+ Kg8 30.Qh7+ Kf8 31.Rf7+ 1-0"):
    move = board.push_san(san_move)

    # find if position has been reached before
    id = chess.polyglot.zobrist_hash(board)
    arr = embedder.embedding_move(board, move, arr)
    print(id if id <= 9223372036854775807 else id - 18446744073709551616)
    print(f"{''.join(arr.astype(int).astype(str))}")
    print()
    print(board)
    print()
    print()
