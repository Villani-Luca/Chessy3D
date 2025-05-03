import functools
import itertools
import math
import pathlib
import time
from time import sleep

import chess.pgn
import chess.polyglot

import pymilvus as milvus

from src.retrieval.src.milvus import MilvusRepository, MilvusBulkWriter, MilvusSetup
from src.retrieval.src.model.gamesrepository import Connection, SqliteGamesRepository, SqliteMovesRepository

import concurrent.futures
import multiprocessing
import queue

import re

from src.retrieval.src.position_embeddings import PositionEmbedder, NaivePositionEmbedder

ROOT = pathlib.Path.cwd().parent.parent.parent
MINIO_FILES_TEMP = (ROOT / 'minio.txt')
SQLITE_PATH = (ROOT / 'data/retrieval/sqlite.db').resolve()
SAN_MOVE_REGEX = r'(?:\d+\.)?([KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?|O-O(?:-O)?)[+#]?'

##### Concurrent tasks #####
class WorkerOutput:
    """Worker task output"""
    game_id: int

    # [zobrist_hash, embedding]
    moves: dict[str, list[int]]

    def __init__(self, game_id: int):
        self.game_id = game_id
        self.moves = {}

    def has_move(self, move_id: str):
        return move_id in self.moves

    def add_move(self, move_id: str, embedding: list[int]) -> None:
        self.moves[move_id] = embedding

def worker(game_id: int, moves: str, embedder: PositionEmbedder):
    # games ( number in batch, headers list, moves [ (embedding, fen) ] )
    output = WorkerOutput(game_id)

    board = chess.Board()
    for move in re.findall(SAN_MOVE_REGEX, moves):
        board.push_san(move)

        # find if position has been reached before
        move_hash = chess.polyglot.zobrist_hash(board)
        output.add_move(str(move_hash), embedder.embedding(board))

    return output

def worker_callback(q: multiprocessing.Queue, future: concurrent.futures.Future[WorkerOutput]):
    result = future.result()
    q.put(result)

##### Consumer class / Saver class #####
class Consumer:
    # multi threading
    thread_id: int
    q: multiprocessing.Queue
    e: multiprocessing.Event

    # sqlite
    dbconn: Connection
    moves_repo: SqliteMovesRepository

    # milvus
    milvus_writer: MilvusBulkWriter

    # batching handleing
    games_number_batch: int
    sqlite_current_batch: list[tuple[int, str]]
    milvus_current_batch: dict[str, list[int]]
    max_batch_size: int

    # stats
    saved_games: int

    def __init__(self, thread_id: int, q: multiprocessing.Queue, e: multiprocessing.Event):
        self.thread_id = thread_id
        self.q = q
        self.e = e

    def cleanup_batches(self):
        self.sqlite_current_batch.clear()
        self.milvus_current_batch.clear()
        self.games_number_batch = 0

    def save_batch(self) -> None:
        if self.games_number_batch == 0:
            print(f'[SAVER {self.thread_id}] Empty batch')
            return

        start_time = time.time()

        # milvus commit
        self.milvus_writer.append(list(self.milvus_current_batch.items()))
        self.milvus_writer.commit()

        # moves commit
        self.moves_repo.save_moves(self.sqlite_current_batch)
        self.dbconn.commit()

        self.cleanup_batches()

        # stats update
        sqlite_time = time.time() - start_time
        self.saved_games += self.games_number_batch
        print(f'[SAVER {self.thread_id}] SAVED BATCH - {self.saved_games} - {sqlite_time}')


    def saver(self):
        while True:
            try:
                queue_item: WorkerOutput = self.q.get(timeout=5)

                for move_id, embedding in queue_item.moves.items():
                    self.sqlite_current_batch.append((queue_item.game_id, str(move_id)))
                    if move_id not in self.milvus_current_batch:
                        self.milvus_current_batch[move_id] = embedding

                self.games_number_batch += 1

                if len(self.milvus_current_batch) >= self.max_batch_size:
                    self.save_batch()

            except queue.Empty:
                print(f'[SAVER {self.thread_id}] QUEUE EMPTY')
                if self.e.is_set():
                    break

                sleep(0.5)

        self.save_batch()
        self.close()

        # TODO: supporta solo un singolo processore, per farlo multi processore usare una queue o un channel
        with MINIO_FILES_TEMP.open('w') as file:
            for batch in self.milvus_writer.batch_files():
                file.writelines(batch)

        print(f'[SAVER {self.thread_id}] CLOSED')

    def close(self):
        self.dbconn.close()

    def setup(self):
        self.milvus_writer = MilvusBulkWriter()
        self.max_batch_size = 2000

        self.dbconn = Connection(SQLITE_PATH.as_posix())
        self.moves_repo = SqliteMovesRepository(self.dbconn)

        self.sqlite_current_batch = []
        self.milvus_current_batch = {}

        self.games_number_batch = 0
        self.saved_games = 0

        self.saver()


if __name__ == '__main__':
    ##### Dependency injection dei poveri #####
    MilvusSetup.setup_milvus(reset=True)

    sqlite_connection = Connection(SQLITE_PATH.as_posix())
    moves_repo = SqliteMovesRepository(sqlite_connection)
    games_repo = SqliteGamesRepository(sqlite_connection)

    # position encoder
    position_embedder = NaivePositionEmbedder()

    # gestione multi processo
    queue = multiprocessing.Queue()
    event = multiprocessing.Event()

    process = multiprocessing.Process(target=Consumer(0, queue, event).setup)
    process.start()

    starting_id = moves_repo.get_highest_gameid() # recupera il primo id
    current_id = starting_id

    print(f'Starting id {starting_id}')
    with concurrent.futures.ProcessPoolExecutor() as executor:
        while True:
            games = games_repo.get_games_moves(current_id, limit=200)
            if len(games) == 0:
                break

            current_id = games[-1][0]

            if current_id - starting_id % 200000 == 0:
                print(current_id - starting_id)

            for game in games:
                executor.submit(worker, game[0], game[1], position_embedder).add_done_callback(functools.partial(worker_callback, queue))

    print(f'All done - sending event ... waiting saver process join')

    sqlite_connection.close()
    event.set()
    process.join()

    print(f'All done - cleaned up')

