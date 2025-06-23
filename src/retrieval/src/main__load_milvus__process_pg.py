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

from src.retrieval.src.milvus import MilvusBulkWriter
from src.retrieval.src.pgsql import Connection, PgGamesRepository, PgMovesRepository
from src.retrieval.src.position_embeddings import PositionEmbedder, NaivePositionEmbedder

ROOT = pathlib.Path.cwd().parent.parent.parent
MINIO_FILES_TEMP = (ROOT / r'minio.txt')
BULK_WRITER_TEMP_PATH = pathlib.Path(r'E:\temp')
PG_CONN = r"host=localhost user=postgres password=password dbname=chessy"

SAN_MOVE_REGEX = r'(?:\d+\.)?([KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?|O-O(?:-O)?)[+#]?'
#STARTING_ID = 22_473_248 # primo id
#N_RECORDS = 15_000_000 # ACTUAL: 14966038
STARTING_ID = 0
#N_RECORDS = 14_600_000
N_RECORDS = 5_000_000
PRODUCERS_NUMBER = 6
CONSUMERS_NUMBER = 5
DISABLE_SQL = False

##### Concurrent tasks #####
class WorkerOutput:
    """Worker task output"""
    game_id: int

    # [zobrist_hash, embedding]
    moves: dict[str, ndarray]

    def __init__(self, game_id: int):
        self.game_id = game_id
        self.moves = {}

    def has_move(self, move_id: str):
        return move_id in self.moves

    def add_move(self, move_id: str, embedding: np.ndarray) -> None:
        self.moves[move_id] = embedding

def worker(
        consumer_queue: multiprocessing.JoinableQueue,
        embedder: PositionEmbedder,
        connstring: str,
        starting_id: int,
        nrecords: int,
        worker_id: int):

    conn = Connection(connstring)
    games_repo = PgGamesRepository(conn)

    current_id = starting_id
    stat_id = starting_id
    maxsentileid = starting_id + nrecords

    print(f'[WORKER] {worker_id} starting {starting_id} {nrecords} {maxsentileid}')

    while True:
        data = games_repo.get_games_moves(current_id, maxsentileid=maxsentileid, limit=500)
        if len(data) == 0:
            break

        current_id = data[-1][0] + 1

        result = []
        delete = []
        for game_id, moves in data:
            output = WorkerOutput(game_id)

            board = chess.Board()
            arr = embedder.create_empty_embedding()
            for san_move in re.findall(SAN_MOVE_REGEX, moves):
                try:
                    move = board.push_san(san_move)

                    # find if position has been reached before
                    move_hash = chess.polyglot.zobrist_hash(board)
                    arr = embedder.embedding_move(board, move, arr)
                    output.add_move(str(move_hash), arr)
                except ValueError as e:
                    print(f'[WORKER {worker_id}] failed {game_id} {move}: {e}')
                    delete.append(game_id)
                    # cleanup database
                    break

            result.append(output)

        if len(delete) > 0:
            conn.begin()
            for game_id in delete:
                games_repo.delete_game(game_id)
            conn.commit()

        if current_id - stat_id >= 25_000:
            print(f'[WORKER {worker_id}] {current_id - starting_id}/{nrecords} {current_id}/{maxsentileid}')
            stat_id = current_id

        consumer_queue.join()
        consumer_queue.put(result)

    conn.close()
    print(f"[WORKER {worker_id}] FINISHED")

##### Consumer class / Saver class #####
class Consumer:
    # multi threading
    thread_id: int
    q: multiprocessing.JoinableQueue
    e: multiprocessing.Event

    # sqlite
    dbconn: Connection
    moves_repo: PgMovesRepository

    # milvus
    milvus_writer: MilvusBulkWriter

    # batching handleing
    games_number_batch: int
    sqlite_current_batch: list[tuple[int, str]]
    milvus_current_batch: dict[str, np.ndarray]
    max_batch_size: int

    # stats
    last_saved_games: int
    saved_games: int
    sqlite_time: float
    milvus_time: float
    saved_batches: int

    def __init__(self, thread_id: int, q: multiprocessing.JoinableQueue, e: multiprocessing.Event):
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

        self.saved_batches += 1
        start_time = time.time()

        # milvus commit
        self.milvus_writer.append(list(self.milvus_current_batch.items()))

        if self.saved_batches % 400 == 0:
            self.milvus_writer.commit()

        milvus_time = time.time()

        # moves commit
        while not DISABLE_SQL:
            try:
                self.dbconn.begin()
                self.moves_repo.save_moves(self.sqlite_current_batch)
                self.dbconn.commit()
                break
            except Exception as e:
                self.dbconn.rollback()
                print(f'[SAVER {self.thread_id}] commit error - rollback', e)
                sleep(3)

        sqlite_time = time.time()

        # stats update
        self.saved_games += self.games_number_batch
        self.sqlite_time += sqlite_time - milvus_time
        self.milvus_time += milvus_time - start_time

        #if self.saved_games - self.last_saved_games >= self.max_batch_size * 40:
        print(f'[SAVER {self.thread_id}] SAVED {self.saved_games} - CURRENT BATCH: SQL {self.sqlite_time} MILVUS: {self.milvus_time}')
        self.last_saved_games = self.saved_games
        self.sqlite_time = 0
        self.milvus_time = 0

        self.cleanup_batches()


    def saver(self):
        while True:
            try:
                items: list[WorkerOutput] = self.q.get(timeout=5)
                self.q.task_done()

                for queue_item in items:
                    for move_id, embedding in queue_item.moves.items():
                        self.sqlite_current_batch.append((queue_item.game_id, move_id))
                        if move_id not in self.milvus_current_batch:
                            self.milvus_current_batch[str(move_id)] = embedding

                    self.games_number_batch += 1
                    if self.games_number_batch >= self.max_batch_size:
                        self.save_batch()


            except queue.Empty:
                print(f'[SAVER {self.thread_id}] QUEUE EMPTY')
                if self.e.is_set():
                    break

                sleep(0.5)

        self.save_batch()
        self.close()

        with (MINIO_FILES_TEMP / f"_{self.thread_id}").open('w') as file:
            for batch in self.milvus_writer.batch_files():
                file.writelines(batch)

        print(f'[SAVER {self.thread_id}] CLOSED')

    def close(self):
        if self.dbconn is not None:
            self.dbconn.close()

    def setup(self):
        self.milvus_writer = MilvusBulkWriter(local_path=BULK_WRITER_TEMP_PATH.as_posix())
        self.max_batch_size = 2_500

        if not DISABLE_SQL:
            self.dbconn = Connection(PG_CONN)
            self.moves_repo = PgMovesRepository(self.dbconn)

        self.sqlite_current_batch = []
        self.milvus_current_batch = {}

        self.games_number_batch = 0
        self.last_saved_games = 0
        self.saved_games = 0
        self.sqlite_time = 0
        self.milvus_time = 0
        self.saved_batches = 0

        self.saver()


if __name__ == '__main__':
    ##### Dependency injection dei poveri #####
    # MilvusSetup.setup_milvus(reset=False)

    # position encoder
    position_embedder = NaivePositionEmbedder()

    # gestione multi processo
    queue = multiprocessing.JoinableQueue()
    event = multiprocessing.Event()

    consumers = [multiprocessing.Process(target=Consumer(i, queue, event).setup) for i in range(CONSUMERS_NUMBER)]

    producer_batch_size = math.ceil(N_RECORDS / PRODUCERS_NUMBER)
    producers = [multiprocessing.Process(
        target=worker,
        args=(
            queue,
            position_embedder,
            PG_CONN,
            STARTING_ID + math.ceil(i*producer_batch_size),
            producer_batch_size,
            i)
    ) for i in range(PRODUCERS_NUMBER)]

    for producer in producers:
        producer.start()

    for consumer in consumers:
        consumer.start()

    for producer in producers:
        producer.join()

    event.set()
    for consumer in consumers:
        consumer.join()

    print(f'All done - cleaned up')

