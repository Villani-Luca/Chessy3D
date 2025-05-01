import functools
import itertools
import math
import pathlib
import time
from time import sleep

import chess.pgn
import chess.polyglot

import pymilvus as milvus

import helper
from src.retrieval.model.gamesrepository import Connection, SqliteGamesRepository, SqliteMovesRepository

import concurrent.futures
import multiprocessing
import queue

import re
import asyncio

ROOT = pathlib.Path.cwd().parent.parent.parent
MILVUS_PATH = (ROOT / 'data/retrieval/milvus.db').resolve()
SQLITE_PATH = (ROOT / 'data/retrieval/sqlite.db').resolve()
SAN_MOVE_REGEX = r'(?:\d+\.)?([KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?|O-O(?:-O)?)[+#]?'


class WorkerOutput:
    gameid: int

    # [zobrist_hash, embedding]
    moves: dict[str, list[int]]

    def __init__(self, gameid: int):
        self.gameid = gameid
        self.moves = {}

    def has_move(self, moveid: str):
        return moveid in self.moves

    def add_move(self, moveid: str, embedding: list[int]) -> None:
        self.moves[moveid] = embedding

def worker(gameid: int, moves: str):
    # games ( number in batch, headers list, moves [ (embedding, fen) ] )
    output = WorkerOutput(gameid)

    board = chess.Board()
    for move in re.findall(SAN_MOVE_REGEX, moves):
        board.push_san(move)

        # find if position has been reached before
        move_hash = str(chess.polyglot.zobrist_hash(board))
        output.add_move(move_hash, helper.board_to_embedding(board))

    return output

def worker_callback(q: multiprocessing.Queue, future: concurrent.futures.Future[WorkerOutput]):
    result = future.result()
    q.put(result)

class Consumer:
    thread_id: int
    q: multiprocessing.Queue
    e: multiprocessing.Event

    sqlite_connection: Connection
    moves_repo: SqliteMovesRepository

    milvus_client: milvus.MilvusClient

    games_number_batch: int
    sqlite_current_batch: list[tuple[int, str]]
    chroma_current_batch: dict[str, list[int]]

    saved_games: int

    def __init__(self, thread_id:int, q: multiprocessing.Queue, e: multiprocessing.Event):
        self.thread_id = thread_id
        self.q = q
        self.e = e

    def get_split_chroma_batch(self, already_in_keys: list[str]):


        return itertools.batched(self.chroma_current_batch.items(), self.chroma_max_batch_size)

    def cleanup_batches(self):
        self.sqlite_current_batch.clear()
        self.chroma_current_batch.clear()
        self.games_number_batch = 0

    async def save_chroma_batch(self):
        already_in = await self.chroma_embeddings.get(ids=[x[0] for x in self.chroma_current_batch])
        already_in_keys = already_in.keys()

        await asyncio.gather(*[self.chroma_embeddings.add(
            ids=[x[0] for x in batch ],
            embeddings=[x[1] for x in batch])
            for batch in self.get_split_chroma_batch(already_in_keys)
        ])


    async def save_batch(self) -> None:
        if self.games_number_batch == 0:
            print(f'[SAVER {self.thread_id}] Empty batch')
            return

        start_time = time.time()

        chroma_request = self.save_chroma_batch()
        self.moves_repo.save_moves(self.sqlite_current_batch)

        self.sqlite_connection.commit()
        await chroma_request

        sqlite_time = time.time() - start_time
        self.saved_games += self.games_number_batch

        self.cleanup_batches()

        print(f'[SAVER {self.thread_id}] SAVED BATCH - {self.saved_games} - {sqlite_time}')


    async def saver(self):
        while True:
            try:
                queue_item: WorkerOutput = self.q.get(timeout=3)

                for move_id, embedding in queue_item.moves.items():
                    self.sqlite_current_batch.append((queue_item.gameid, move_id))
                    if move_id not in self.chroma_current_batch:
                        self.chroma_current_batch[move_id] = embedding

                self.games_number_batch += 1

                if len(self.chroma_current_batch) >= self.chroma_max_batch_size:
                    await self.save_batch()

            except queue.Empty:
                print(f'[SAVER {self.thread_id}] QUEUE EMPTY')
                if self.e.is_set():
                    break

                sleep(0.1)

        await self.save_batch()
        self.sqlite_connection.close()

        print(f'[SAVER {self.thread_id}] CLOSED')

    async def setup(self):
        self.milvus_client = milvus.MilvusClient('')

        self.chroma_client = await chromadb.AsyncHttpClient(host=CHROMA_PATH[0], port=CHROMA_PATH[1])

        self.chroma_embeddings = await self.chroma_client.create_collection("embeddings", get_or_create=True)
        self.chroma_max_batch_size = await self.chroma_client.get_max_batch_size()

        self.sqlite_connection = Connection(SQLITE_PATH.as_posix())
        self.moves_repo = SqliteMovesRepository(self.sqlite_connection)

        self.sqlite_current_batch = []
        self.chroma_current_batch = {}

        self.games_number_batch = 0
        self.saved_games = 0

        return await self.saver()

    def process(self):
        asyncio.run(self.setup())

if __name__ == '__main__':
    queue = multiprocessing.Queue()
    event = multiprocessing.Event()

    consumer_processes = [multiprocessing.Process(target=Consumer(i, queue, event).process) for i in range(1)]
    for process in consumer_processes:
        process.start()

    sqlite_connection = Connection(SQLITE_PATH.as_posix())
    moves_repo = SqliteMovesRepository(sqlite_connection)
    games_repo = SqliteGamesRepository(sqlite_connection)

    #starting_id = games_repo.get_games_moves(0, limit=1)[0][0] # recupera il primo id
    starting_id = moves_repo.get_highest_gameid() # recupera il primo id
    current_id = starting_id

    print(f'Starting id {starting_id}')
    with concurrent.futures.ProcessPoolExecutor(math.floor(multiprocessing.cpu_count() / 2)) as executor:
        while current_id - starting_id < 100000:
            games = games_repo.get_games_moves(current_id, limit=100)
            current_id = games[-1][0]

            print(f'RET {current_id}')

            for game in games:
                executor.submit(worker, game[0], game[1]).add_done_callback(functools.partial(worker_callback, queue))

    sqlite_connection.close()

    print(f'All done - sending event ... waiting saver process join')

    event.set()
    for process in consumer_processes:
        process.join()

    print(f'All done - cleaned up')

