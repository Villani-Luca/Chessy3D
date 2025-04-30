import functools
import itertools
import math
import pathlib
import time
from time import sleep

import chess.pgn
import chess.polyglot

import chromadb

import helper
from src.retrieval.model.gamesrepository import Connection, SqliteGamesRepository, SqliteMovesRepository

import concurrent.futures
import multiprocessing
import queue

import re

ROOT = pathlib.Path.cwd().parent.parent.parent
CHROMA_PATH = (ROOT / 'data/retrieval/chroma').resolve()
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
    print(f'[PROCESSED] {result.gameid} - {len(result.moves)} moves')
    q.put(result)

def saver(q: multiprocessing.Queue, e: multiprocessing.Event):
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH.as_posix(), settings=chromadb.Settings(allow_reset=False))
    #chroma_client.reset()

    chroma_embeddings = chroma_client.create_collection("embeddings", get_or_create=True)
    chroma_max_batch_size = chroma_client.get_max_batch_size()

    sqlite_connection = Connection(SQLITE_PATH.as_posix())
    moves_repo = SqliteMovesRepository(sqlite_connection)

    sqlite_current_batch = []
    chroma_current_batch = {}

    while True:
        try:
            queue_item: WorkerOutput = q.get(timeout=10)

            for moveid, embedding in queue_item.moves.items():
                sqlite_current_batch.append((queue_item.gameid, moveid))
                if moveid not in chroma_current_batch:
                    chroma_current_batch[moveid] = embedding

            if len(chroma_current_batch) >= chroma_max_batch_size:
                start_time = time.time()
                for batch in itertools.batched(chroma_current_batch.items(), chroma_max_batch_size):
                    chroma_embeddings.add(ids=[x[0] for x in batch], embeddings=[x[1] for x in batch])

                chroma_time = time.time() - start_time

                moves_repo.save_moves(sqlite_current_batch)
                sqlite_connection.commit()

                sqlite_time = time.time() - chroma_time - start_time

                sqlite_current_batch.clear()
                chroma_current_batch.clear()
                print(f'[SAVER] SAVED BATCH - CHROMA {chroma_time} SQLITE {sqlite_time}')

        except queue.Empty:
            print('[SAVER] QUEUE EMPTY')
            if e.is_set():
                break

            sleep(0.1)


    start_time = time.time()
    for batch in itertools.batched(chroma_current_batch.items(), chroma_max_batch_size):
        chroma_embeddings.add(ids=[x[0] for x in batch], embeddings=[x[1] for x in batch])

    chroma_time = time.time() - start_time

    moves_repo.save_moves(sqlite_current_batch)
    sqlite_connection.commit()

    sqlite_time = time.time() - chroma_time - start_time

    sqlite_current_batch.clear()
    chroma_current_batch.clear()
    print(f'[SAVER] SAVED BATCH - CHROMA {chroma_time} SQLITE {sqlite_time}')

    sqlite_connection.close()
    print(f'[SAVER] CLOSED')


if __name__ == '__main__':
    queue = multiprocessing.Queue()
    event = multiprocessing.Event()

    sqlite_connection = Connection(SQLITE_PATH.as_posix())
    moves_repo = SqliteMovesRepository(sqlite_connection)
    games_repo = SqliteGamesRepository(sqlite_connection)

    saver_process = multiprocessing.Process(target=saver, args=(queue, event))
    saver_process.start()

    #starting_id = games_repo.get_games_moves(0, limit=1)[0][0] # recupera il primo id
    starting_id = moves_repo.get_highest_gameid() # recupera il primo id
    id = starting_id
    with concurrent.futures.ProcessPoolExecutor(math.floor(multiprocessing.cpu_count() / 2)) as executor:
        while id - starting_id < 1000:
            games = games_repo.get_games_moves(id, limit=100)
            id = games[-1][0]

            print(f'RET {id}')

            for game in games:
                executor.submit(worker, game[0], game[1]).add_done_callback(functools.partial(worker_callback, queue))

    print(f'All done - sending event ... waiting saver process join')
    event.set()
    saver_process.join()

    sqlite_connection.close()
