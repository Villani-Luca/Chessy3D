import datetime
import functools
import hashlib
import itertools
import math
import pathlib
from threading import Event
from typing import Tuple

import chess.pgn
import chess.polyglot

import chromadb
import pandas as pd

import helper
from src.retrieval.model.game import Game
from src.retrieval.model.gamesrepository import Connection, SqliteGamesRepository

import concurrent.futures
import multiprocessing
import queue

from io import StringIO

ROOT = pathlib.Path.cwd().parent.parent.parent
CHROMA_PATH = (ROOT / 'data/retrieval/chroma').resolve()
SQLITE_PATH = (ROOT / 'data/retrieval/sqlite.db').resolve()
SQLITE_MIGRATION_PATH = (ROOT / 'src/retrieval/migrations/init.sql').resolve()

PGN_FOLDER = (ROOT / "data/retrieval/lumbrasgigabase/splitted").resolve()

PGN_HEADERS = [
    'Event',
    'Site',
    'Date',
    'Round',
    'White',
    'Black',
    'Result',
    'ResultDecimal',
    'WhiteTitle',
    'BlackTitle',
    'WhiteElo',
    'BlackElo',
    'ECO',
    'Opening',
    'Variation',
    'WhiteFideId',
    'BlackFideId',
    'EventDate',
    'Annotator',
    'PlyCount',
    'TimeControl',
    'Time',
    'Termination',
    'Mode',
    'FEN',
    'SetUp',
    'Moves',
    'Embedding'
]

class WorkerOutput:
    file: str
    starting_game_id: int

    # [(game_number = index, headers, [moves id])]
    games: list[Tuple[int, chess.pgn.Headers, list[int]]]

    # [zobrist_hash, embedding]
    moves: dict[int, list[int]]

    def __init__(self, file: str, starting_game_id: int):
        self.file = file
        self.starting_game_id = starting_game_id
        self.games = []
        self.moves = {}

    def add_game(self, game_index: int, headers: chess.pgn.Headers) -> None:
        self.games.append((game_index + self.starting_game_id, headers, []))

    def game_add_move(self, game_index: int, moveid: int, embedding: list[int]) -> None:
        if len(self.games) <= game_index:
            return

        # hashing moves
        saved_embedding = self.moves.get(moveid)
        if saved_embedding is None:
            self.moves[moveid] = embedding

        self.games[game_index][2].append(moveid)

def worker(pngfile: pathlib.Path):
    # games ( number in batch, headers list, moves [ (embedding, fen) ] )
    output = WorkerOutput(pngfile.name, int(pngfile.name.split('_')[1]))

    game_n = -1

    f = StringIO(pngfile.read_text())
    while True:
        game_n = game_n + 1
        game = chess.pgn.read_game(f)

        ## If there are no more games, exit the loop
        if game is None:
            break

        # extract metadata
        # play move by move
        #   for each move create embedding of position
        output.add_game(game_n, game.headers)

        if game_n % 500 == 0:
            print(f"{output.file} - Processing game {game_n}")

        board = game.board()
        for move in game.mainline_moves():
            board.push(move)

            # find if position has been reached before
            zobrist_hash = chess.polyglot.zobrist_hash(board)
            output.game_add_move(game_n, zobrist_hash, helper.board_to_embedding(board))

    return output

def worker_callback(q: multiprocessing.Queue, future: concurrent.futures.Future[WorkerOutput]):
    result = future.result()
    print(f'Processed {result.file} - {len(result.games)} and {len(result.moves)} moves')
    queue.put(result)


def saver(q: multiprocessing.Queue, event: multiprocessing.Event):
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH.as_posix(), settings=chromadb.Settings(allow_reset=True))
    chroma_client.reset()
    chroma_embeddings = chroma_client.create_collection("embeddings", get_or_create=True)

    sqlite_connection = Connection(SQLITE_PATH.as_posix())
    sqlite_connection.migrate(SQLITE_MIGRATION_PATH)
    games_repository: GamesRepository = SqliteGamesRepository(sqlite_connection)

    chroma_max_batch_size = chroma_client.get_max_batch_size()

    while True:
        try:
            queue_item: WorkerOutput = q.get(timeout=10)

            print(f'Saving {queue_item.file}')

            for batch in itertools.batched(queue_item.moves.items(), chroma_max_batch_size):
                # TODO: trovare una strategia di creazione degli id migliore

                embeddings = [x[1] for x in batch]
                ids = [str(x[0]) for x in batch]

                chroma_embeddings.upsert(
                    embeddings=embeddings,
                    ids=ids,
                )

            games = pd.DataFrame(columns=PGN_HEADERS)
            for game_data in queue_item.games:
                games.iloc[game_data[0]] = pd.Series(game_data[1])

            for game_id, game_record in games.iterrows():
                # TODO: gameid deve essere aggiustato e trovato un modo migliore
                game = Game(game_id)
                game.event = game_record['Event']
                game.site = game_record['Site']

                games_repository.save_game(game)

                # TODO: moves - games table

            sqlite_connection.commit()

        except queue.Empty:
            if not event.is_set():
                break


if __name__ == '__main__':
    pgnfiles = PGN_FOLDER.glob('*.pgn')

    queue = multiprocessing.Queue()
    event = multiprocessing.Event()

    saver_process = multiprocessing.Process(target=saver, args=(queue, event,))
    saver_process.start()

    with concurrent.futures.ProcessPoolExecutor(math.floor(multiprocessing.cpu_count() / 2)) as executor:
        for filepath in pgnfiles:
            executor.submit(worker, filepath).add_done_callback(functools.partial(worker_callback, queue))

    print(f'All done - sending event ... waiting saver process join')
    event.set()
    saver_process.join()