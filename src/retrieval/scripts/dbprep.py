import datetime
import itertools
import pathlib

import chess.pgn
import chromadb
import pandas as pd

import helper
from src.retrieval.model.game import Game
from src.retrieval.model.gamesrepository import Connection, GamesRepository, SqliteGamesRepository

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

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH.as_posix(), settings=chromadb.Settings(allow_reset=True))
chroma_client.reset()
chroma_embeddings = chroma_client.create_collection("embeddings", get_or_create=True)

sqlite_connection = Connection(SQLITE_PATH.as_posix())
sqlite_connection.migrate(SQLITE_MIGRATION_PATH)
games_repository: GamesRepository = SqliteGamesRepository(sqlite_connection)

# TODO: processo paralello
for pgn_folder_file in PGN_FOLDER.glob('*.pgn'):
    if not pgn_folder_file.is_file():
        continue

    games = pd.DataFrame(columns=PGN_HEADERS)
    game_n = 0

    temp_storage = dict()  # array di tuple (id, fen, embedding)

    with pgn_folder_file.open() as f:
        while True:
            game = chess.pgn.read_game(f)

            ## If there are no more games, exit the loop
            if game is None:
                break

            # extract metadata
            # play move by move
            #   for each move create embedding of position

            games.loc[game_n] = pd.Series(game.headers)
            game_n = game_n + 1

            if game_n % 50 == 0:
                print(f'Processing game {game_n} {datetime.datetime.now()}')

            board = game.board()
            for move in game.mainline_moves():
                board.push(move)

                # find if position has been reached before
                fen = board.fen()
                found = temp_storage.get(fen)
                if found is None:
                    temp_storage[fen] = ([game_n], helper.board_to_embedding(board))
                else:
                    found[0].append(game_n)


    # update database
    # TODO: spostare in delle classi fatte apposta che deserializzino e serializzino
    # per ora é piú un test di come funziona chroma, tanto questa parte sará da velocizzare assolutamente
    chroma_max_batch_size = chroma_client.get_max_batch_size()
    temp_storage_len = len(temp_storage)

    chroma_embeddings_startid = chroma_embeddings.count()

    print('Saving')

    for batch in itertools.batched(temp_storage.values(), chroma_max_batch_size):
        # TODO: trovare una strategia di creazione degli id migliore

        embeddings = [x[1] for x in batch]
        ids = [str(x) for x in range(len(batch))]

        chroma_embeddings.upsert(
            embeddings=embeddings,
            ids=ids,
        )

    for gameid, game_record in games.iterrows():
        # TODO: gameid deve essere aggiustato e trovato un modo migliore
        game = Game(gameid)
        game.event = game_record['Event']
        game.site = game_record['Site']

        games_repository.save_game(game)

        # TODO: moves - games table

    sqlite_connection.commit()