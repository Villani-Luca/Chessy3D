import pandas as pd
import chess.pgn
import chromadb

path = "D:/uni/Chessy3D_Lumbras/LumbrasGigaBase -1899.pgn"
path = "D:/uni/Chessy3D_Lumbras/LumbrasGigaBase 2020.pgn"
path = "D:/uni/Chessy3D_Lumbras/splitted/LumbrasGigaBase 2020.pgn_1.pgn"

pgn_headers = [
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
]

games = pd.DataFrame(columns=pgn_headers)
game_n = 0

chroma_client = chromadb.PersistentClient(path="D:/uni/Chessy3D_Lumbras/chroma")
chroma_embeddings = chroma_client.create_collection("embeddings", get_or_create=True)
chroma_indices = chroma_client.create_collection("indices", get_or_create=True)
chroma_games = chroma_client.create_collection("games", get_or_create=True)

with open(path) as f:

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

print(len(games))
print('test')