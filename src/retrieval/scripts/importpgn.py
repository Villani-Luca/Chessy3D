import chess.pgn
import pandas as pd

pgn_file_name = "D:/uni/Chessy3D_Lumbras/LumbrasGigaBase 2020.pgn"

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

with open(pgn_file_name) as f:

    while True:
        game = chess.pgn.read_game(f)
## If there are no more games, exit the loop
        if game is None:
            break

        games.loc[game_n] = pd.Series(game.headers)
        game_n = game_n + 1

print(len(games))