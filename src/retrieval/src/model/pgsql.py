from src.retrieval.src.model.game import Game

import psycopg

from datetime import datetime
import typing

class Connection:
    def __init__(self, connection_string: str) -> None:
        self.conn: psycopg.Connection = psycopg.connect(connection_string)
        self.cursor: psycopg.Cursor = self.conn.cursor()

    # Should be called only from the Connection not from the repositories
    def commit(self):
        self.conn.commit()

    def begin(self):
        self.conn.transaction()

    def rollback(self):
        self.conn.rollback()

    def close(self):
        self.cursor.close()
        self.conn.close()

class PgMovesRepository:
    conn: Connection = None

    def __init__(self, conn: Connection):
        self.conn = conn

    def get_highest_gameid(self):
        res = self.conn.cursor.execute("SELECT MAX(gameid) FROM moves").fetchone()
        return res[0] or 0

    def save_moves(self, moves: list[tuple[int, str]]):
        '''
        necessitá del commit subito dopo
        '''
        self.conn.cursor.execute("""CREATE TEMPORARY TABLE temp_moves ON COMMIT DROP AS SELECT * from moves LIMIT 0""")
        with self.conn.cursor.copy("""COPY temp_moves (gameid, embeddingid) FROM STDIN (FORMAT BINARY)""") as copy:
            for move in moves:
                copy.write_row(move)

        self.conn.cursor.execute("""INSERT INTO moves (gameid, embeddingid) SELECT gameid, embeddingid FROM temp_moves ON CONFLICT DO NOTHING""")

class PgGamesRepository:
    conn: Connection = None

    def __init__(self, conn: Connection):
        self.conn = conn

    def get_games_moves(self, sentinelid: int, maxsentileid: int = None, limit: int = 100) -> list[typing.Tuple[int, str]]:
        """
        :returns list[(gameid, moves)]
        """
        return self.conn.cursor.execute(
            """SELECT id, moves FROM games WHERE id >= %s ORDER BY id LIMIT %s""" if maxsentileid is None else """SELECT id, moves FROM games WHERE id >= %s AND id <= %s ORDER BY id LIMIT %s""",
            (sentinelid, limit) if maxsentileid is None else (sentinelid, maxsentileid, limit),
            prepare=True
        ).fetchall()

    def delete_game(self, gameid: int):
        self.conn.cursor.execute("""DELETE FROM games WHERE id = %s""", (gameid,))

    @staticmethod
    def __game_from_result(res):
        # TODO: da rifare
        game = Game(res[0])
        game.date = datetime.fromtimestamp(res[1]) if res[1] is not None else None
        game.event = res[2]
        game.site = res[3]

        return game