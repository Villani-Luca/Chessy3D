import pathlib
from src.retrieval.src.model.game import Game
import sqlite3
from datetime import datetime
import typing

class Connection:
    conn = None

    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path, isolation_level=None)
        self.cursor = self.conn.cursor()
        self.cursor.execute('pragma journal_mode=wal')

    # Should be called only from the Connection not from the repositories
    def commit(self):
        self.conn.commit()

    def begin(self):
        if self.conn.in_transaction:
            raise RuntimeError('Cannot begin a transaction inside another transaction')

        self.conn.execute('BEGIN')

    def rollback(self):
        self.conn.rollback()

    def migrate(self, migration_path: pathlib.Path):
        self.cursor.executescript(migration_path.read_text())
        self.commit()

    def close(self):
        self.cursor.close()
        self.conn.close()

class SqliteMovesRepository:
    conn: Connection = None

    def __init__(self, conn: Connection):
        self.conn = conn

    def get_highest_gameid(self):
        query = "SELECT MAX(gameid) FROM moves"
        res = self.conn.cursor.execute(query).fetchone()
        return res[0] or 0

    def save_moves(self, moves: list[tuple[int, str]]):
        #query = """INSERT or IGNORE INTO moves (gameid, chromaid) VALUES (?, ?)"""
        query = """INSERT INTO moves (gameid, chromaid) VALUES (?, ?) ON CONFLICT (gameid, chromaid) DO NOTHING"""
        self.conn.cursor.executemany(query, moves)

class SqliteGamesRepository:
    conn: Connection = None

    def __init__(self, conn: Connection):
        self.conn = conn

    def get_games_moves(self, sentinelid: int, maxsentileid: int = None, limit: int = 100) -> list[typing.Tuple[int, str]]:
        """
        :returns list[(gameid, moves)]
        """
        query = """SELECT id, moves FROM games WHERE id >= ? ORDER BY id LIMIT ?""" if maxsentileid is None else """SELECT id, moves FROM games WHERE id >= ? AND id <= ? ORDER BY id LIMIT ?"""
        return self.conn.cursor.execute(query, (sentinelid, limit) if maxsentileid is None else (sentinelid, maxsentileid, limit)).fetchall()

    @staticmethod
    def __game_from_result(res):
        # TODO: da rifare
        game = Game(res[0])
        game.date = datetime.fromtimestamp(res[1]) if res[1] is not None else None
        game.event = res[2]
        game.site = res[3]

        return game