import pathlib
from abc import ABC, abstractmethod
from src.retrieval.model.game import Game
import sqlite3
from datetime import datetime

class GamesRepository(ABC):
    @abstractmethod
    def game_by_id(self, game_id) -> Game:
        pass

    @abstractmethod
    def games_by_move(self, moveid) -> list[Game]: # TODO: define pagination and filters
        pass

    @abstractmethod
    def save_game(self, game: Game):
        pass

class Connection:
    _conn = None

    def __init__(self, db_path):
        if self._conn is None:
            self._conn = sqlite3.connect(db_path)
        self.cursor = self._conn.cursor()

    def __del__(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # Should be called only from the Connection not from the repositories
    def commit(self):
        self._conn.commit()

    def migrate(self, migration_path: pathlib.Path):
        self.cursor.executescript(migration_path.read_text())
        self.commit()


class SqliteGamesRepository(GamesRepository):
    conn: Connection = None

    def __init__(self, conn: Connection):
        self.conn = conn

    def game_by_id(self, game_id) -> Game:
        query = """SELECT id, date, event, site 
FROM games 
WHERE id = ?"""

        self.conn.cursor.execute(query, (game_id,))
        res = self.conn.cursor.fetchone()
        return SqliteGamesRepository.__game_from_result(res)

    def games_by_move(self, moveid, top = 50) -> list[Game]:
        query = """SELECT id, date, event, site 
FROM games 
WHERE id IN (
    SELECT m.gameid
    FROM moves m
    WHERE m.chromaid = ?
    LIMIT ?
)
"""

        self.conn.cursor.execute(query, (moveid, top))
        res = self.conn.cursor.fetchall()
        return [SqliteGamesRepository.__game_from_result(x) for x in res]


    def save_game(self, game: Game):
        query = """INSERT INTO games (id, date, event, site) VALUES (?, ?, ?, ?)"""
        self.conn.cursor.execute(query, (
            game.id,
            datetime.timestamp(game.date) if game.date is not None else None,
            game.event,
            game.site
        ))

    @staticmethod
    def __game_from_result(res):
        game = Game(res[0])
        game.date = datetime.fromtimestamp(res[1]) if res[1] is not None else None
        game.event = res[2]
        game.site = res[3]

        return game