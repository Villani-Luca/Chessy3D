import numpy as np
from pgvector import Bit

import psycopg
from pgvector.psycopg import register_vector

import typing

class Connection:
    def __init__(self, connection_string: str) -> None:
        self.conn: psycopg.Connection = psycopg.connect(connection_string)
        self.cursor: psycopg.Cursor = self.conn.cursor()
        register_vector(self.conn)

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

    def get_best_positions_from_naive(self, embedding: np.array):
        return self.conn.cursor.execute(
            """
            SELECT v.embeddingid, v.embedding <~> $1 AS distance
            FROM naivevectors v
            ORDER BY distance
            LIMIT 5 
            """,
            (Bit(embedding).to_text(),)
        ).fetchall()

    def get_best_games_from_naiveposition(self, position: np.array):
        return self.conn.cursor.execute(
            """
            SELECT g.id, g.event, g.date, g.white, g.whitetitle, g.black, g.blacktitle, v.embedding <~> %s as distance, v.embeddingid, g.moves
            from games g
            join moves m on g.id = m.gameid
            join naivevectors v on m.embeddingid = v.embeddingid
            order by distance
            limit 5
            """,
            (Bit(position).to_text(),)
        ).fetchall()

    def get_games_from_move(self, move_id: str, limit = 5):
        return self.conn.cursor.execute(
            """SELECT g.* FROM games g JOIN moves m ON g.id = m.gameid WHERE m.embeddingid = %s LIMIT %s""",
            (move_id,limit)
        ).fetchall()