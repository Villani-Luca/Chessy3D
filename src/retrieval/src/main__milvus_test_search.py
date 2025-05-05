from milvus import MilvusRepository
import chess
from position_embeddings import NaivePositionEmbedder
from model.pgsql import Connection, PgGamesRepository

PG_CONN = "host=localhost user=postgres password=password dbname=chessy"


repo = MilvusRepository()
repo.load_collection()

emb = repo.get_embedding('10329633534602812817')
print('Embedding', emb)

embedder = NaivePositionEmbedder()
board = chess.Board("rnbqkb1r/pppp1ppp/5n2/4p3/2P4P/4P3/PP1P1PP1/RNBQKBNR b KQkq - 0 3")
embedding = embedder.embedding(board)

search = repo.search_embeddings(embedding, limit=5)

pgconn = Connection(PG_CONN)
games_repo = PgGamesRepository(pgconn)

for hits in search:
    for hit in hits:
        games = games_repo.get_games_from_move(hit['id'])
        print(hit, games)

