import chess
import cv2
import ultralytics
from PySide6.QtCore import Signal, QThreadPool

import src.chessboard_localization_temp.main as chess_localization
from src.gui.worker import Worker, WorkerSignals
from src.retrieval.src.milvus import MilvusRepository
from src.retrieval.src.model.pgsql import PgGamesRepository
from src.retrieval.src.position_embeddings import PositionEmbedder


class RetrievalJobSignals(WorkerSignals):
    pass

class RetrievalJob(Worker):
    def __init__(self,
        embedder: PositionEmbedder,
        games_repo: PgGamesRepository,
        board: chess.Board,
        search_limit = 5):
        super().__init__(RetrievalJobSignals())

        # Store constructor arguments (re-used for processing)

        self.embedder = embedder
        self.board = board.copy()
        self.games_repo = games_repo
        self.signals = RetrievalJobSignals()
        self.search_limit = search_limit

    def execute(self):
        embedding = self.embedder.embedding(self.board)
        #self.signals.progress.emit(10, 'Searching game...')

        # milvus_result = self.milvus_repo.search_embeddings(embedding, self.search_limit)
        #self.signals.progress.emit(40, 'Searching similar games...')

        #result = self.games_repo.get_games_from_move()
        return [('id', 'match', 'test'),('id', 'match', 'test'),('id', 'match', 'test'),('id', 'match', 'test'),('id', 'match', 'test'),('id', 'match', 'test')]