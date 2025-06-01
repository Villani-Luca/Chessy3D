import chess
import cv2
import numpy as np
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
        embedding: np.ndarray | np.array,
        games_repo: PgGamesRepository,
        search_limit = 5):
        super().__init__(RetrievalJobSignals())

        # Store constructor arguments (re-used for processing)
        self.games_repo = games_repo
        self.signals = RetrievalJobSignals()
        self.search_limit = search_limit
        self.embedding = embedding

    def execute(self):
        result = self.games_repo.get_best_games_from_naiveposition(self.embedding)
        print(result)

        return result
