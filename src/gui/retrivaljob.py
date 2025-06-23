import numpy as np

from src.gui.worker import Worker, WorkerSignals
from src.retrieval.src.pgsql import PgGamesRepository


class RetrievalJobSignals(WorkerSignals):
    pass

class RetrievalJob(Worker):
    def __init__(self,
        embedding: np.ndarray,
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
