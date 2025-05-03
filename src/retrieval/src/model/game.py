from datetime import datetime

class Game:
    id: int
    event: str | None = None
    site: str | None = None
    date: datetime | None = None

    def __init__(self, game_id):
        self.id = game_id
