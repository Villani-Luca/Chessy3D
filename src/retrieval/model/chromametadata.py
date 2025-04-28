from typing import Protocol
import json

class JsonSerializable(Protocol):
    def to_json(self) -> str:
        return json.dumps(self.__dict__)

    def __repr__(self) -> str:
        return json.dumps(self.__dict__)

class EmbeddingsDocument(JsonSerializable):
    games: list[int]

    def __init__(self, games):
        self.games = games

class GameMetadata(JsonSerializable):
    event: str

    def __init__(self, event):
        self.event = event