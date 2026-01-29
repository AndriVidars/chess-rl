import chess
from abc import ABC, abstractmethod

class Agent(ABC):
    def __init__(self, board: chess.Board):
        self.board = board

    @abstractmethod
    def sample_move(self) -> chess.Move:
        pass
    