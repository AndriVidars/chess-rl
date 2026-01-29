import chess
from stockfish import Stockfish
from RL.game_environment.agent import Agent

class StockFishAgent(Agent):
    def __init__(self, board: chess.Board, stockfish_path: str, elo: int = 1600, time_per_move: int = 10):
        super().__init__(board)
        self.stockfish = Stockfish(path=stockfish_path)
        self.stockfish.set_elo_rating(elo)
        self.time_per_move = time_per_move
    
    def sample_move(self) -> chess.Move:
        self.stockfish.set_fen_position(self.board.fen())
        best_move_uci = self.stockfish.get_best_move_time(self.time_per_move)
        return chess.Move.from_uci(best_move_uci)