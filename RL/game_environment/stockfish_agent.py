import chess
from stockfish import Stockfish
from RL.game_environment.agent import Agent

class StockFishAgentHandler:
    def __init__(self, stockfish_path: str, elo: int = 1600, time_per_move: int = 10):
        self.stockfish_path = stockfish_path
        self.elo = elo
        self.time_per_move = time_per_move
        self.stockfish = Stockfish(path=stockfish_path)
        self.stockfish.set_elo_rating(elo)

class StockFishAgent(Agent):
    def __init__(self, stockfish_handler: StockFishAgentHandler, board: chess.Board):
        super().__init__(board)
        self.stockfish_handler = stockfish_handler
    
    def sample_move(self) -> chess.Move:
        self.stockfish_handler.stockfish.set_fen_position(self.board.fen())
        best_move_uci = self.stockfish_handler.stockfish.get_best_move_time(self.stockfish_handler.time_per_move)
        return chess.Move.from_uci(best_move_uci)