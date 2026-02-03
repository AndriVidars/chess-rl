import chess
from stockfish import Stockfish
from RL.game_environment.agent import Agent

class StockFishAgentHandler:
    def __init__(self, stockfish_path: str, elo: int = 1600, depth: int = 10):
        self.stockfish_path = stockfish_path
        self.elo = elo
        self.depth = depth
        self.stockfish = Stockfish(path=stockfish_path)
        self.stockfish.set_elo_rating(elo)
        self.stockfish.set_depth(depth)

class StockFishAgent(Agent):
    def __init__(self, stockfish_handler: StockFishAgentHandler, board: chess.Board):
        super().__init__(board)
        self.stockfish_handler = stockfish_handler
    
    def sample_move(self) -> chess.Move:
        self.stockfish_handler.stockfish.set_fen_position(self.board.fen())
        # ensure depth is enforced if it resets
        self.stockfish_handler.stockfish.set_depth(self.stockfish_handler.depth) 
        best_move_uci = self.stockfish_handler.stockfish.get_best_move()
        return chess.Move.from_uci(best_move_uci)