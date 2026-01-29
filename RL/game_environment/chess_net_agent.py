import chess
import torch
from stockfish import Stockfish
from RL.game_environment.agent import Agent

from RL.chess_net import ChessNet, sample_moves, sample_move


class ChessNetHandler:
    def __init__(self, boards: list[chess.Board], model: ChessNet, device: torch.device):
        self.model = model
        self.device = device
        self.boards = boards
        self.num_boards = len(boards)
        self.cur_moves = None
        self.cur_log_probs = None
    
    def sample_moves(self):
        assert self.cur_moves is None
        self.cur_moves, self.cur_log_probs = sample_moves(self.model, self.boards, self.device)
        

class ChessNetAgent(Agent):
    def __init__(self, model_handler: ChessNetHandler, board: chess.Board, board_idx: int):
        super().__init__(board)
        self.model_handler = model_handler
        self.board_idx = board_idx
    
    def sample_move(self) -> chess.Move:
        if self.model_handler.cur_moves is None:
            self.model_handler.sample_moves()
        
        move = self.model_handler.cur_moves[self.board_idx]
        self.model_handler.cur_moves[self.board_idx] = None
        return move

        