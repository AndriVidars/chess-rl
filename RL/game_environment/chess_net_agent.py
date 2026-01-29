import chess
import torch
from stockfish import Stockfish
from RL.game_environment.agent import Agent

from RL.chess_net import ChessNet, sample_moves, sample_move
from typing import List, Dict


class Trajectory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.values = []
        self.reward = None

class ChessNetHandler:
    def __init__(self, 
                 boards: list[chess.Board], 
                 model: ChessNet, 
                 device: torch.device, 
                 collect_trajectories: bool = False,
                 trajectories: List[Dict[chess.Color, Trajectory]] = None):
        
        self.model = model.to(device)
        self.device = device
        self.boards = boards
        self.num_boards = len(boards)
        self.cur_moves = None
        self.cur_board_states = None
        self.cur_action_indices = None
        self.cur_values = None
        self.collect_trajectories = collect_trajectories
        self.trajectories = trajectories
    
    def sample_moves(self):
        assert self.cur_moves is None
        self.cur_moves, self.cur_board_states, self.cur_action_indices, self.cur_values = sample_moves(self.model, self.boards, self.device)
        
        if self.collect_trajectories:
            for i in range(self.num_boards):
                turn = self.boards[i].turn
                self.trajectories[i][turn].states.append(self.cur_board_states[i])
                self.trajectories[i][turn].actions.append(self.cur_action_indices[i])
                self.trajectories[i][turn].values.append(self.cur_values[i])
        

class ChessNetAgent(Agent):
    def __init__(self, model_handler: ChessNetHandler, board: chess.Board, board_idx: int):
        super().__init__(board)
        self.model_handler = model_handler
        self.board_idx = board_idx
    
    def sample_move(self) -> chess.Move:
        if self.model_handler.cur_moves is None or self.model_handler.cur_moves[self.board_idx] is None:
            self.model_handler.cur_moves = None
            self.model_handler.sample_moves()
        
        move = self.model_handler.cur_moves[self.board_idx]
        self.model_handler.cur_moves[self.board_idx] = None
        return move

        