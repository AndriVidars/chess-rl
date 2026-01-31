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
                 trajectories: List[Trajectory] = None):
        
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
        self.agent_colors = [None] * self.num_boards # chess.WHITE or chess.BLACK for each board

    def set_agent_colors(self, agent_colors: List[chess.Color]):
        self.agent_colors = agent_colors
    
    def sample_moves(self):
        assert self.cur_moves is None # after last turn, all moves should have been consumed
        
        self.cur_moves = [None] * self.num_boards
        self.cur_board_states = [None] * self.num_boards
        self.cur_action_indices = [None] * self.num_boards
        self.cur_values = [None] * self.num_boards

        active_indices = []
        active_boards = []
        
        for i, board in enumerate(self.boards):
            if board.turn == self.agent_colors[i]:
                active_indices.append(i)
                active_boards.append(board)
        
        if not active_indices:
             return

        # batch sample moves across all active boards
        moves, board_states, action_indices, values = sample_moves(self.model, active_boards, self.device)
        
        for idx_in_batch, original_idx in enumerate(active_indices):
            self.cur_moves[original_idx] = moves[idx_in_batch]
            self.cur_board_states[original_idx] = board_states[idx_in_batch]
            self.cur_action_indices[original_idx] = action_indices[idx_in_batch]
            self.cur_values[original_idx] = values[idx_in_batch]
        
        if self.collect_trajectories:
            for original_idx in active_indices:
                self.trajectories[original_idx].states.append(self.cur_board_states[original_idx])
                self.trajectories[original_idx].actions.append(self.cur_action_indices[original_idx])
                self.trajectories[original_idx].values.append(self.cur_values[original_idx])


class ChessNetAgent(Agent):
    def __init__(self, model_handler: ChessNetHandler, board: chess.Board, board_idx: int):
        super().__init__(board)
        self.model_handler = model_handler
        self.board_idx = board_idx
    
    def sample_move(self) -> chess.Move:
        # If no moves are cached, or specifically no move for this board (and we are here, so it must be our turn)
        if self.model_handler.cur_moves is None or self.model_handler.cur_moves[self.board_idx] is None:
            self.model_handler.cur_moves = None # Invalidate current cache strictly
            self.model_handler.sample_moves()
        
        move = self.model_handler.cur_moves[self.board_idx]     
        self.model_handler.cur_moves[self.board_idx] = None # consume the move
        return move

        