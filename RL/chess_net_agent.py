import chess
from chess_net import ChessNet
from encode_board import encode_board_state, encode_legal_moves, decode_move
import torch
from typing import List

class ChessNetAgent:
    def __init__(self, net: ChessNet):
        self.net = net

    def act(self, board: chess.Board):
        self.net.eval()
        with torch.no_grad():
            board_state = encode_board_state(board).unsqueeze(0) # (1, 65)
            legal_moves_mask = encode_legal_moves(board) # (4096,)

            logits = self.net(board_state).squeeze(0) # (4096,)
            
            # Mask illegal moves
            logits[legal_moves_mask == 0] = float('-inf')
            
            # Sample action
            probs = torch.softmax(logits, dim=0)
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample().item()
            
            move = decode_move(action_idx)
            return move
    
    def sample_moves(self, boards: List[chess.Board]) -> List[chess.Move]:
        """
        Batched sampling for a list of ACTIVE boards. 
        Caller is responsible for filtering out finished games.
        """
        self.net.eval()
        parsed_states = [encode_board_state(board) for board in boards]
        batch_states = torch.stack(parsed_states) # (Batch, 65)

        with torch.no_grad():
            batch_logits = self.net(batch_states) # (Batch, 4096)
        
        # Batch Masking & Sampling
        batch_moves = []
        
        for i, board in enumerate(boards):
            logits = batch_logits[i]
            
            legal_mask = encode_legal_moves(board)
            logits[legal_mask == 0] = float('-inf')
            
            probs = torch.softmax(logits, dim=0)
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample().item()
            
            batch_moves.append(decode_move(action_idx))
            
        return batch_moves

        
        
