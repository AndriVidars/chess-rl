import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
from RL.encode_board import encode_board_state, encode_legal_moves, decode_move
from typing import List
import os
import concurrent.futures


class ChessNet(nn.Module):
    def __init__(self, embedding_dim=32, num_convs=3, num_linear=3):
        super(ChessNet, self).__init__()
        self.embedding_dim = embedding_dim
        
        # Embedding: 13 distinct piece types (0-12) 0=empty, 1-6 White, 7-12 Black
        self.embedding = nn.Embedding(num_embeddings=13, embedding_dim=embedding_dim)
        
        # Convolutions
        self.convolutions = nn.ModuleList()
        for i in range(num_convs):
            # Input starts at embedding_dim, then increases
            in_channels = (i + 1) * embedding_dim
            out_channels = (i + 2) * embedding_dim
            
            self.convolutions.append(nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ))
        
        # Fully Connected MLP
        conv_output_size = ((num_convs + 1) * embedding_dim) * 8 * 8
        
        linear_dim = 4096 # 64*64 moves
        self.mlp = nn.ModuleList()
        
        for i in range(num_linear):
            in_features = (conv_output_size + 1) if i == 0 else linear_dim # +1 for turn info
            out_features = linear_dim
            
            if i < num_linear - 1:
                self.mlp.append(nn.Sequential(
                    nn.Linear(in_features=in_features, out_features=out_features),
                    nn.BatchNorm1d(out_features),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                ))
            else:
                self.mlp.append(nn.Linear(in_features=in_features, out_features=out_features))
        

    def forward(self, x):
        # x shape: (Batch, 65) -> [Turn, 64 squares]
        turn = x[:, 0].float().unsqueeze(1) # (Batch, 1)
        board_indices = x[:, 1:].long()     # (Batch, 64)
        
        # Embed the board: (Batch, 64, Embed_Dim)
        x = self.embedding(board_indices)
        
        # Reshape for Conv2d: (Batch, Embed_Dim, 8, 8)
        # Permute to (Batch, Channel, Height, Width)
        x = x.view(-1, 8, 8, self.embedding_dim).permute(0, 3, 1, 2)
        
        # conv
        for conv in self.convolutions:
            x = conv(x)
        
        x = x.flatten(start_dim=1)
        
        # Concatenate Turn info
        x = torch.cat([x, turn], dim=1)

        # mlp
        for linear in self.mlp:
            x = linear(x)
        
        return x

def sample_moves(model: ChessNet, boards: List[chess.Board], device: torch.device, num_workers: int = os.cpu_count()):
    model.eval()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
    
    future_states = executor.map(encode_board_state, boards)
    future_masks = executor.map(encode_legal_moves, boards)
    
    board_states = torch.stack(list(future_states)).to(device) # (Batch, 65)
    legal_moves_masks = torch.stack(list(future_masks)).to(device) # (Batch, 4096)

    with torch.no_grad():
        logits = model(board_states) # (Batch, 4096)
        
        # Mask illegal moves
        logits[legal_moves_masks == 0] = float('-inf')
        
        # Sample action
        probs = torch.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(probs)
        action_indices = dist.sample() # (Batch,)
        log_probs = dist.log_prob(action_indices) # (Batch,)
        
        moves = [decode_move(idx.item()) for idx in action_indices]
        return moves, log_probs

def sample_move(model: ChessNet, board: chess.Board, device: torch.device):
    moves, log_probs = sample_moves(model, [board], device)
    return moves[0], log_probs[0]
