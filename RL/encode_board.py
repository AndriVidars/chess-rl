import chess
import torch

def encode_board_state(board: chess.Board):
    """
    Encodes the game state into a tensor.
    Returns: Tensor (65,) [turn, piece_at_sq_0, ... piece_at_sq_63]

    Values:
    - distinct from 0-12 representing piece types (0=empty)
    - turn: 0=White, 1=Black
    """
    # 65 elements: 1 for turn + 64 squares
    state = torch.zeros(65, dtype=torch.long)
    state[0] = 0 if board.turn == chess.WHITE else 1
    
    for sq in range(64):
        piece = board.piece_at(sq)
        if piece:
            # 1-6 for White, 7-12 for Black
            val = piece.piece_type
            if piece.color == chess.BLACK:
                val += 6
            state[sq + 1] = val
    
    return state

def encode_legal_moves(board: chess.Board):
    """
    Encodes legal moves into a tensor.
    Returns: Tensor (64*64,) [from_square * 64 + to_square]
    """
    moves = board.legal_moves
    move_tensor = torch.zeros(64*64, dtype=torch.float32)
    for move in moves:
        idx = move.from_square * 64 + move.to_square
        move_tensor[idx] = 1
    return move_tensor

def decode_move(move_idx: int):
    from_square = move_idx // 64
    to_square = move_idx % 64
    return chess.Move(from_square, to_square)
