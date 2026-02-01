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

def encode_board_scalars(board: chess.Board):
    """
    Encodes scalar features of the board.
    Returns: Tensor (6,) [fullmove_number, halfmove_clock, castling_rights_wk, castling_rights_wq, castling_rights_bk, castling_rights_bq]
    """
    state = torch.zeros(6, dtype=torch.float32)
    
    # 1. Fullmove number (normalized, assuming typical game < 200 moves, but can go higher)
    state[0] = min(board.fullmove_number / 100.0, 5.0) # Cap at 500 moves just in case
    
    # 2. Halfmove clock (moves since last pawn move or capture) - important for 50 move rule draw
    state[1] = board.halfmove_clock / 100.0
    
    # 3. Castling rights
    state[2] = 1.0 if board.has_castling_rights(chess.WHITE) else 0.0 # Any white castling
    # Actually explicit rights are better
    state[2] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    state[3] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    state[4] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    state[5] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    
    return state
