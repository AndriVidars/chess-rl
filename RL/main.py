import chess

def main():
    board = chess.Board()
    board.is_game_over()
    board.is_checkmate()
    # board.push(from_square, to_square))
    
    moves = board.legal_moves
    for m in moves:
        print(f"{m.from_square})
    
    pass

if __name__ == "__main__":
    main()