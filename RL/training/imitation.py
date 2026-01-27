import sys
import os
import random
import concurrent.futures
from stockfish import Stockfish
import chess
import torch
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from encode_board import encode_board_state


def worker_generate_batch(stockfish_path, batch_size, elo=1600, time_per_move=10):
    stockfish = Stockfish(path=stockfish_path)
    stockfish.set_elo_rating(elo)

    data = []
    
    for _ in range(batch_size):
        board = chess.Board()
        
        while not board.is_game_over() and board.fullmove_number < 120:
            fen = board.fen()
            stockfish.set_fen_position(fen)
            
            best_move_uci = stockfish.get_best_move_time(time_per_move)
            if not best_move_uci:
                break
                
            move = chess.Move.from_uci(best_move_uci)
            
            # Encode state and move
            # Encoded state: 65-dim tensor
            state_tensor = encode_board_state(board)
            
            # Target: index of the move (from_sq * 64 + to_sq)
            move_idx = move.from_square * 64 + move.to_square
            
            data.append((state_tensor, move_idx))
            
            # Make the move on the board to proceed
            board.push(move)
                
    return data

class ImitationTrainer:
    def __init__(self, stockfish_path: str, elo: int = 1600, batch_size: int = 32, num_games: int = 100_000):
        self.stockfish_path = stockfish_path
        self.elo = elo
        self.batch_size = batch_size
        self.num_games = num_games
        self.data = []

    def collect_moves_data(self):
        num_workers = os.cpu_count() or 1
        games_per_worker = self.num_games // num_workers
        
        print(f"Starting generation of {self.num_games} games across {num_workers} workers...")
        
        start_time = time.time()
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    worker_generate_batch, 
                    self.stockfish_path, 
                    games_per_worker, 
                    self.elo
                ) 
                for _ in range(num_workers)
            ]
            
            results = []
            for future in concurrent.futures.as_completed(futures):
                batch_data = future.result()
                results.extend(batch_data)
                print(f"Worker finished. Collected {len(batch_data)} samples so far...")
        
        self.data = results
        total_time = time.time()     - start_time
        print(f"Total samples collected: {len(self.data)}")
        print(f"Total time taken: {total_time:.2f} seconds, {total_time / self.num_games:.2f} seconds per game")

if __name__ == "__main__":
    stockfish_path = os.path.join(os.path.dirname(__file__), "../../stockfish.exe")
    
    # Example usage
    trainer = ImitationTrainer(stockfish_path, num_games=1024) # Small number for testing
    trainer.collect_moves()
