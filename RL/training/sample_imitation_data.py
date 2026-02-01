import sys
import os
import random
import concurrent.futures
from stockfish import Stockfish
import chess
import torch
import time

from RL.encode_board import encode_board_state, encode_board_scalars

def worker_generate_batch(stockfish_path, batch_size, elo=1500, time_per_move=5):
    stockfish = Stockfish(path=stockfish_path)
    stockfish.set_elo_rating(elo)

    data = []
    
    #st = time.time()
    for i in range(batch_size):
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
            scalar_tensor = encode_board_scalars(board)
            
            # Target: index of the move (from_sq * 64 + to_sq)
            move_idx = move.from_square * 64 + move.to_square
            
            data.append((state_tensor, scalar_tensor, move_idx))
            
            # Make the move on the board to proceed
            board.push(move)
                        
    return data

def generate_imitation_data(stockfish_path: str, elo: int = 1600, num_games: int = 4096):
    num_workers = os.cpu_count() or 1
    games_per_worker = num_games // num_workers
    
    print(f"Starting generation of {num_games} games across {num_workers} workers")
    
    start_time = time.time()
    results = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                worker_generate_batch, 
                stockfish_path, 
                games_per_worker, 
                elo
            ) 
            for _ in range(num_workers)
        ]
        
        for future in concurrent.futures.as_completed(futures):
            batch_data = future.result()
            results.extend(batch_data)
            print(f"Worker finished. Collected {len(batch_data)} move samples so far...")
    
    total_time = time.time() - start_time
    print(f"Total move samples collected: {len(results)}")
    print(f"Total time taken: {total_time:.2f} seconds, {total_time / num_games:.2f} seconds per game")

    # Convert to TensorDataset
    states = torch.stack([x[0] for x in results])
    scalars = torch.stack([x[1] for x in results])
    move_indices = torch.tensor([x[2] for x in results], dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(states, scalars, move_indices)
    
    filename = f"imitation_data_{num_games}_{elo}.pt"
    save_path = os.path.join(os.path.dirname(__file__), "..", "data", filename)
    torch.save(dataset, save_path)
    print(f"Dataset saved to {save_path}")
    
    return save_path

if __name__ == "__main__":
    stockfish_path = os.path.join(os.path.dirname(__file__), "..", "stockfish", "stockfish.exe")
    generate_imitation_data(stockfish_path, num_games=128, elo=1500)