import chess
import random
import torch
import os
from RL.chess_net import ChessNet
from RL.game_environment.game import Game
from RL.game_environment.chess_net_agent import ChessNetAgent, ChessNetHandler
from RL.game_environment.stockfish_agent import StockFishAgent


class EvalHandler:
    def __init__(self, num_games: int, batch_size: int, weights_path: str, stockfish_path: str, stockfish_elo, stockfish_time_per_move):
        self.num_games = num_games
        self.batch_size = batch_size
        self.boards = [chess.Board() for _ in range(batch_size)]
        
        self.stockfish_path = stockfish_path
        self.stockfish_elo = stockfish_elo
        self.stockfish_time_per_move = stockfish_time_per_move
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        self.model_handler = ChessNetHandler(self.boards, ChessNet(), device)
        self.model_handler.model.load_state_dict(torch.load(weights_path))
        self.model_handler.model.eval()

        self.games = [self.assign_agents(board_idx) for board_idx in range(self.batch_size)]

    def assign_agents(self, board_idx: int):
        stockfish_agent = StockFishAgent(self.boards[board_idx], self.stockfish_path, self.stockfish_elo, self.stockfish_time_per_move)
        chessnet_agent = ChessNetAgent(self.model_handler, self.boards[board_idx], board_idx)
        
        agent_white = stockfish_agent if random.random() < 0.5 else chessnet_agent
        agent_black = stockfish_agent if agent_white == chessnet_agent else chessnet_agent
        return Game(self.boards[board_idx], agent_white, agent_black)


    def eval(self): 
        results = []
        games_completed = 0
        games_started = self.batch_size
        while games_completed < self.num_games:            
            for i, game in enumerate(self.games):
                if game is None:
                    continue
                game.make_turn()
                if game.board.is_game_over():
                    results.append(game.get_result())
                    print(f"Game completed in {game.board.fullmove_number} moves")
                    games_completed += 1
                    self.boards[i].reset()
                    if games_started < self.num_games:
                        games_started += 1
                        self.games[i] = self.assign_agents(i)
                    else:
                        self.games[i] = None
            
            self.model_handler.cur_moves = None
            self.model_handler.cur_log_probs = None
        
        return results

def main():
    stockfish_path = os.path.join(os.path.dirname(__file__), "..", "..", "stockfish.exe")
    weights_path = os.path.join(os.path.dirname(__file__), "..", "training", "imitation_training_best_eval.pth")
    stockfish_elo = 1350
    stockfish_time_per_move = 10
    num_games = 128
    batch_size = 32
    
    handler = EvalHandler(num_games, batch_size, weights_path, stockfish_path, stockfish_elo, stockfish_time_per_move)
    results = handler.eval()
    
    print(results)
    

if __name__ == "__main__":
    main()

    

