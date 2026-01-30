import chess
import random
import torch
import os
from RL.chess_net import ChessNet
from RL.game_environment.game import Game
from RL.game_environment.chess_net_agent import ChessNetAgent, ChessNetHandler

class EvalHandler:
    def __init__(self, num_games: int, batch_size: int, weights_path_primary: str, weights_path_baseline: str):
        self.num_games = num_games
        self.batch_size = batch_size
        self.boards = [chess.Board() for _ in range(batch_size)]
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #print(f"Using device: {device}")
        
        self.model_handler_primary = ChessNetHandler(self.boards, ChessNet(), device)
        self.model_handler_primary.model.load_state_dict(torch.load(weights_path_primary), strict=False)
        self.model_handler_primary.model.eval()

        self.model_handler_baseline = ChessNetHandler(self.boards, ChessNet(), device)
        self.model_handler_baseline.model.load_state_dict(torch.load(weights_path_baseline), strict=False)
        self.model_handler_baseline.model.eval()

        self.games = [self.assign_agents(board_idx) for board_idx in range(self.batch_size)]

    def assign_agents(self, board_idx: int):
        chessnet_agent_primary = ChessNetAgent(self.model_handler_primary, self.boards[board_idx], board_idx)
        chessnet_agent_baseline = ChessNetAgent(self.model_handler_baseline, self.boards[board_idx], board_idx)
        
        agent_white = chessnet_agent_primary if random.random() < 0.5 else chessnet_agent_baseline
        agent_black = chessnet_agent_primary if agent_white == chessnet_agent_baseline else chessnet_agent_baseline
        return Game(self.boards[board_idx], agent_white, agent_black)


    def eval(self): 
        wins = 0
        ties = 0
        num_moves_total = 0

        games_completed = 0
        games_started = self.batch_size
        while games_completed < self.num_games:            
            for i, game in enumerate(self.games):
                if game is None:
                    continue
                game.make_turn()
                if game.board.is_game_over():
                    winner = game.get_result()
                    if winner is None:
                        ties += 1
                    elif isinstance(winner, ChessNetAgent) and winner.model_handler == self.model_handler_primary:
                        wins += 1
                    num_moves_total += game.board.fullmove_number
                    games_completed += 1
                    self.boards[i].reset()
                    if games_started < self.num_games:
                        games_started += 1
                        self.games[i] = self.assign_agents(i)
                    else:
                        self.games[i] = None
            
            self.model_handler_primary.cur_moves = None
            self.model_handler_primary.cur_log_probs = None
            self.model_handler_baseline.cur_moves = None
            self.model_handler_baseline.cur_log_probs = None
        
        win_rate = wins / self.num_games
        tie_rate = ties / self.num_games
        loss_rate = 1 - win_rate - tie_rate
        avg_moves = num_moves_total / self.num_games
        return win_rate, tie_rate, loss_rate, avg_moves

def main():
    weights_path_primary = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "pre_trained_4096_1600.pth")
    weights_path_baseline = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "pre_trained_32_1600.pth")
    
    num_games = 128 # 1024
    batch_size = 16 # 64
    
    handler = EvalHandler(num_games, batch_size, weights_path_primary, weights_path_baseline)
    results = handler.eval()
    
    print(f"Win rate: {results[0]}")
    print(f"Tie rate: {results[1]}")
    print(f"Loss rate: {results[2]}")
    print(f"Average moves: {results[3]}")
    return results
    

if __name__ == "__main__":
    main()