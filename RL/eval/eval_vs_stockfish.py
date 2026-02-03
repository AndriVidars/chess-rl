import chess
import random
import torch
import os
from RL.chess_net import ChessNet
from RL.game_environment.game import Game
from RL.game_environment.chess_net_agent import ChessNetAgent, ChessNetHandler
from RL.game_environment.stockfish_agent import StockFishAgent, StockFishAgentHandler


class EvalHandler:
    def __init__(self,
                 num_games: int, 
                 batch_size: int, 
                 weights_path: str, 
                 stockfish_path: str,
                 stockfish_elo: int,
                 stockfish_time_per_move: int):
        self.num_games = num_games
        self.batch_size = batch_size
        self.boards = [chess.Board() for _ in range(batch_size)]
        
        self.stockfish_handler = StockFishAgentHandler(stockfish_path, stockfish_elo, stockfish_time_per_move)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #print(f"Using device: {device}")
        
        self.model_handler = ChessNetHandler(self.boards, ChessNet(), device)
        self.model_handler.model.load_state_dict(torch.load(weights_path), strict=False)
        self.model_handler.model.eval()

        self.games = [self.assign_agents(board_idx) for board_idx in range(self.batch_size)]

    def assign_agents(self, board_idx: int):
        stockfish_agent = StockFishAgent(self.stockfish_handler, self.boards[board_idx])
        chessnet_agent = ChessNetAgent(self.model_handler, self.boards[board_idx], board_idx)
        
        stockfish_is_white = random.random() < 0.5
        
        self.model_handler.agent_colors[board_idx] = chess.BLACK if stockfish_is_white else chess.WHITE
        
        agent_white = stockfish_agent if stockfish_is_white else chessnet_agent
        agent_black = stockfish_agent if not stockfish_is_white else chessnet_agent
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
                if game.board.is_game_over(claim_draw=True):
                    winner = game.get_winner()
                    if winner is None:
                        ties += 1
                    elif isinstance(winner, ChessNetAgent):
                        wins += 1
                    
                    num_moves_total += game.board.fullmove_number
                    games_completed += 1
                    self.boards[i].reset()
                    if games_started < self.num_games:
                        games_started += 1
                        self.games[i] = self.assign_agents(i)
                    else:
                        self.games[i] = None
        
        win_rate = wins / self.num_games
        tie_rate = ties / self.num_games
        loss_rate = 1 - win_rate - tie_rate
        avg_moves = num_moves_total / self.num_games
        return win_rate, tie_rate, loss_rate, avg_moves

def main():
    stockfish_path = os.path.join(os.path.dirname(__file__), "..", "stockfish", "stockfish.exe")
    weights_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "pre_trained_4096_1600.pth")
    
    stockfish_elo = 1350
    stockfish_time_per_move = 25
    num_games = 128
    batch_size = 32
    
    handler = EvalHandler(num_games, batch_size, weights_path, stockfish_path, stockfish_elo, stockfish_time_per_move)
    results = handler.eval()
    
    print(f"Win rate: {results[0]}")
    print(f"Tie rate: {results[1]}")
    print(f"Loss rate: {results[2]}")
    print(f"Average moves: {results[3]}")
    return results
    

if __name__ == "__main__":
    main()

    

