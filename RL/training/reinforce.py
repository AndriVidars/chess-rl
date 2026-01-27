from chess_net_agent import ChessNetAgent
import chess
import torch
from tqdm import tqdm


class TrainableAgent:
    def __init__(self, agent: ChessNetAgent, optimizer, init_weights=None, frozen=False):
        self.agent = agent
        self.optimizer = optimizer
        if init_weights:
            self.agent.net.load_state_dict(init_weights)
        if frozen:
            for param in self.agent.net.parameters():
                param.requires_grad = False

def reinforce_train(
    agent_wrapper: TrainableAgent, 
    batch_size=64,
    gamma=0.99,
    num_moves=10_000):
   
    # In self-play, the same agent controlls both White and Black.
    # So we don't strictly need to assign roles 50/50 for "training" logic 
    # unless we were playing against a fixed opponent.
    # For pure self-play, the agent just sees state -> acts -> gets reward.
    
    games = [chess.Board() for _ in range(batch_size)]
    
    # Store trajectories: list of (log_prob, reward_to_go?)
    # For REINFORCE, we need to store log_probs per game until it finishes.
    # This is complex in a vector loop. 
    # Simpler V1: Just run batch simulation. Training logic needs memory buffer.
    
    active_games_indices = list(range(batch_size))
    
    for _ in tqdm(range(num_moves), desc="Self-Play Steps"):
        
        # 1. Filter only active games (redundant with simple restart logic, but good practice)
        # For this simple loop, we assume all slots in `games` list are always "active" (reset immediately)
        
        # 2. Get moves for ALL games (Agent plays effectively for side to move)
        # Note: sample_moves returns valid moves for whoever is to move.
        moves = agent_wrapper.agent.sample_moves(games)
        
        # 3. Apply moves & Check termination
        for i, move in enumerate(moves):
            board = games[i]
            if move is None: 
                # Should not happen if we reset immediately, but safety check
                games[i] = chess.Board()
                continue
                
            board.push(move)
            
            if board.is_game_over():
                result = board.result() # '1-0', '0-1', '1/2-1/2'
                # TODO: Parse result, assign reward, store trajectory, optimize
                
                games[i] = chess.Board() # Reset immediately
 

    



