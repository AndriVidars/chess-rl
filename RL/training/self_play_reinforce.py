import chess
import random
import torch
from tqdm import tqdm
import os
import torch.nn.functional as F
from RL.chess_net import ChessNet
from RL.game_environment.game import Game
from RL.game_environment.chess_net_agent import ChessNetAgent, ChessNetHandler, Trajectory
from RL.eval.eval_net_vs_net import EvalHandler as NetVsNetEvalHandler
from RL.eval.eval_vs_stockfish import EvalHandler as StockfishEvalHandler


class ReinforceTrainer:
    # TODO: change and tune defaults
    # TODO: add clipping/kl control
    # TODO: gradient update triggered by number of moves(states), not full game rollouts?
    def __init__(self,
                 weights_path: str,
                 device: torch.device,
                 num_games: int,
                 checkpoint_interval: int,
                 game_batch_size: int = 8,
                 minibatch_size: int = 32, 
                 update_rollout_size: int = 128,  
                 epochs: int = 5,
                 max_grad_norm: float = 1.0,
                 ):
        
        self.device = device
        print(f"\n\nRunning on Device: {self.device}\n\n")

        self.init_weights_path = weights_path
        self.num_games = num_games
        self.game_batch_size = game_batch_size # number of games run in parallel when collecting trajectories
        self.boards = [chess.Board() for _ in range(self.game_batch_size)]
        
        self.active_rollouts = [{chess.WHITE: Trajectory(), chess.BLACK: Trajectory()} for _ in range(self.game_batch_size)]
        self.model_handler = ChessNetHandler(self.boards, ChessNet(), device, collect_trajectories=True, trajectories=self.active_rollouts)
        self.model_handler.model.load_state_dict(torch.load(weights_path), strict=False)
        self.model_handler.model.eval()
        
        self.games = [
            Game(
                self.boards[i], 
                ChessNetAgent(self.model_handler, self.boards[i], i), 
                ChessNetAgent(self.model_handler, self.boards[i], i)
            ) 
            for i in range(self.game_batch_size)
        ]
        
        self.minibatch_size = minibatch_size # batch size for gradient update
        self.update_rollout_size = update_rollout_size # number complete game rollouts to collect before updating with gradient on moves
        self.epochs = epochs # number of epochs to run for each update
        self.completed_rollouts = []
        self.checkpoint_interval = checkpoint_interval
        self.stockfish_path = os.path.join(os.path.dirname(__file__), "..", "stockfish", "stockfish.exe")
        self.eval_net_vs_net_results = {}
        self.eval_stockfish_results = {}
        self.checkpoint_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
        self.max_grad_norm = max_grad_norm

        self.optimizer = torch.optim.Adam(self.model_handler.model.parameters(), lr=1e-4)

        num_params = sum(p.numel() for p in self.model_handler.model.parameters())
        print(f"Model has {num_params} parameters\n")
    
    def train(self):
        num_games_completed = 0
        last_checkpoint = 0
        
        while num_games_completed < self.num_games:
            for i, game in enumerate(self.games):
                if game is None:
                    continue
                game.make_turn()
                
                if game.board.is_game_over():
                    num_games_completed += 1
                    results = game.get_result()
                    if results == chess.WHITE:
                        self.active_rollouts[i][chess.WHITE].reward = 1
                        self.active_rollouts[i][chess.BLACK].reward = -1
                    elif results == chess.BLACK:
                        self.active_rollouts[i][chess.WHITE].reward = -1
                        self.active_rollouts[i][chess.BLACK].reward = 1
                    else:
                        self.active_rollouts[i][chess.WHITE].reward = 0
                        self.active_rollouts[i][chess.BLACK].reward = 0

                    self.completed_rollouts.append(self.active_rollouts[i])
                    self.active_rollouts[i] = {chess.WHITE: Trajectory(), chess.BLACK: Trajectory()}

                    self.boards[i].reset()
                    if num_games_completed < self.num_games:
                        self.games[i] = Game(self.boards[i], ChessNetAgent(self.model_handler, self.boards[i], i), ChessNetAgent(self.model_handler, self.boards[i], i))
                    else:
                        self.games[i] = None  


            if len(self.completed_rollouts) >= self.update_rollout_size:
                print(f"Completed {len(self.completed_rollouts)} game rollouts")

                # gather rollouts and convert to tensors
                all_states = []
                all_actions = []
                all_returns = []
                all_values = []

                for rollout in self.completed_rollouts:
                    for traj in rollout.values():
                        all_states.extend(traj.states)
                        all_actions.extend(traj.actions)
                        all_values.extend(traj.values)
                        all_returns.extend([traj.reward] * len(traj.states))
                
                print(f"Number of moves in rollout batch: {len(all_states)}")

                state_tensor = torch.stack(all_states)
                action_tensor = torch.stack(all_actions)
                value_tensor = torch.stack(all_values).squeeze(-1) # (N,)
                return_tensor = torch.tensor(all_returns, device=self.device, dtype=torch.float32)

                # Compute Advantage
                # A = R - V
                advantage_tensor = return_tensor - value_tensor
                
                # Training Loop
                dataset_size = state_tensor.shape[0]
                indices = list(range(dataset_size))
                
                self.model_handler.model.train()
                # Capture initial parameters to measure update magnitude
                initial_params = [p.clone().detach() for p in self.model_handler.model.parameters()]
                for epoch in tqdm(range(self.epochs), desc="Rollout Batch Epochs"):
                    random.shuffle(indices)
                    for start_idx in tqdm(range(0, dataset_size, self.minibatch_size), desc="Minibatches", disable=False):
                        end_idx = min(start_idx + self.minibatch_size, dataset_size)
                        batch_indices = indices[start_idx:end_idx]
                        
                        batch_states = state_tensor[batch_indices]
                        batch_actions = action_tensor[batch_indices]
                        batch_returns = return_tensor[batch_indices]
                        batch_advantages = advantage_tensor[batch_indices]
                        
                        logits, values = self.model_handler.model(batch_states)
                        values = values.squeeze(-1)
                        
                        # Policy Loss
                        log_probs = F.log_softmax(logits, dim=1)
                        # Gather log probs for the specific actions taken
                        action_log_probs = log_probs.gather(1, batch_actions.unsqueeze(1)).squeeze(1)
                        
                        policy_loss = -(action_log_probs * batch_advantages).mean()
                        
                        # Value Loss
                        value_loss = F.mse_loss(values, batch_returns)
                        
                        loss = policy_loss + value_loss
                        
                        self.optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model_handler.model.parameters(), self.max_grad_norm)
                        self.optimizer.step()
                
                self.model_handler.model.eval()
                # Calculate update magnitude
                final_params = [p for p in self.model_handler.model.parameters()]
                update_diff = [p_final - p_init for p_final, p_init in zip(final_params, initial_params)]
                update_magnitude = torch.norm(torch.stack([torch.norm(diff) for diff in update_diff]))
                print(f"Update Magnitude (L2 Norm of param change): {update_magnitude.item():.6f}\n")

                if num_games_completed - last_checkpoint >= self.checkpoint_interval:
                    print(f"Saving checkpoint at {num_games_completed} games completed")
                    checkpoint_path = os.path.join(self.checkpoint_dir, f"self_play_{num_games_completed}.pth")
                    torch.save(self.model_handler.model.state_dict(), checkpoint_path)
                    last_checkpoint = num_games_completed

                    # Eval Net vs Net
                    print("Running Eval Net vs Net...")
                    net_vs_net_handler = NetVsNetEvalHandler(num_games=512, batch_size=64, weights_path_primary=checkpoint_path, weights_path_baseline=self.init_weights_path)
                    net_vs_net_res = net_vs_net_handler.eval()
                    self.eval_net_vs_net_results[num_games_completed] = net_vs_net_res
                    print(f"Net vs Net Results (Win Rate, Tie Rate, Loss Rate, AvgMoves): {net_vs_net_res}")

                    # Eval vs Stockfish
                    print("Running Eval vs Stockfish...")
                    stockfish_handler = StockfishEvalHandler(num_games=512, batch_size=64, weights_path=checkpoint_path, stockfish_path=self.stockfish_path, stockfish_elo=1350, stockfish_time_per_move=10)
                    stockfish_res = stockfish_handler.eval()
                    self.eval_stockfish_results[num_games_completed] = stockfish_res
                    print(f"Stockfish Results (Win Rate, Tie Rate, Loss Rate, AvgMoves): {stockfish_res}")
                
                self.completed_rollouts = []


            
def main():
    trainer = ReinforceTrainer(
        weights_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "pre_trained_4096_1600.pth"),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        num_games=100_000, # TODO increase
        checkpoint_interval=10_000,
        game_batch_size=64,
        minibatch_size=64,
        update_rollout_size=128, # TODO TUNE or CHANGE to use number of moves(states) instead of full game trajectories
        epochs=3 # TUNE
    )
    trainer.train()

if __name__ == "__main__":
    main()