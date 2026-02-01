import chess
import random
import torch
from tqdm import tqdm
import os
import logging
from datetime import datetime
import torch.nn.functional as F
from RL.chess_net import ChessNet
from RL.game_environment.game import Game
from RL.game_environment.chess_net_agent import ChessNetAgent, ChessNetHandler, Trajectory
from RL.eval.eval_net_vs_net import EvalHandler as NetVsNetEvalHandler
from RL.eval.eval_vs_stockfish import EvalHandler as StockfishEvalHandler


class ReinforceTrainer:
    def __init__(self,
                 weights_path_init_trainable: str,
                 weights_path_init_opponent: str, # frozen weights for opponent, incrementaly updated
                 weights_path_eval: str, # frozen weights for evaluation, incrementaly updated
                 device: torch.device,
                 num_games: int,
                 checkpoint_interval: int,
                 game_batch_size: int = 8,
                 minibatch_size: int = 32, 
                 update_rollout_size: int = 128,  
                 epochs: int = 5,
                 max_grad_norm: float = 1.0,
                 base_num_games: int = 10_000,
                 base_elo: int = 1500,
                 eval_elo: int = 1350, # incrementaly update? TODO?
                 gamma: float = 0.99, # discount rate
                 entropy_coef: float = 0.01, # entropy regularization coefficient
                 lr: float = 1e-4, # learning rate
                 lr_step_size: int = 1000, # number of steps between learning rate updates
                 lr_gamma: float = 0.9 # learning rate decay factor
                 ):
        
        
        
        log_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"log_{base_num_games}_{base_elo}_{num_games}_{timestamp}.txt")
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
        
        self.device = device
        logging.info(f"Running on Device: {self.device}")

        self.weights_path_init_trainable = weights_path_init_trainable
        self.weights_path_init_opponent = weights_path_init_opponent
        self.weights_path_eval = weights_path_eval
        
        self.base_num_games = base_num_games
        self.base_elo = base_elo
        self.eval_elo = eval_elo
        self.prev_best_checkpoint_path = None
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.num_games = num_games
        self.game_batch_size = game_batch_size # number of games run in parallel when collecting trajectories
        self.boards = [chess.Board() for _ in range(self.game_batch_size)]
        
        self.active_rollouts = [Trajectory() for _ in range(self.game_batch_size)]
        
        self.model_handler_trainable = ChessNetHandler(self.boards, ChessNet(), device, collect_trajectories=True, trajectories=self.active_rollouts)
        self.model_handler_trainable.model.load_state_dict(torch.load(weights_path_init_trainable), strict=False)
        self.model_handler_trainable.model.eval()
        
        # TODO: generalize to use a distribution of different opponent agents
        self.model_handler_opponent = ChessNetHandler(self.boards, ChessNet(), device, collect_trajectories=False, trajectories=None)
        self.model_handler_opponent.model.load_state_dict(torch.load(weights_path_init_opponent), strict=False)
        self.model_handler_opponent.model.eval()

        self.games = []
        self.training_agent_is_white = []
        for i in range(self.game_batch_size):
            is_white = random.random() < 0.5 # if true, trainable plays white
            self.training_agent_is_white.append(is_white)

            self.games.append(Game(
                self.boards[i], 
                ChessNetAgent(self.model_handler_trainable if is_white else self.model_handler_opponent, self.boards[i], i), 
                ChessNetAgent(self.model_handler_trainable if not is_white else self.model_handler_opponent, self.boards[i], i)
            ))
        
        # Assign colors to handlers based on the schedule
        trainable_colors = [chess.WHITE if is_white else chess.BLACK for is_white in self.training_agent_is_white]
        opponent_colors = [chess.BLACK if is_white else chess.WHITE for is_white in self.training_agent_is_white]
        
        self.model_handler_trainable.set_agent_colors(trainable_colors)
        self.model_handler_opponent.set_agent_colors(opponent_colors)

        self.minibatch_size = minibatch_size # batch size (number of states) for gradient update
        self.update_rollout_size = update_rollout_size # number of complete game rollouts to collect before updating with gradient on states and actions (moves)
        self.epochs = epochs # number of epochs to run for each update
        self.completed_rollouts = []
        self.checkpoint_interval = checkpoint_interval
        self.stockfish_path = os.path.join(os.path.dirname(__file__), "..", "stockfish", "stockfish.exe")
        self.eval_net_vs_net_results = {}
        self.eval_stockfish_results = {}
        self.checkpoint_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
        self.max_grad_norm = max_grad_norm

        self.best_stockfish_win_rate = -1.0
        self.best_stockfish_tie_rate = -1.0

        self.optimizer = torch.optim.Adam(self.model_handler_trainable.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma)

        num_params = sum(p.numel() for p in self.model_handler_trainable.model.parameters())
        logging.info(f"Model has {num_params} parameters")
    
    def train(self):
        num_turns = 0
        num_games_completed = 0
        last_checkpoint = 0
        
        while num_games_completed < self.num_games:
            for i, game in enumerate(self.games):
                if game is None:
                    continue
                game.make_turn()
                
                if game.board.is_game_over():
                    num_games_completed += 1
                    winner = game.get_winner()
                    if winner is None:
                        self.active_rollouts[i].reward = 0 # draw
                    elif winner == game.agent_white and isinstance(game.agent_white, ChessNetAgent):
                        if game.agent_white.model_handler == self.model_handler_trainable:
                            self.active_rollouts[i].reward = 1
                        else:
                            self.active_rollouts[i].reward = -1
                    elif winner == game.agent_black and isinstance(game.agent_black, ChessNetAgent):
                        if game.agent_black.model_handler == self.model_handler_trainable:
                            self.active_rollouts[i].reward = 1
                        else:
                            self.active_rollouts[i].reward = -1

                    self.completed_rollouts.append(self.active_rollouts[i])
                    self.active_rollouts[i] = Trajectory()
                    self.boards[i].reset() # reset board, agents/handlers stay the same
                
                num_turns += 1

            if len(self.completed_rollouts) >= self.update_rollout_size:
                logging.info(f"Running policy update after: {num_games_completed} games completed")
                # gather rollouts and convert to tensors
                all_states = []
                all_actions = []
                all_values = []
                traj_rewards = []
                traj_lengths = []

                for traj in self.completed_rollouts:
                    all_states.extend(traj.states)
                    all_actions.extend(traj.actions)
                    all_values.extend(traj.values)
                    traj_rewards.append(traj.reward)
                    traj_lengths.append(len(traj.states))
                
                logging.info(f"Number of states (moves) in rollout batch: {len(all_states)}")

                state_tensor = torch.stack(all_states)
                action_tensor = torch.stack(all_actions)
                value_tensor = torch.stack(all_values).squeeze(-1) # (N,)

                # Vectorized return calculation
                rewards_tensor = torch.tensor(traj_rewards, device=self.device, dtype=torch.float32)
                lengths_tensor = torch.tensor(traj_lengths, device=self.device, dtype=torch.long)
                
                # Create a temporary rewards tensor for the final rewards and use the number of moves within each rollout to discount and expand later on
                expanded_rewards = torch.repeat_interleave(rewards_tensor, lengths_tensor)
                powers = torch.cat([torch.arange(l - 1, -1, -1, device=self.device) for l in traj_lengths])

                return_tensor = expanded_rewards * (self.gamma ** powers)
                advantage_tensor = return_tensor - value_tensor # advantage function, actor critic style
                
                # Training Loop
                dataset_size = state_tensor.shape[0]
                indices = list(range(dataset_size))
                
                self.model_handler_trainable.model.train()
                # Capture initial parameters to measure update magnitude
                initial_params = [p.clone().detach() for p in self.model_handler_trainable.model.parameters()]
                for epoch in tqdm(range(self.epochs), desc="Rollout Batch Epochs", disable=True):
                    random.shuffle(indices)
                    for start_idx in tqdm(range(0, dataset_size, self.minibatch_size), desc="Minibatches", disable=True):
                        end_idx = min(start_idx + self.minibatch_size, dataset_size)
                        if end_idx - start_idx < self.minibatch_size / 2:
                            break
                        
                        batch_indices = indices[start_idx:end_idx]
                        
                        batch_states = state_tensor[batch_indices]
                        batch_actions = action_tensor[batch_indices]
                        batch_returns = return_tensor[batch_indices]
                        batch_advantages = advantage_tensor[batch_indices]
                        
                        logits, values = self.model_handler_trainable.model(batch_states)
                        values = values.squeeze(-1)
                        
                        # Policy Loss
                        probs = F.softmax(logits, dim=1)
                        log_probs = F.log_softmax(logits, dim=1)
                        
                        # Gather log probs for the specific actions taken
                        action_log_probs = log_probs.gather(1, batch_actions.unsqueeze(1)).squeeze(1)
                        
                        policy_loss = -(action_log_probs * batch_advantages).mean()
                        value_loss = F.mse_loss(values, batch_returns)
                        
                        # Entropy Loss (Regularization)
                        dist = torch.distributions.Categorical(probs)
                        entropy = dist.entropy().mean()
                        
                        loss = policy_loss + value_loss - self.entropy_coef * entropy
                        
                        self.optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model_handler_trainable.model.parameters(), self.max_grad_norm)
                        self.optimizer.step()
                
                self.model_handler_trainable.model.eval()
                # Calculate update magnitude
                final_params = [p for p in self.model_handler_trainable.model.parameters()]
                update_diff = [p_final - p_init for p_final, p_init in zip(final_params, initial_params)]
                update_magnitude = torch.norm(torch.stack([torch.norm(diff) for diff in update_diff]))
                logging.info(f"Update Magnitude (L2 Norm of param change): {update_magnitude.item():.6f}")

                if num_games_completed - last_checkpoint >= self.checkpoint_interval:
                    logging.info(f"Saving checkpoint at {num_games_completed} games completed")
                    checkpoint_path = os.path.join(self.checkpoint_dir, f"self_play_{num_games_completed}.pth")
                    torch.save(self.model_handler_trainable.model.state_dict(), checkpoint_path)
                    last_checkpoint = num_games_completed

                    # Eval Net vs Net
                    logging.info("Running Eval Net vs Net...")
                    net_vs_net_handler = NetVsNetEvalHandler(num_games=128, batch_size=128, weights_path_primary=checkpoint_path, weights_path_baseline=self.weights_path_eval)
                    net_vs_net_res = net_vs_net_handler.eval()
                    self.eval_net_vs_net_results[num_games_completed] = net_vs_net_res
                    logging.info(f"Net vs Net Results (Win Rate, Tie Rate, Loss Rate, AvgMoves): {net_vs_net_res}")

                    # Eval vs Stockfish
                    logging.info("Running Eval vs Stockfish...")
                    stockfish_handler = StockfishEvalHandler(num_games=128, batch_size=128, weights_path=checkpoint_path, stockfish_path=self.stockfish_path, stockfish_elo=self.eval_elo, stockfish_time_per_move=5)
                    stockfish_res = stockfish_handler.eval()
                    self.eval_stockfish_results[num_games_completed] = stockfish_res
                    logging.info(f"Stockfish Results (Win Rate, Tie Rate, Loss Rate, AvgMoves): {stockfish_res}")

                    win_rate, tie_rate, _, _ = stockfish_res                    
                    if win_rate > self.best_stockfish_win_rate or (win_rate == self.best_stockfish_win_rate and tie_rate > self.best_stockfish_tie_rate):
                        logging.info(f"New best model found! (Win Rate: {win_rate}, Tie Rate: {tie_rate})")
                        self.best_stockfish_win_rate = win_rate
                        self.best_stockfish_tie_rate = tie_rate
                        
                        # Rename to permanent best checkpoint
                        formatted_win_rate = f"{win_rate:.4f}".replace(".", "")
                        new_best_path = os.path.join(self.checkpoint_dir, f"post_trained_{self.base_num_games}_{self.base_elo}_{self.eval_elo}_{formatted_win_rate}.pth")
                        
                        if self.prev_best_checkpoint_path and os.path.exists(self.prev_best_checkpoint_path):
                            os.remove(self.prev_best_checkpoint_path)
                            
                        os.rename(checkpoint_path, new_best_path)
                        self.prev_best_checkpoint_path = new_best_path
                    else:
                        if os.path.exists(checkpoint_path):
                            os.remove(checkpoint_path)
                            
                self.completed_rollouts = []
                self.scheduler.step()
                logging.info(f"Stepped Scheduler. New Learning Rate: {self.scheduler.get_last_lr()[0]}")


            
def main():
    base_num_games = 10_000
    base_elo = 1500
    eval_elo = 1350
    weights_path_init_trainable = os.path.join(os.path.dirname(__file__), "..", "checkpoints", f"pre_trained_{base_num_games}_{base_elo}.pth")
    weights_path_init_opponent = weights_path_eval = os.path.join(os.path.dirname(__file__), "..", "checkpoints", f"pre_trained_5000_1750.pth")
    weights_path_eval = weights_path_init_opponent

    trainer = ReinforceTrainer(
        weights_path_init_trainable = weights_path_init_trainable,
        weights_path_init_opponent = weights_path_init_opponent,
        weights_path_eval = weights_path_eval,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        num_games=200_000, # TODO increase
        checkpoint_interval=4096,
        game_batch_size=512,
        minibatch_size=2048, # states per minibatch
        update_rollout_size=1024, # TODO TUNE or CHANGE to use number of moves(states) instead of full game trajectories
        epochs=4, # TUNE
        base_num_games=base_num_games,
        base_elo=base_elo,
        eval_elo=eval_elo
    )
    trainer.train()

if __name__ == "__main__":
    main()