import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import sys
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from RL.chess_net import ChessNet


class ImitationTrainer:
    def __init__(self, 
                net: ChessNet, 
                optimizer, 
                dataset: Dataset, 
                device: torch.device, 
                init_weights=None, 
                batch_size=64,
                num_epochs=5
                ):
        
        self.net = net.to(device)
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.device = device
        
        self.train_losses = []
        self.eval_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        
        if init_weights:
            self.net.load_state_dict(init_weights, strict=False)
        
        train_size = int(len(dataset) * 0.8)
        eval_size = len(dataset) - train_size
        train_ds, eval_ds = random_split(dataset, [train_size, eval_size])
        
        self.train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        self.eval_loader = DataLoader(eval_ds, batch_size=self.batch_size, shuffle=False)

    def train(self):
        self.net.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            num_batches = 0
            
            with tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}") as pbar:
                for batch in pbar:
                    states, true_moves = batch
                    states, true_moves = states.to(self.device), true_moves.to(self.device)
    
                    self.optimizer.zero_grad()
                    logits, _ = self.net(states)
                    loss = self.criterion(logits, true_moves)
                    
                    loss.backward()
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    if num_batches % 10 == 0:
                        pbar.set_postfix({' loss': total_loss / num_batches})
            
            epoch_avg_train_loss = total_loss / num_batches
            self.train_losses.append(epoch_avg_train_loss)
            print(f"Epoch {epoch+1} Training Loss: {epoch_avg_train_loss:.4f}")

            self.evaluate()

    def evaluate(self):
        self.net.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in self.eval_loader:
                states, true_moves = batch
                states, true_moves = states.to(self.device), true_moves.to(self.device)
                logits, _ = self.net(states)
                loss = self.criterion(logits, true_moves)
                total_loss += loss.item()
                
        avg_loss = total_loss / len(self.eval_loader)
        self.eval_losses.append(avg_loss)
        print(f"Validation Loss: {avg_loss:.4f}")

        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.best_model_state = {
                k: v.detach().cpu().clone()
                for k, v in self.net.state_dict().items()
            }

        self.net.train()

    def plot_losses(self, save_path):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.num_epochs + 1), self.train_losses, label='Training Loss')
        plt.plot(range(1, self.num_epochs + 1), self.eval_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        print(f"Loss plot saved to {save_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_imitation_games = 4096
    elo = 1600
    batch_size = 512 # 64
    num_epochs = 5 # 5
    
    
    dataset_path = os.path.join(os.path.dirname(__file__), "..", "data", f"imitation_data_{num_imitation_games}_{elo}.pt")
    dataset = torch.load(dataset_path, weights_only=False)
    
    net = ChessNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)
    trainer = ImitationTrainer(net, optimizer, dataset, device, batch_size=batch_size, num_epochs=num_epochs)

    trainer.train()
    
    plot_save_path = os.path.join(os.path.dirname(__file__), "..", "plots", f"imitation_{num_imitation_games}_loss_curve_{num_epochs}.png")
    trainer.plot_losses(save_path=plot_save_path)

    save_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", f"pre_trained_{num_imitation_games}_{elo}.pth")
    torch.save(trainer.best_model_state, save_path)


if __name__ == "__main__":
    main()
