import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import sys
import os
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from chess_net import ChessNet


class ImitationTrainer:
    def __init__(self, net: ChessNet, optimizer, dataset: Dataset, init_weights=None, batch_size=64):
        self.net = net
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss()
        
        if init_weights:
            self.net.load_state_dict(init_weights)
        
        train_size = int(len(dataset) * 0.8)
        eval_size = len(dataset) - train_size
        train_ds, eval_ds = random_split(dataset, [train_size, eval_size])
        
        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        self.eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)

    def train(self, num_epochs=10):
        self.net.train()
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            with tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
                for batch in pbar:
                    states, true_moves = batch
    
                    self.optimizer.zero_grad()
                    logits = self.net(states)
                    loss = self.criterion(logits, true_moves)
                    
                    loss.backward()
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    if num_batches % 10 == 0:
                        pbar.set_postfix({' loss': total_loss / num_batches})

            self.evaluate()

    def evaluate(self):
        self.net.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in self.eval_loader:
                states, true_moves = batch
                logits = self.net(states)
                loss = self.criterion(logits, true_moves)
                total_loss += loss.item()
                
        avg_loss = total_loss / len(self.eval_loader)
        print(f"Validation Loss: {avg_loss:.4f}")
        self.net.train()

if __name__ == "__main__":
    dataset_path = os.path.join(os.path.dirname(__file__), "imitation_data_1024_1600.pt") # TODO use larger?
    dataset = torch.load(dataset_path, weights_only=False)
    
    net = ChessNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    
    trainer = ImitationTrainer(net, optimizer, dataset, batch_size=64)

    num_epochs = 4
    trainer.train(num_epochs=num_epochs)
    
    save_path = os.path.join(os.path.dirname(__file__), f"chess_net_imitation_{num_epochs}.pth")
    torch.save(net.state_dict(), save_path)
    print(f"Model saved to {save_path}")


