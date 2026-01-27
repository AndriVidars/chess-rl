import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessNet(nn.Module):
    def __init__(self, embedding_dim=32, num_convs=3, num_linear=3):
        super(ChessNet, self).__init__()
        self.embedding_dim = embedding_dim
        
        # Embedding: 13 distinct piece types (0-12) 0=empty, 1-6 White, 7-12 Black
        self.embedding = nn.Embedding(num_embeddings=13, embedding_dim=embedding_dim)
        
        # Convolutions
        convs = []
        for i in range(num_convs):
            # Input starts at embedding_dim, then increases
            in_channels = (i + 1) * embedding_dim
            out_channels = (i + 2) * embedding_dim
            convs.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1))
        self.convolutions = nn.ModuleList(convs)
        
        # Fully Connected MLP
        # Calculate flattened size from last conv layer
        # Last conv out_channels = (num_convs + 1) * embedding_dim
        # Spatial dimensions stay 8x8
        conv_output_size = ((num_convs + 1) * embedding_dim) * 8 * 8
        
        linear_dim = 4096 # 64*64 moves
        mlp = []
        for i in range(num_linear):
            in_features = (conv_output_size + 1) if i == 0 else linear_dim # +1 for turn info
            out_features = linear_dim
            mlp.append(nn.Linear(in_features=in_features, out_features=out_features))
        self.mlp = nn.ModuleList(mlp)
        

    def forward(self, x):
        # x shape: (Batch, 65) -> [Turn, 64 squares]
        turn = x[:, 0].float().unsqueeze(1) # (Batch, 1)
        board_indices = x[:, 1:].long()     # (Batch, 64)
        
        # Embed the board: (Batch, 64, Embed_Dim)
        x = self.embedding(board_indices)
        
        # Reshape for Conv2d: (Batch, Embed_Dim, 8, 8)
        # Permute to (Batch, Channel, Height, Width)
        x = x.view(-1, 8, 8, self.embedding_dim).permute(0, 3, 1, 2)
        
        # Apply Convolutions
        for conv in self.convolutions:
            x = F.relu(conv(x))
        
        x = x.flatten(start_dim=1)
        
        # Concatenate Turn info
        x = torch.cat([x, turn], dim=1)
        
        # Apply MLP
        for i, layer in enumerate(self.mlp):
            x = layer(x)
            if i < len(self.mlp) - 1:
                x = F.relu(x)
        
        return x