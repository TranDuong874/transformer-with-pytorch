from src.encoder import Encoder
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            Encoder(d_model, num_heads, d_ff) 
            for _ in range(n_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x