import torch.nn as nn
import torch
from src.decoder import Decoder

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, n_layers):
        super().__init__()
        self.decoder_layers = nn.ModuleList([
            Decoder(d_model, num_heads, d_ff)
            for _ in range(n_layers)
        ])
    
    def forward(self, x, encoder_output, mask=None):
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, mask)  
        return x
