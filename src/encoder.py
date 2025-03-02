import torch
import torch.nn as nn
from src.multihead_attention import MultiHeadAttention
from src.feed_forward import FeedForward

class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)  # Fix class name
        self.feed_fwd  = FeedForward(d_model, d_ff)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x) 
        x = x + attn_output 
        x = self.norm1(x)  

        ff_output = self.feed_fwd(x)
        x = x + ff_output 
        x = self.norm2(x)  
        return x
    