
from src.multihead_attention import MultiHeadAttention
from src.feed_forward import FeedForward
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.msk_attention = MultiHeadAttention(d_model, num_heads)
        self.attention     = MultiHeadAttention(d_model, num_heads)
        self.feed_fwd      = FeedForward(d_model, d_ff)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, mask):
        attn_output, _ = self.msk_attention(x, x, x, mask)
        x = x + attn_output
        x = self.norm1(x)

        attn_output, _ = self.attention(x, encoder_output, encoder_output)
        x = x + attn_output
        x = self.norm2(x)

        x = x + self.feed_fwd(x)
        x = self.norm3(x)

        return x
