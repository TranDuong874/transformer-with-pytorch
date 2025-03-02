import torch
import torch.nn as nn
import math

class PositionalEncoder(nn.Module):
    def __init__(self, embedding_dim, max_len=5000):
        super().__init__()

        # pos = position
        # i is the dimension in the paper
        pe  = torch.zeros(max_len, embedding_dim) # maxlen x embedding dim
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # row = max_len

        # 10000^(2i/d_model)
        denominator = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0)) / embedding_dim)

        # PE pos, 2i
        # PE pos, 2i+1
        pe[:, 0::2] = torch.sin(pos / denominator)
        pe[:, 1::2] = torch.cos(pos / denominator)


        pe = pe.unsqueeze(0)  # 1 x max_len x d_model
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
    