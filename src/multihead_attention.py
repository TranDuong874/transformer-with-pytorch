import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear layers for Q, K, V transformations
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, query, key, value, mask=None):
        batch_size, seq_length, _ = query.shape

        Q = self.W_q(query).view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            mask = mask.unsqueeze(1) 
            attention_scores.masked_fill_(mask == 0, float('-inf'))

        # Compute attention probabilities
        attention_probs = torch.softmax(attention_scores, dim=-1)

        attention_output = torch.matmul(attention_probs, V)  # (batch, num_heads, seq_len, d_k)

        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        output = self.W_o(attention_output)

        return output, attention_probs
