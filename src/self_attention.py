import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super(SelfAttention, self).__init__()
        self.d_k = d_model # Dim of keys/queries

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # batch x seq_length x d_model
        # transpose K so that Q = batch x seq_len x d_model 
        # and K = batch x d_model x seq_len
        # Attention score = batch x seq_len x seq_len
        attention_score = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        attention_probs  = torch.softmax(attention_score, dim=-1) # Softmax on last dimension
        output = torch.matmul(attention_probs, V) # Weighted sum of values

        return output, attention_probs