import torch.nn as nn
from src.transformer_encoder import TransformerEncoder
from src.transformer_decoder import TransformerDecoder
from src.positional_encoder import PositionalEncoder
from src.input_embedder import InputEmbedder

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, n_layers, num_classes):
        super().__init__()
        self.input_embedder = InputEmbedder(vocab_size, d_model)
        self.positional_encoder = PositionalEncoder(d_model)
        
        self.encoder = TransformerEncoder(d_model, num_heads, d_ff, n_layers)
        self.decoder = TransformerDecoder(d_model, num_heads, d_ff, n_layers)

        self.fc_out = nn.Linear(d_model, num_classes) 
        self.softmax = nn.Softmax(dim=-1)  

    def forward(self, src, trg, mask=None):
        src = self.input_embedder(src)
        src = self.positional_encoder(src)
        
        encoder_output = self.encoder(src)
        output = self.decoder(trg, encoder_output, mask)
        
        output = self.fc_out(output)
        output = self.softmax(output)  
        
        return output

import torch.nn as nn

import torch.nn as nn
import torch
from src.transformer_encoder import TransformerEncoder
from src.transformer_decoder import TransformerDecoder
from src.positional_encoder import PositionalEncoder
from src.input_embedder import InputEmbedder

class TransformerSentiment(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, n_layers, num_classes=3):
        super().__init__()
        self.input_embedder = InputEmbedder(vocab_size, d_model)
        self.positional_encoder = PositionalEncoder(d_model)
        
        self.encoder = TransformerEncoder(d_model, num_heads, d_ff, n_layers)
        self.decoder = TransformerDecoder(d_model, num_heads, d_ff, n_layers)

        self.fc_out = nn.Linear(d_model, num_classes)  
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, src, trg, mask=None, labels=None):
        src = self.input_embedder(src)
        src = self.positional_encoder(src)
        trg = self.input_embedder(trg)
        trg = self.positional_encoder(trg)
        
        encoder_output = self.encoder(src)
        output = self.decoder(trg, encoder_output, mask)
        
        # Use mean pooling over sequence
        output = self.fc_out(output.mean(dim=1))  

        if labels is not None:
            loss = self.loss_fn(output, labels)
            return output, loss
        
        return output

