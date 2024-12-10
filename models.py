import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class SASRec(nn.Module):
    def __init__(self, num_items, d_model, nhead, num_layers, dropout):
        super().__init__()
        self.num_items = num_items
        self.item_embeddings = nn.Embedding(num_items, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_model, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.d_model = d_model
        # output_layer 제거

    def forward(self, src):
        if torch.any(src < 0) or torch.any(src >= self.num_items):
            print(f"Warning: Input tensor contains out-of-range indices. Min: {src.min()}, Max: {src.max()}")
            src = torch.clamp(src, 0, self.num_items - 1)
        
        src = self.item_embeddings(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        # output_layer 적용 제거
        return output




class AlignmentNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        return self.projection(x)
