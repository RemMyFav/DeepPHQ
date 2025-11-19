import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return self.dropout(x)


class DeepPHQTransformer(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, nhead, num_layers, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(input_size, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)

        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True  # <<< VERY IMPORTANT so we don't need to permute dims
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)

        self.fc_out = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len)
        x = self.embedding(x)   # (batch, seq_len, hidden_dim)
        x = self.pos_encoder(x) # add positional encodings
        x = self.transformer_encoder(x)  # (batch, seq_len, hidden_dim)

        # Regression → use mean pooling (much more stable than last token)
        x = x.mean(dim=1)

        out = self.fc_out(x)   # (batch, output_size)
        return out