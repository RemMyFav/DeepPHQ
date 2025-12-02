import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ============================
# 🔥 Attention Pooling Layer
# ============================
class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.att = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (B, L, H)
        weights = torch.softmax(self.att(x), dim=1)   # (B, L, 1)
        pooled = (weights * x).sum(dim=1)             # (B, H)
        return pooled


class DeepPHQTransformer(nn.Module):

    def __init__(self, 
                 input_size, 
                 hidden_dim, 
                 nhead, 
                 num_layers, 
                 dropout=0.1,
                 output_size=8):
        super().__init__()

        self.embedding = nn.Embedding(input_size, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)

        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)


        self.att_pool = AttentionPooling(hidden_dim)


        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_size)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)

        x = self.att_pool(x)

        return self.fc_out(x)