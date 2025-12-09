import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


# =========================
# Positional Encoding
# =========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

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
        return self.dropout(x + self.pe[:, :x.size(1), :])
    


# =========================
# Attention Pooling
# =========================
class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.att = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        weights = torch.softmax(self.att(x), dim=1)
        pooled = (weights * x).sum(dim=1)
        return pooled



# =========================
# DeepPHQ Transformer — Ordinal version
# =========================
class DeepPHQTransformer(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_dim,
        nhead,
        num_layers,
        dropout=0.1,
        num_items=8,
        num_levels=4          # ordinal labels: 0,1,2,3 → 3 boundaries
    ):
        super().__init__()

        self.num_items = num_items
        self.num_levels = num_levels
        self.num_boundaries = num_levels - 1      # 4 → 3 boundaries

        total_output_dim = num_items * self.num_boundaries   # 8 * 3 = 24

        # embedding + transformer
        self.embedding = nn.Embedding(input_size, hidden_dim)
        self.emb_norm = nn.LayerNorm(hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)

        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)
        self.final_norm = nn.LayerNorm(hidden_dim)

        # Attention pooling
        self.att_pool = AttentionPooling(hidden_dim)
        self.pool_norm = nn.LayerNorm(hidden_dim)

        # CORAL ordinal head
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, total_output_dim)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.emb_norm(x)

        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.final_norm(x)

        pooled = self.att_pool(x)
        pooled = self.pool_norm(pooled)

        logits = self.fc_out(pooled)  # (B, 24)

        # reshape into (B, 8, 3)
        logits = logits.view(-1, self.num_items, self.num_boundaries)

        return logits