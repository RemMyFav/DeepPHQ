# models/lstm/lstm_model.py

import torch
import torch.nn as nn

class DeepPHQLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, num_layers, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(input_size, hidden_dim)

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,                # (batch, seq_len, hidden_dim)
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )

        self.fc_out = nn.Linear(hidden_dim * 2, output_size)

    def forward(self, x):
        # x: (batch, seq_len) — raw token ids

        # ---- 1. embedding ----
        emb = self.embedding(x)                # (batch, seq_len, hidden_dim)

        # ---- 2. LSTM ----
        outputs, (h_n, c_n) = self.lstm(emb)   # outputs: (batch, seq, hidden_dim)

        # ---- 3. Mask padding (fix here!) ----
        # Use input ids to build mask — NOT embedding
        mask = (x != 0).float()                # (batch, seq)
        mask = mask.unsqueeze(-1)              # (batch, seq, 1)

        # zero out padded positions
        masked_outputs = outputs * mask

        # avoid divide by zero
        lengths = mask.sum(dim=1).clamp(min=1)   # (batch, 1)

        # ---- 4. Mean pooling w/ mask ----
        pooled = masked_outputs.sum(dim=1) / lengths   # (batch, hidden_dim)

        # ---- 5. Final regression ----
        out = self.fc_out(pooled)              # (batch, output_size)
        return out