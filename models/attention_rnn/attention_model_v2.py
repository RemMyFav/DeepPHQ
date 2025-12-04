import numpy as np
import pandas as pd

import nltk
from nltk.tokenize import word_tokenize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# A. Attention (Lower param attention)
class VectorAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.context = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, x, mask=None):
        u = torch.tanh(self.attn(x))
        scores = torch.matmul(u, self.context)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        alpha = torch.softmax(scores, dim=1)
        ctx = torch.sum(x * alpha.unsqueeze(-1), dim=1)

        return ctx, alpha

# B. Word-Level Encoder
class WordEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.embedding_dropout = nn.Dropout(0.3)

        self.gru = nn.GRU(
            embed_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
            num_layers=2,
            dropout=0.3
        )

        self.attention = VectorAttention(hidden_dim * 2)

    def forward(self, x, mask=None):
        emb = self.embedding_dropout(self.embedding(x))
        outputs, _ = self.gru(emb)
        ctx, alpha = self.attention(outputs, mask)
        return ctx, alpha

# C. Sentence-Level Encoder
class SentenceEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.gru = nn.GRU(
            hidden_dim * 2,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
            num_layers=2,
            dropout=0.3
        )

        self.attention = VectorAttention(hidden_dim * 2)

    def forward(self, x, mask=None):
        outputs, _ = self.gru(x)
        ctx, alpha = self.attention(outputs, mask)
        return ctx, alpha


class AttentionHanV2(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_dim,
                 hidden_dim,
                 num_classes,
                 pad_idx,
                 max_words,
                 max_sentences,
                 dropout):
        super().__init__()

        self.pad_idx = pad_idx
        self.max_words = max_words
        self.max_sentences = max_sentences

        self.word_encoder = WordEncoder(
            vocab_size, embed_dim, hidden_dim, pad_idx
        )

        self.sent_encoder = SentenceEncoder(hidden_dim)

        self.ln = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        """
        x: (B, S, W)
        """
        B, S, W = x.shape

        x = x.view(B * S, W)
        word_mask = (x != self.pad_idx)

        word_vecs, word_alpha = self.word_encoder(x, mask=word_mask)
        word_vecs = word_vecs.view(B, S, -1)

        sent_mask = (word_mask.view(B, S, W).sum(dim=2) > 0)
        doc_vecs, sent_alpha = self.sent_encoder(word_vecs, mask=sent_mask)

        out = self.ln(doc_vecs)
        out = self.dropout(out)
        logits = self.classifier(out)   # (B, C)

        return logits, word_alpha, sent_alpha
