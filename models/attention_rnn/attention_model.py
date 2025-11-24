
import numpy as np
import pandas as pd

import nltk
from nltk.tokenize import word_tokenize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============================================================================
# Attention RNN Implementation
# ============================================================================

# A. Attention (Additive implementation)
class HierarchicalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn_linear = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.context_vec = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x, mask=None):
        # x: (batch, time, hidden)
        u = torch.tanh(self.attn_linear(x))          
        scores = self.context_vec(u).squeeze(-1)     

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        alpha = torch.softmax(scores, dim=1)          
        context = torch.sum(x * alpha.unsqueeze(-1), dim=1)
        return context
    
# B. Word encoder + word attention
class WEncoder(nn.Module):
    def __init__(
            self,
            vocab_size, 
            embed_dim,
            hidden_dim,
            pad_idx
        ):

        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = HierarchicalAttention(hidden_dim * 2)


    def forward(self, x):
        word_embed = self.word_embedding(x)
        hierarchy, hidden_val = self.gru(word_embed)
        return self.attention(hierarchy)

# C. Sentence encoder + sentence embedding
class SEncoder(nn.Module):
    def __init__(
            self,
            hidden_dim
    ):
        super().__init__()
        self.gru = nn.GRU(hidden_dim*2, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = HierarchicalAttention(hidden_dim * 2)
    
    def forward(self,x):
        hierarchy, _ = self.gru(x)
        return self.attention(hierarchy)

class AttentionRNN(nn.Module):
    def __init__(
            self,
            vocab_size,
            embed_dim,
            hidden_dim,
            num_classes,
            pad_idx,
            max_words,
            max_sentences
        ):

        super().__init__()

        self.max_words = max_words
        self.max_sentences = max_sentences

        # Word-level GRU + Attention
        self.word_encoder = WEncoder(vocab_size, embed_dim, hidden_dim, pad_idx)
        # Sentence-level GRU + Attention
        self.sentence_encoder =  SEncoder(hidden_dim)
        # Final document vector → classifier
        self.classifier = nn.Linear(hidden_dim*2, num_classes)
    
    def forward(self, x):
        B, S, W = x.shape
        x = x.view(B*S, W)

        word_enc = self.word_encoder(x)
        word_enc = word_enc.view(B, S, -1)

        sentence_enc = self.sentence_encoder(word_enc)
        out = self.classifier(sentence_enc)
        return out.squeeze(1)