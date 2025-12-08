"""
CNN-based Text Classification Framework (PyTorch Implementation)
Based on BMC Medical Research Methodology paper:
"A comparative study on deep learning models for text classification"
https://doi.org/10.1186/s12874-022-01665-y
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import re


# ============================================================================
# CNN MODEL ARCHITECTURE
# ============================================================================

class CNNTextRegressor(nn.Module):
    """
    CNN text Regressor following BMC paper architecture.
    
    Model Architecture:
    - Embedding layer
    - Global Max Pooling
    - Dropout (0.3)
    - Dense output layer with sigmoid activation
    - Uses multiple filter sizes in covlutional layers to capture different n-gram patterns.

    """
    
    def __init__(self, vocab_size, embedding_dim=200, kernel_sizes=[3, 4, 5], num_filters=8, 
                 dropout_rate=0.3, pretrained_embedding = False, embedding_matrix=None, freeze_embeddings=True):
        super(CNNTextRegressor, self).__init__()
        
        # Create an Embedding layer, if a pretrained embedding layer has been supplied use it
        if pretrained_embedding == True:
            vocab_size, embedding_dim = embedding_matrix.shape
            
            # Embedding layer with pre-trained weights
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
            
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False

        else: # New embedding if none supplied
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Multiple convolutional layers with different kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=fs)
            for fs in kernel_sizes
        ])
        
        self.dropout = nn.Dropout(dropout_rate) 
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, 1)
        
    def forward(self, x):# x shape: (batch_size, seq_len)
        
        # Embedding: (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(x).transpose(1, 2)
        
        # Apply multiple convolutions and max pooling
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(pooled)
        
        # Concatenate all outputs
        concat = torch.cat(conv_outputs, dim=1)
        
        # Dropout then do sigmoid layer and pass
        dropped = self.dropout(concat)
        output = self.fc(dropped) # Output: (batch_size, 1)
        
        return output



