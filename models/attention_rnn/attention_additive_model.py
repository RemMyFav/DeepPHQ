import numpy as np
import pandas as pd

import nltk
from nltk.tokenize import word_tokenize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class SimpleRNN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        pass