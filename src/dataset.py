import re
from pathlib import Path
from collections import Counter
from torch.utils.data import Dataset
import torch


# -----------------------------
# Build vocabulary
# -----------------------------
def build_vocab(all_texts, min_freq=1):
    """
    all_texts: list of strings (already-processed inputs)
    """
    counter = Counter()
    for text in all_texts:
        words = text.lower().split()
        counter.update(words)

    vocab = {"<PAD>": 0, "<UNK>": 1}

    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)

    print(f"Vocab size = {len(vocab)}")
    return vocab


# -----------------------------
# Tokenizer
# -----------------------------
def tokenize(text, vocab, max_length=512):
    tokens = []

    for w in text.lower().split():
        tokens.append(vocab.get(w, vocab["<UNK>"]))

    # pad / truncate
    if len(tokens) < max_length:
        tokens += [vocab["<PAD>"]] * (max_length - len(tokens))
    else:
        tokens = tokens[:max_length]

    return tokens


# -----------------------------
# Dataset
# -----------------------------
class DeepPHQDataset(Dataset):

    def __init__(self, data, vocab, max_length=512):
        """
        data: List[(pid, text, phq_score)]
              text must already be processed into:
                  - 512 shuffled words  OR
                  - 512 word-equivalent sentences OR
                  - truncated dialogue

        vocab: word -> id
        """
        self.data = data
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pid, text, score = self.data[idx]

        token_ids = tokenize(text, self.vocab, self.max_length)

        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "label": torch.tensor(score, dtype=torch.float32),
            "pid": torch.tensor(pid, dtype=torch.long)
        }