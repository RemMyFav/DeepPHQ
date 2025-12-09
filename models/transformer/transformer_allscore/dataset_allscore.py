import re
from pathlib import Path
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torch
import random
from torch.utils.data import DataLoader, WeightedRandomSampler
import nltk
import ast
import numpy as np
nltk.download("punkt")


# ============================================================
#  Vocabulary
# ============================================================

def build_vocab(all_texts, min_freq=1):
    counter = Counter()
    for text in all_texts:
        counter.update(text.lower().split())

    vocab = {"<PAD>": 0, "<UNK>": 1}

    for w, c in counter.items():
        if c >= min_freq:
            vocab[w] = len(vocab)

    print(f"Vocab size = {len(vocab)}")
    return vocab


# ============================================================
#  Tokenizer
# ============================================================

def tokenize(text, vocab, max_length=512):
    tokens = [vocab.get(w, vocab["<UNK>"]) for w in text.lower().split()]

    if len(tokens) < max_length:
        tokens += [vocab["<PAD>"]] * (max_length - len(tokens))
    else:
        tokens = tokens[:max_length]

    return tokens


# ============================================================
#  PHQ item names
# ============================================================

PHQ_ITEMS = [
    "PHQ_8NoInterest",
    "PHQ_8Depressed",
    "PHQ_8Sleep",
    "PHQ_8Tired",
    "PHQ_8Appetite",
    "PHQ_8Failure",
    "PHQ_8Concentrating",
    "PHQ_8Moving"
]


def parse_numpy_vector(x):
    """
    Convert a NumPy-style string "[0. 1. 0. 1. ...]" into a list of floats.
    """
    if isinstance(x, str):
        clean = x.replace("[", "").replace("]", "")
        arr = np.fromstring(clean, sep=" ")
        return arr.astype(float).tolist()

    # already numeric (list/tuple/np array)
    return list(x)

# ============================================================
#  Dataset for Training  (regular)
# ============================================================

class DeepPHQDataset(Dataset):
    """
    Stores (pid, text, item_scores)
    where item_scores = list of 8 ints.
    """
    def __init__(self, data, vocab, max_length=512):
        self.data = data
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pid, text, item_scores = self.data[idx]

        item_scores = parse_numpy_vector(item_scores)

        token_ids = tokenize(text, self.vocab, self.max_length)

        return {
            "pid": torch.tensor(pid, dtype=torch.long),
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "label": torch.tensor(item_scores, dtype=torch.float32)  # shape (8,)
        }

# ============================================================
#  Sliding window function
# ============================================================

def sliding_windows(words, window_size=512, stride=128):
    L = len(words)
    if L <= window_size:
        padded = words + ["<PAD>"] * (window_size - L)
        return [padded]

    windows = []
    for start in range(0, L - window_size + 1, stride):
        windows.append(words[start:start + window_size])
    return windows


# ============================================================
#  Val/Test Dataset (PID-level)
# ============================================================
class DeepPHQValDataset(Dataset):
    """
    Expands each PID into sliding windows, label still = 8 items.
    """
    def __init__(self, df, vocab, max_length=512, stride=128):
        import ast
        self.samples = []
        self.vocab = vocab

        for pid, text, item_scores in df[["PID", "Text", "PHQ_Score"]].values:


            item_scores = parse_numpy_vector(item_scores)

            words = text.lower().split()
            windows = sliding_windows(words, window_size=max_length, stride=stride)

            for w in windows:
                token_ids = [vocab.get(tok, vocab["<UNK>"]) for tok in w]
                self.samples.append((pid, token_ids, item_scores))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pid, token_ids, item_scores = self.samples[idx]
        return {
            "pid": torch.tensor(pid, dtype=torch.long),
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "label": torch.tensor(item_scores, dtype=torch.float32)  # shape (8,)
        }

# ============================================================
#  PID split
# ============================================================

def split_by_pid(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1) < 1e-6

    pids = list(df["PID"].unique())
    random.seed(seed)
    random.shuffle(pids)

    total = len(pids)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_pids = pids[:train_end]
    val_pids = pids[train_end:val_end]
    test_pids = pids[val_end:]

    return (
        df[df["PID"].isin(train_pids)],
        df[df["PID"].isin(val_pids)],
        df[df["PID"].isin(test_pids)]
    )



def create_balanced_dataloader(dataset, batch_size=32):
    labels = []
    for i in range(len(dataset)):
        item_vec = dataset[i]["label"]        # tensor(8,)
        total_score = float(item_vec.sum().item())
        labels.append(total_score)

    class_counts = {}
    for lab in labels:
        class_counts[lab] = class_counts.get(lab, 0) + 1

    weights = [1.0 / class_counts[lab] for lab in labels]

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler
    )