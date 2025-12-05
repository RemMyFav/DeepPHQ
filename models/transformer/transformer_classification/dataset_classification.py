import re
from pathlib import Path
from collections import Counter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch
import random
import nltk
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
# Tokenizer
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


# ============================================================
#  numpy string parser → ALWAYS return int list
# ============================================================

def parse_numpy_vector(x):
    """
    Convert a NumPy-style string "[0. 1. 0. 1. ...]" into a list[int].
    """
    if isinstance(x, str):
        clean = x.replace("[", "").replace("]", "")
        arr = np.fromstring(clean, sep=" ")
        return arr.astype(int).tolist()

    # already numeric: list, tuple, array
    return list(map(int, x))


# ============================================================
#  Training Dataset
# ============================================================

class DeepPHQDataset(Dataset):
    """
    Returns:
        input_ids : (L)
        label     : (8)  int classes 0–3
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
            "label": torch.tensor(item_scores, dtype=torch.long)  # ★ classification
        }


# ============================================================
# Sliding windows
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
#  Validation/Test Dataset
# ============================================================

class DeepPHQValDataset(Dataset):
    """
    Returns each PID expanded into sliding windows.
    Label stays the same 8-d class vector.
    """
    def __init__(self, df, vocab, max_length=512, stride=128):
        self.samples = []
        self.vocab = vocab

        for pid, text, item_scores in df[["PID", "Text", "PHQ_Score"]].values:

            item_scores = parse_numpy_vector(item_scores)
            words = text.lower().split()
            windows = sliding_windows(words, max_length, stride)

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
            "label": torch.tensor(item_scores, dtype=torch.long)
        }


# ============================================================
#  PID Split
# ============================================================

def split_by_pid(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
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


# ============================================================
# Balanced Sampler by total PHQ score
# ============================================================

def create_balanced_dataloader(dataset, batch_size=32):
    labels = []
    for i in range(len(dataset)):
        item_vec = dataset[i]["label"]       # tensor(8,)
        total_score = int(item_vec.sum().item())
        labels.append(total_score)

    class_counts = {}
    for t in labels:
        class_counts[t] = class_counts.get(t, 0) + 1

    weights = [1.0 / class_counts[t] for t in labels]

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