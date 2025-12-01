import re
from pathlib import Path
from collections import Counter
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

import random

import nltk
nltk.download("punkt")
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

# Helper function for HAN
def split_into_sentences(text):
    return nltk.sent_tokenize(text)

def split_into_words(sentence):
    return nltk.word_tokenize(sentence)

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

# HAN specific datasets
class HANDataset(Dataset):
    def __init__(self, df, vocab, max_sentences=10, max_words=30):
        self.vocab = vocab
        self.max_sentences = max_sentences
        self.max_words = max_words
        self.data = []

        for pid, text, label in df[["PID","Text","PHQ_Score"]].values:
            sentences = split_into_sentences(text)

            sent_word_ids = []
            for sent in sentences[:max_sentences]:
                words = split_into_words(sent)
                word_ids = [
                    vocab.get(token.lower(), vocab["<UNK>"])
                    for token in words[:max_words]
                ]
                # pad words
                word_ids += [vocab["<PAD>"]] * (max_words - len(word_ids))
                sent_word_ids.append(word_ids)

            # pad sentences
            while len(sent_word_ids) < max_sentences:
                sent_word_ids.append([vocab["<PAD>"]] * max_words)

            sent_word_ids = torch.tensor(sent_word_ids)
            self.data.append((pid, sent_word_ids, float(label)))

    def __getitem__(self, idx):
        return {
            "pid": self.data[idx][0],
            "input_ids": self.data[idx][1],
            "label": torch.tensor(int(self.data[idx][2]), dtype=torch.long),
        }

    def __len__(self):
        return len(self.data)

    
def create_balanced_dataloader(dataset, batch_size=32):
    """
    dataset: DeepPHQDataset
        - dataset[i]["label"] must exist and be PHQ score
    """

    # Step 1: count frequencies
    labels = [float(dataset[i]["label"]) for i in range(len(dataset))]
    
    # Convert to classes if needed (e.g. regression → buckets)
    # But here we treat each integer PHQ score as a class
    class_counts = {}
    for lab in labels:
        class_counts[lab] = class_counts.get(lab, 0) + 1

    # Step 2: compute weights (inverse freq)
    weights = [1.0 / class_counts[lab] for lab in labels]

    # Step 3: create sampler
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),  # same number of samples per epoch
        replacement=True
    )

    # Step 4: return DataLoader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler
    )

def split_by_pid(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    # 1. collect unique PIDs
    pids = list(df["PID"].unique())

    # 2. set seed for reproducibility
    random.seed(seed)

    # 3. shuffle
    random.shuffle(pids)

    # 4. assign splits
    total = len(pids)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_pids = pids[:train_end]
    val_pids   = pids[train_end:val_end]
    test_pids  = pids[val_end:]

    # 5. select rows
    train_df = df[df["PID"].isin(train_pids)]
    val_df   = df[df["PID"].isin(val_pids)]
    test_df  = df[df["PID"].isin(test_pids)]

    return train_df, val_df, test_df

def sliding_windows(words, window_size=512, stride=128):
    """
    Split a list of words into overlapping fixed-size windows.
    """
    windows = []
    L = len(words)

    if L <= window_size:
        # pad to full length
        padded = words + ["<PAD>"] * (window_size - L)
        return [padded]

    for start in range(0, L - window_size + 1, stride):
        win = words[start:start + window_size]
        windows.append(win)

    return windows

class DeepPHQValDataset(Dataset):
    """
    Deterministic evaluation dataset using sliding windows.
    """

    def __init__(self, df, vocab, max_length=512, stride=128):
        self.samples = []
        self.vocab = vocab
        self.max_length = max_length

        for pid, text, score in df[["PID", "Text", "PHQ_Score"]].values:
            words = text.lower().split()
            windows = sliding_windows(words, window_size=max_length, stride=stride)

            for w in windows:
                token_ids = [vocab.get(tok, vocab["<UNK>"]) for tok in w]
                self.samples.append((pid, token_ids, float(score)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pid, token_ids, score = self.samples[idx]
        return {
            "pid": torch.tensor(pid, dtype=torch.long),
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "label": torch.tensor(score, dtype=torch.float32)
        }