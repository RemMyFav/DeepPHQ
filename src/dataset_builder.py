from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# ============================================================================
# MULTI TRANSCRIPT DATASETS
# ============================================================================


class Vocabulary:
    """Build vocabulary from text corpus."""
    
    def __init__(self, max_words=10000):
        self.max_words = max_words
        self.word2idx = {'<STR>': 0, '<END>': 1}
        self.idx2word = {0: '<STR>', 1: '<END>'}
        self.word_counts = Counter()
        
    def build_vocab(self, texts):
        """Build vocabulary from list of texts."""
        for text in texts:
            self.word_counts.update(text.split())
        
        # Get most common words
        most_common = self.word_counts.most_common(self.max_words - 2)
        
        # Add to vocabulary
        for idx, (word, count) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
    
    def encode(self, text):
        """Convert text to sequence of indices."""
        return [self.word2idx.get(word, 1) for word in text.split()]
    
    def __len__(self):
        return len(self.word2idx)

class TextDataset(Dataset):
    """PyTorch Dataset for text regression."""
    
    def __init__(self, texts, scores, vocab, max_len=557):
        # Handle different input types safely
        if isinstance(texts, pd.Series):
            self.texts = texts.reset_index(drop=True).tolist()
        elif isinstance(texts, np.ndarray):
            self.texts = texts.tolist()
        elif isinstance(texts, list):
            self.texts = texts
        else:
            self.texts = list(texts)
        
        if isinstance(scores, pd.Series):
            self.scores = scores.reset_index(drop=True).tolist()
        elif isinstance(scores, np.ndarray):
            self.scores = scores.tolist()
        elif isinstance(scores, list):
            self.scores = scores
        else:
            self.scores = list(scores)
        
        self.vocab = vocab
        self.max_len = max_len
        
        # Verify lengths match
        assert len(self.texts) == len(self.scores), \
            f"Length mismatch: {len(self.texts)} texts vs {len(self.scores)} scores"
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        # Validate index
        if idx >= len(self.texts):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.texts)}")
        
        # Encode text
        encoded = self.vocab.encode(self.texts[idx])
        
        # Pad or truncate to max_len
        if len(encoded) < self.max_len:
            encoded = encoded + [0] * (self.max_len - len(encoded))
        else:
            encoded = encoded[:self.max_len]
        
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(self.scores[idx], dtype=torch.float32)


def preprocess_data(transcripts, scores, max_words=10000, max_len=557, batch_size=32):
    """
    Preprocess text data.
    
    Args:
        transcripts: List/Series/Array of text documents
        scores: List/Series/Array of continuous target scores (regression values)
        max_words: Maximum vocabulary size (default: 10000)
        max_len: Maximum sequence length (default: 557, avg length from paper)
        batch_size: Batch size for DataLoader
    
    Returns:
        train_loader, test_loader, vocab
    """
    
    # Convert to lists if needed and ensure clean data
    if isinstance(transcripts, pd.Series):
        transcripts = transcripts.reset_index(drop=True).tolist()
    elif not isinstance(transcripts, list):
        transcripts = list(transcripts)
    
    if isinstance(scores, pd.Series):
        scores = scores.reset_index(drop=True).tolist()
    elif not isinstance(scores, list):
        scores = list(scores)
    
    # Validate data
    print(f"Preprocessing {len(transcripts)} samples")
    assert len(transcripts) == len(scores), \
        f"Length mismatch: {len(transcripts)} transcripts vs {len(scores)} scores"

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        transcripts, scores, test_size=0.25, random_state=42
    )
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Build vocabulary on training data only
    vocab = Vocabulary(max_words=max_words)
    vocab.build_vocab(X_train)
    
    print(f"Vocabulary size: {len(vocab)}")
    
    # Create datasets
    train_dataset = TextDataset(X_train, y_train, vocab, max_len)
    test_dataset = TextDataset(X_test, y_test, vocab, max_len)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, vocab


def normalize_scores(scores):
    """
    Normalize scores to have mean=0, std=1.
    """
    scores_array = np.array(scores)
    mean = scores_array.mean()
    std = scores_array.std()
    
    normalized = (scores_array - mean) / std
    
    print(f"Score normalization:")
    print(f"  Original: mean={mean:.2f}, std={std:.2f}, range=[{scores_array.min():.2f}, {scores_array.max():.2f}]")
    print(f"  Normalized: mean={normalized.mean():.2f}, std={normalized.std():.2f}, range=[{normalized.min():.2f}, {normalized.max():.2f}]")
    
    return normalized.tolist(), mean, std


def denormalize_predictions(predictions, mean, std):
    """
    Convert normalized predictions back to original scale.
    """
    return predictions * std + mean

if __name__ == "__main__":
    input_file = "../data/raw/transcripts/300_TRANSCRIPT.csv"
    sentences = read_csv_sentences(input_file)
    print(word_level(sentences, 512))
    print(sentence_level(sentences, 512))
    print(dialogue_level(sentences, 512))

