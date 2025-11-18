from collections import Counter

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
    """PyTorch Dataset for text classification."""
    
    def __init__(self, texts, scores, vocab, max_len=557):
        self.texts = texts
        self.scores = scores
        self.vocab = vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
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
    Args:
        texts: List of text documents
        scores: List of continuous target scores (regression values)
        max_words: Maximum vocabulary size (default: 10000)
        max_len: Maximum sequence length (default: 557, avg length from paper)
        batch_size: Batch size for DataLoader
    
    Returns:
        train_loader, test_loader, vocab
    """
    
    X_train, X_test, y_train, y_test = train_test_split(
        transcripts, scores, test_size=0.25, random_state=42
    )
    
    # Build vocabulary on training data only
    vocab = Vocabulary(max_words=max_words)
    vocab.build_vocab(X_train)
    
    # Create datasets
    train_dataset = TextDataset(X_train, y_train, vocab, max_len)
    test_dataset = TextDataset(X_test, y_test, vocab, max_len)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, vocab


if __name__ == "__main__":
    input_file = "../data/raw/transcripts/300_TRANSCRIPT.csv"
    sentences = read_csv_sentences(input_file)
    print(word_level(sentences, 512))
    print(sentence_level(sentences, 512))
    print(dialogue_level(sentences, 512))

