import numpy as np
import torch
import pickle



# ============================================================================
# METHOD 1: Loading from common embedding formats
# ============================================================================

def load_embedding_file(filepath):
    """
    Load embeddings from various file formats.
    Detects format based on file extension.
    """
    if filepath.endswith('.npy'):
        # NumPy binary format
        embeddings = np.load(filepath)
        return embeddings
    
    elif filepath.endswith('.npz'):
        # Compressed NumPy format
        data = np.load(filepath)
        embeddings = data['embeddings']  # or whatever key is used
        return embeddings
    
    elif filepath.endswith('.pkl') or filepath.endswith('.pickle'):
        # Pickle format
        with open(filepath, 'rb') as f:
            embeddings = pickle.load(f)
        return embeddings
    
    elif filepath.endswith('.txt') or filepath.endswith('.vec'):
        # Text format (like GloVe or Word2Vec text format)
        embeddings, _ = load_text_embeddings(filepath)
        return embeddings
    
    else:
        raise ValueError(f"Unsupported file format: {filepath}")


def load_text_embeddings(filepath, encoding='utf-8'):
    """
    Load embeddings from text format (GloVe style).
    Handles punctuation and special characters properly.
    
    Expected format: word dim1 dim2 dim3 ... dimN
    """
    embeddings_dict = {}
    embedding_dim = None
    
    with open(filepath, 'r', encoding=encoding, errors='ignore') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            values = line.split()
            
            # The first token is the word, rest are the vector values
            word = values[0]
            
            try:
                # Convert remaining values to float vector
                vector = np.array(values[1:], dtype='float32')
                
                # Verify dimension consistency
                if embedding_dim is None:
                    embedding_dim = len(vector)
                    #print(f"Detected embedding dimension: {embedding_dim}")
                elif len(vector) != embedding_dim:
                    #print(f"Warning: Line {line_num} has inconsistent dimension "
                    #      f"({len(vector)} vs {embedding_dim}), skipping")
                    continue
                
                embeddings_dict[word] = vector
                
            except ValueError as e:
                #print(f"Warning: Could not parse line {line_num}: {line[:50]}... Error: {e}")
                continue
    
    print(f"Loaded {len(embeddings_dict)} word vectors (dimension: {embedding_dim})")
    return embeddings_dict, embedding_dim


# ============================================================================
# METHOD 2: Create embedding matrix aligned with your vocabulary
# ============================================================================

def create_embedding_matrix(embedding_file, vocab, embedding_dim=200):
    """
    Create embedding matrix aligned with your vocabulary.
    
    Args:
        training_data: Training data dataframe
        vocab_size: Number of words to put in the vocab
        embedding_file: Path to embedding file
        embedding_dim: Dimension of embeddings (e.g., 200, 300)
    
    Returns:
        embedding_matrix: NumPy array of shape (vocab_size, embedding_dim)
    """
    # Load the embedding file
    print(f"Loading embeddings from {embedding_file}...")

    
    # Try loading as numpy array first
    try:
        embeddings = np.load(embedding_file)
        print(f"Loaded embedding array with shape: {embeddings.shape}")
        
        # If it's a pre-built matrix, return it directly
        if embeddings.shape[0] == len(vocab):
            print("Embedding matrix matches vocabulary size!")
            return embeddings
        
        # Otherwise, we need word-to-vector mapping
        raise ValueError("Embedding matrix size doesn't match vocabulary")
        
    except:
        # Try loading as dictionary format
        if embedding_file.endswith('.txt') or embedding_file.endswith('.vec'):
            embeddings_dict, _ = load_text_embeddings(embedding_file)
        else:
            # Try pickle
            with open(embedding_file, 'rb') as f:
                embeddings_dict = pickle.load(f)
    
    print(f"Building embedding matrix for vocabulary size: {len(vocab)}")
    
    # Initialize embedding matrix with random values
    embedding_matrix = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    
    # Set padding token to zeros
    embedding_matrix[0] = np.zeros(embedding_dim)
    
    # Fill in embeddings for words in vocabulary
    found = 0
    for word, idx in vocab.items():
        if word in embeddings_dict:
            embedding_matrix[idx] = embeddings_dict[word]
            found += 1
    
    print(f"Found embeddings for {found}/{len(vocab)} words ({100*found/len(vocab):.1f}%)")
    
    return embedding_matrix


