import math
import time
import random

# Pytorch packages
import torch
import torch.optim as optim
import torch.nn as nn

# Numpy
import numpy as np

# Tqdm progress bar
from tqdm import tqdm


import matplotlib.pyplot as plt

RANDOM_SEED = 0


def set_seed():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)


def set_seed_nb():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED + 1)


def deterministic_init(net: nn.Module):
    for p in net.parameters():
        if p.data.ndimension() >= 2:
            set_seed_nb()
            nn.init.xavier_uniform_(p.data)
        else:
            nn.init.zeros_(p.data)

def train(model, criterion, optimizer, train_loader, val_loader, epochs=20, lr=0.001, device='cpu'):
    """    
    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        epochs: Number of epochs (default: 20 as in paper)
        lr: Learning rate
        device: 'cuda' or 'cpu'
    
    Returns:
        Training history
    """
    model = model.to(device)
    
    history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': []}
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        
        for texts, scores in train_loader:
            texts, scores = texts.to(device), scores.to(device).view(-1, 1)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, scores)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            train_mae += torch.abs(outputs - scores).mean().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        
        with torch.no_grad():
            for texts, scores in val_loader:
                texts, scores = texts.to(device), scores.to(device).view(-1, 1)
                outputs = model(texts)
                loss = criterion(outputs, scores)
                
                val_loss += loss.item()
                val_mae += torch.abs(outputs - scores).mean().item()
        
        # Calculate epoch metrics
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_mae = train_mae / len(train_loader)
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_mae = val_mae / len(val_loader)
        
        history['train_loss'].append(epoch_train_loss)
        history['train_mae'].append(epoch_train_mae)
        history['val_loss'].append(epoch_val_loss)
        history['val_mae'].append(epoch_val_mae)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss (MSE): {epoch_train_loss:.4f}, Train MAE: {epoch_train_mae:.4f}")
        print(f"Val Loss (MSE): {epoch_val_loss:.4f}, Val MAE: {epoch_val_mae:.4f}")
        print("-" * 50)
        
        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            translation = model(source)
        translation = translation.reshape(-1, translation.shape[-1])
        target = target.reshape(-1)

        optimizer.zero_grad()
        loss = criterion(translation, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_description_str(
            "Batch: %d, Loss: %.4f" % ((batch_idx + 1), loss.item()))

    return total_loss, total_loss / len(dataloader)


def evaluate(model, test_loader, device='cpu'):
    """
    Evaluate regression model performance with proper shape handling.
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    model = model.to(device)
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for texts, scores in test_loader:
            texts = texts.to(device)
            outputs = model(texts)  # Shape: (batch_size, 1) or (batch_size,)
            
            # Flatten to 1D arrays
            if outputs.dim() > 1:
                outputs = outputs.squeeze()  # Remove extra dimensions
            if scores.dim() > 1:
                scores = scores.squeeze()
            
            all_predictions.extend(outputs.cpu().numpy().flatten())
            all_scores.extend(scores.cpu().numpy().flatten())
    
    # Convert to 1D numpy arrays
    all_predictions = np.array(all_predictions).flatten()
    all_scores = np.array(all_scores).flatten()
    
    # Verify shapes match
    print(f"Predictions shape: {all_predictions.shape}")
    print(f"Scores shape: {all_scores.shape}")
    
    if all_predictions.shape != all_scores.shape:
        raise ValueError(f"Shape mismatch: predictions {all_predictions.shape} vs scores {all_scores.shape}")
    
    # Calculate regression metrics
    mse = mean_squared_error(all_scores, all_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_scores, all_predictions)
    r2 = r2_score(all_scores, all_predictions)
    
    # Calculate correlation (now both are 1D arrays)
    correlation = np.corrcoef(all_scores, all_predictions)[0, 1]
    
    results = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'correlation': correlation,
        'predictions': all_predictions,
        'actual': all_scores
    }
    
    return results


def plot_curves(train_perplexity_history, valid_perplexity_history, filename):
    '''
    Plot learning curves with matplotlib. Training perplexity and validation perplexity are plot in the same figure
    :param train_perplexity_history: training perplexity history of epochs
    :param valid_perplexity_history: validation perplexity history of epochs
    :param filename: filename for saving the plot
    :return: None, save plot in the current directory
    '''
    epochs = range(len(train_perplexity_history))
    plt.plot(epochs, train_perplexity_history, label='train')
    plt.plot(epochs, valid_perplexity_history, label='valid')

    plt.xlabel('Epochs')
    plt.ylabel('Perplexity')
    plt.legend()
    plt.title('Perplexity Curve - '+filename)
    plt.savefig(filename+'.png')
    plt.show()
