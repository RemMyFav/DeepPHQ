import math
import time
import random
import os

# Pytorch packages
import torch
import torch.optim as optim
import torch.nn as nn

# Numpy
import numpy as np

# Tqdm progress bar
from tqdm import tqdm


import matplotlib.pyplot as plt

seed = 0

def set_seed(seed):
    seed = seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def set_seed_nb():
    torch.manual_seed(seed)
    np.random.seed(seed + 1)


def deterministic_init(net: nn.Module):
    for p in net.parameters():
        if p.data.ndimension() >= 2:
            set_seed_nb()
            nn.init.xavier_uniform_(p.data)
        else:
            nn.init.zeros_(p.data)

def train(model, criterion, optimizer, train_loader, val_loader, 
          epochs=20, gradient_clip=1.0, device='cpu'):
    """
    Train CNN regression model.
    
    Args:
        model: PyTorch model
        criterion: Loss function (e.g., nn.MSELoss())
        optimizer: Optimizer (e.g., torch.optim.Adam)
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        epochs: Number of epochs
        device: Device to train on
    
    Returns:
        history: Dictionary with training metrics
    """

    
    model = model.to(device)
    
    history = {
        'train_loss': [], 
        'val_loss': [], 
        'train_mae': [], 
        'val_mae': []
    }
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_mae = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in progress:
            texts = batch["input_ids"].to(device)
            scores = batch["label"].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(texts).squeeze()
            
            loss = criterion(outputs, scores)

            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            
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
            for batch in val_loader:
                texts = batch["input_ids"].to(device)
                scores = batch["label"].to(device)
                
                outputs = model(texts).squeeze()
                
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
        print(f"  Train Loss: {epoch_train_loss:.4f}, Train MAE: {epoch_train_mae:.4f}")
        print(f"  Val Loss: {epoch_val_loss:.4f}, Val MAE: {epoch_val_mae:.4f}")
        print("-" * 50)
        
        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model
    if os.path.exists('best_model.pth'):
        model.load_state_dict(torch.load('best_model.pth'))
        print("Loaded best model weights")
    
    return history



def evaluate(model, test_loader, device='cpu'):
    """
    Evaluate regression model performance with proper shape handling.
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_scores = []

    
    
    with torch.no_grad():
        for batch in test_loader:
            texts = batch["input_ids"].to(device)
            scores = batch["label"].to(device).squeeze()

            outputs = model(texts).squeeze()
            
            all_predictions.extend(outputs.cpu().numpy().flatten())
            all_scores.extend(scores.cpu().numpy().flatten())
    
    # Convert to 1D numpy arrays
    all_predictions = np.array(all_predictions).flatten()
    all_scores = np.array(all_scores).flatten()
    
    
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

def plot_curves(train_history, valid_history, filename):

    epochs = range(len(train_history))
    plt.plot(epochs, train_history, label='train')
    plt.plot(epochs, valid_history, label='valid')

    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title('Loss Curve - '+filename)
    plt.savefig(filename+'.png')
    plt.show()
