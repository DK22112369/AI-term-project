import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', save_path=None):
    """
    Plots and saves a confusion matrix heatmap.
    """
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix plot saved to {save_path}")
        plt.close()
    else:
        plt.show()

def plot_loss_curve(history, save_path=None):
    """
    Plots and saves the training and validation loss curve.
    
    Args:
        history (list of dict): List containing 'epoch', 'train_loss', 'val_loss'.
    """
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Loss curve saved to {save_path}")
        plt.close()
    else:
        plt.show()
