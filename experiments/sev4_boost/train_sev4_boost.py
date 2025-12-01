import argparse
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import joblib

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.preprocess_us_accidents import (
    load_full_dataset, 
    stratified_sample, 
    clean_and_engineer_features, 
    fit_feature_transformers,
    transform_with_preprocessors,
    time_based_split,
    save_preprocessors
)
from models.sev4_boost.crash_severity_boosted import CrashSeverityBoosted
from utils.metrics import evaluate_metrics, save_metrics
from utils.common import set_seed

# --- Custom Losses ---

class FocalLoss(nn.Module):
    """
    Focal Loss with support for list-based alpha (per-class weights).
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        if alpha is not None:
            if isinstance(alpha, list):
                self.alpha = torch.tensor(alpha).float()
            else:
                self.alpha = alpha
        else:
            self.alpha = None
            
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            at = self.alpha.gather(0, targets)
            loss = at * (1 - pt) ** self.gamma * ce_loss
        else:
            loss = (1 - pt) ** self.gamma * ce_loss
            
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# --- Training Logic ---

def get_criterion(loss_type, y_train, device, config):
    """
    Factory for Loss Function.
    """
    if loss_type == "focal":
        alpha = config.get("alpha", [0.25, 0.25, 0.25, 0.25])
        gamma = config.get("gamma", 2.0)
        print(f"Using Focal Loss: gamma={gamma}, alpha={alpha}")
        return FocalLoss(alpha=alpha, gamma=gamma).to(device)
        
    elif loss_type == "ce":
        # Calculate balanced weights
        classes = np.unique(y_train.cpu().numpy())
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train.cpu().numpy())
        
        # Apply Sev4 Boost Factor
        boost_factor = config.get("sev4_boost_factor", 1.0)
        if len(weights) == 4:
            print(f"Original Weights: {weights}")
            weights[3] *= boost_factor # Boost Severity 4 (Index 3)
            print(f"Boosted Weights: {weights}")
            
        class_weights = torch.FloatTensor(weights).to(device)
        return nn.CrossEntropyLoss(weight=class_weights)
        
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

def train_sev4_boost(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Sev4 Boost Training on {device}")
    print(json.dumps(config, indent=4))
    
    # 1. Data Loading (Reusing existing pipeline)
    # We use a fixed sample fraction for this experiment or load from config
    data_path = "data/raw/US_Accidents_March23.csv"
    if not os.path.exists(data_path):
         data_path = "US_Accidents_March23.csv" # Fallback
         
    df = load_full_dataset(data_path)
    df = stratified_sample(df, frac=0.1) # Fixed 10% for speed/consistency
    df = clean_and_engineer_features(df)
    
    # Time Split
    df_train, df_val, df_test = time_based_split(df)
    
    # Preprocessing
    bundle = fit_feature_transformers(df_train)
    os.makedirs("models/sev4_boost", exist_ok=True)
    save_preprocessors(bundle, "models/sev4_boost/preprocessors.joblib")
    
    X_t_train, X_w_train, X_r_train, X_s_train, y_train = transform_with_preprocessors(df_train, bundle, device)
    X_t_val, X_w_val, X_r_val, X_s_val, y_val = transform_with_preprocessors(df_val, bundle, device)
    # Test set loaded later to save VRAM if needed, or now
    X_t_test, X_w_test, X_r_test, X_s_test, y_test = transform_with_preprocessors(df_test, bundle, device)
    
    # Datasets
    train_dataset = TensorDataset(X_t_train, X_w_train, X_r_train, X_s_train, y_train)
    val_dataset = TensorDataset(X_t_val, X_w_val, X_r_val, X_s_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])
    
    # 2. Model Setup
    input_dims = {
        'temporal': X_t_train.shape[1],
        'weather': X_w_train.shape[1],
        'road': X_r_train.shape[1],
        'spatial': X_s_train.shape[1]
    }
    
    model = CrashSeverityBoosted(input_dims, num_classes=4).to(device)
    criterion = get_criterion(config["loss_type"], y_train, device, config)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    
    # 3. Training Loop
    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = config.get("patience", 10)
    
    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0
        for xt, xw, xr, xs, y in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            optimizer.zero_grad()
            inputs = {'temporal': xt, 'weather': xw, 'road': xr, 'spatial': xs}
            out = model(inputs)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        if (epoch + 1) % config.get("val_freq", 1) == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for xt, xw, xr, xs, y in val_loader:
                    inputs = {'temporal': xt, 'weather': xw, 'road': xr, 'spatial': xs}
                    out = model(inputs)
                    loss = criterion(out, y)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
            
            # Checkpoint
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), "models/sev4_boost/best_model.pt")
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience_limit:
                print("Early stopping triggered.")
                break
                
    print("Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    train_sev4_boost(args.config)
