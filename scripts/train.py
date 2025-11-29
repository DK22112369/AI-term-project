import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path to allow importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import joblib

from data.preprocess_us_accidents import (
    load_full_dataset, 
    stratified_sample, 
    clean_and_engineer_features, 
    fit_feature_transformers,
    transform_with_preprocessors,
    time_based_split,
    save_preprocessors,
    apply_smote_nc
)
from models.crash_severity_net import CrashSeverityNet
from models.early_fusion_mlp import EarlyFusionMLP
from models.tab_transformer import TabTransformer
from models.losses import FocalLoss
from utils.metrics import evaluate_metrics, save_metrics
from visualization.plots import plot_confusion_matrix, plot_loss_curve
from utils.common import set_seed

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train CrashSeverityNet or Baseline on US Accidents Dataset")
    parser.add_argument("--data_path", type=str, default="data/raw/US_Accidents_March23.csv")
    parser.add_argument("--sample_frac", type=float, default=0.1, help="Fraction of data to sample")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--split_strategy", type=str, default="random",
                        choices=["random", "time"],
                        help="Data splitting strategy: random (stratified) or time (chronological)")
    parser.add_argument("--model_type", type=str, default="crash_severity_net",
                        choices=["crash_severity_net", "early_mlp", "tab_transformer"],
                        help="Model architecture")
    parser.add_argument("--loss_type", type=str, default="ce",
                        choices=["ce", "ce_weighted", "focal"],
                        help="Loss function type")
    parser.add_argument("--use_sampler", action="store_true", help="Use WeightedRandomSampler")
    parser.add_argument("--use_smote", action="store_true", help="Use SMOTE-NC (Only for early_mlp/tab_transformer)")
    parser.add_argument("--gamma", type=float, default=2.0, help="Gamma for Focal Loss")
    return parser.parse_args()

def get_class_weights(y, device):
    """Calculates inverse class frequencies as weights."""
    classes = np.unique(y)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    return torch.FloatTensor(weights).to(device)

def get_criterion(loss_type, class_weights, gamma=2.0):
    if loss_type == "ce":
        return nn.CrossEntropyLoss()
    elif loss_type == "ce_weighted":
        return nn.CrossEntropyLoss(weight=class_weights)
    elif loss_type == "focal":
        return FocalLoss(alpha=class_weights, gamma=gamma)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

def main():
    args = parse_arguments()
    set_seed(args.seed)
    print(f"Starting experiment: {args.model_type}, Loss: {args.loss_type}, Split: {args.split_strategy}")
    
    # 1. Load Data
    if not os.path.exists(args.data_path):
        # Fallback for testing
        if os.path.exists("US_Accidents_small.csv"):
            args.data_path = "US_Accidents_small.csv"
        elif os.path.exists("US_Accidents_March23.csv"):
             args.data_path = "US_Accidents_March23.csv"
        else:
            print(f"Error: Dataset not found at {args.data_path}")
            return

    df = load_full_dataset(args.data_path)
    df = stratified_sample(df, frac=args.sample_frac)
    df = clean_and_engineer_features(df)
    
    # 2. Split Data
    if args.split_strategy == "random":
        print("Splitting data (Stratified Random)...")
        from sklearn.model_selection import train_test_split
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=args.seed, stratify=df["Severity"])
        df_train, df_val = train_test_split(df_train, test_size=0.125, random_state=args.seed, stratify=df_train["Severity"]) # 0.125 * 0.8 = 0.1
    else:
        print("Splitting data (Time-based)...")
        df_train, df_val, df_test = time_based_split(df)
    
    # 3. Fit Transformers
    bundle = fit_feature_transformers(df_train)
    
    # Save Preprocessors
    experiment_name = f"{args.model_type}_{args.loss_type}"
    if args.use_sampler: experiment_name += "_sampler"
    if args.use_smote: experiment_name += "_smote"
    if args.split_strategy == "time": experiment_name += "_time"
    save_preprocessors(bundle, f"models/{experiment_name}_preprocessors.joblib")
    
    print("Transforming datasets...")
    # Returns: Temporal, Weather, Road, Spatial, Targets
    X_t_train, X_w_train, X_r_train, X_s_train, y_train = transform_with_preprocessors(df_train, bundle, args.device)
    X_t_val, X_w_val, X_r_val, X_s_val, y_val = transform_with_preprocessors(df_val, bundle, args.device)
    X_t_test, X_w_test, X_r_test, X_s_test, y_test = transform_with_preprocessors(df_test, bundle, args.device)
    
    # 4. SMOTE-NC Handling
    if args.use_smote:
        if args.model_type == "crash_severity_net":
            print("Warning: SMOTE-NC is not recommended for CrashSeverityNet (Late Fusion). Ignoring.")
        else:
            # Concatenate all features
            X_train_cat = torch.cat([X_t_train, X_w_train, X_r_train, X_s_train], dim=1).cpu().numpy()
            y_train_np = y_train.cpu().numpy()
            
            X_res, y_res = apply_smote_nc(X_train_cat, y_train_np, cat_indices=[])
            
            # Update tensors
            X_train_cat_t = torch.FloatTensor(X_res).to(args.device)
            y_train = torch.LongTensor(y_res).to(args.device)
            
            # Split back
            t_dim = X_t_train.shape[1]
            w_dim = X_w_train.shape[1]
            r_dim = X_r_train.shape[1]
            s_dim = X_s_train.shape[1]
            
            X_t_train = X_train_cat_t[:, :t_dim]
            X_w_train = X_train_cat_t[:, t_dim:t_dim+w_dim]
            X_r_train = X_train_cat_t[:, t_dim+w_dim:t_dim+w_dim+r_dim]
            X_s_train = X_train_cat_t[:, t_dim+w_dim+r_dim:]
            
            print(f"SMOTE applied. Train size: {len(y_train)}")

    # 5. Dataset & DataLoader
    train_dataset = TensorDataset(X_t_train, X_w_train, X_r_train, X_s_train, y_train)
    val_dataset = TensorDataset(X_t_val, X_w_val, X_r_val, X_s_val, y_val)
    test_dataset = TensorDataset(X_t_test, X_w_test, X_r_test, X_s_test, y_test)
    
    sampler = None
    if args.use_sampler and not args.use_smote:
        print("Using WeightedRandomSampler...")
        class_counts = np.bincount(y_train.cpu().numpy())
        class_weights_sampler = 1. / class_counts
        sample_weights = class_weights_sampler[y_train.cpu().numpy()]
        g = torch.Generator()
        g.manual_seed(args.seed)
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True, generator=g)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # 6. Model Initialization
    num_classes = 4
    input_dims = {
        'temporal': X_t_train.shape[1],
        'weather': X_w_train.shape[1],
        'road': X_r_train.shape[1],
        'spatial': X_s_train.shape[1]
    }
    
    if args.model_type == "crash_severity_net":
        model = CrashSeverityNet(input_dims, num_classes).to(args.device)
    elif args.model_type == "early_mlp":
        total_dim = sum(input_dims.values())
        model = EarlyFusionMLP(total_dim, num_classes).to(args.device)
    elif args.model_type == "tab_transformer":
        # Order matters for TabTransformer list input
        model = TabTransformer(
            cat_cardinalities=[input_dims['temporal'], input_dims['weather'], input_dims['road'], input_dims['spatial']],
            num_continuous=0,
            num_classes=num_classes,
            use_ohe_input=True
        ).to(args.device)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    print(model)

    # 7. Loss & Optimizer
    class_weights = get_class_weights(y_train.cpu().numpy(), args.device)
    criterion = get_criterion(args.loss_type, class_weights, args.gamma)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 8. Training Loop
    history = []
    best_val_loss = float('inf')
    
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for xt, xw, xr, xs, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
            xt, xw, xr, xs, y = xt.to(args.device), xw.to(args.device), xr.to(args.device), xs.to(args.device), y.to(args.device)
            
            optimizer.zero_grad()
            
            if args.model_type == "crash_severity_net":
                inputs = {'temporal': xt, 'weather': xw, 'road': xr, 'spatial': xs}
                out = model(inputs)
            elif args.model_type == "tab_transformer":
                out = model([xt, xw, xr, xs], torch.empty(xt.size(0), 0).to(args.device))
            elif args.model_type == "early_mlp":
                # Concatenate for EarlyMLP
                concat_input = torch.cat([xt, xw, xr, xs], dim=1)
                out = model(concat_input)
            else:
                # Should not happen
                out = model(xt, xw, xr, xs)
                
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for xt, xw, xr, xs, y in val_loader:
                xt, xw, xr, xs, y = xt.to(args.device), xw.to(args.device), xr.to(args.device), xs.to(args.device), y.to(args.device)
                
                if args.model_type == "tab_transformer":
                    out = model([xt, xw, xr, xs], torch.empty(xt.size(0), 0).to(args.device))
                elif args.model_type == "early_mlp":
                    concat_input = torch.cat([xt, xw, xr, xs], dim=1)
                    out = model(concat_input)
                elif args.model_type == "crash_severity_net":
                    inputs = {'temporal': xt, 'weather': xw, 'road': xr, 'spatial': xs}
                    out = model(inputs)
                else:
                    out = model(xt, xw, xr, xs)

                loss = criterion(out, y)
                val_loss += loss.item()
                preds = torch.argmax(out, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_metrics = evaluate_metrics(all_targets, all_preds)
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={val_metrics['accuracy']:.4f}, Macro F1={val_metrics['macro_f1']:.4f}")
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_acc': val_metrics['accuracy'],
            'val_macro_f1': val_metrics['macro_f1']
        })
        
        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"models/{experiment_name}.pt")
            print(f"  Saved best model (Val Loss: {best_val_loss:.4f})")

    # 9. Test Evaluation
    print("Evaluating on Test Set...")
    model.load_state_dict(torch.load(f"models/{experiment_name}.pt"))
    model.eval()
    test_preds = []
    test_targets = []
    with torch.no_grad():
        for xt, xw, xr, xs, y in test_loader:
            xt, xw, xr, xs, y = xt.to(args.device), xw.to(args.device), xr.to(args.device), xs.to(args.device), y.to(args.device)
            
            if args.model_type == "tab_transformer":
                out = model([xt, xw, xr, xs], torch.empty(xt.size(0), 0).to(args.device))
            elif args.model_type == "early_mlp":
                concat_input = torch.cat([xt, xw, xr, xs], dim=1)
                out = model(concat_input)
            elif args.model_type == "crash_severity_net":
                inputs = {'temporal': xt, 'weather': xw, 'road': xr, 'spatial': xs}
                out = model(inputs)
            else:
                out = model(xt, xw, xr, xs)
                
            preds = torch.argmax(out, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(y.cpu().numpy())
            
    test_metrics = evaluate_metrics(test_targets, test_preds)
    print("\nTest Set Results:")
    print(test_metrics['classification_report'])
    
    # 10. Save Results
    config_path = f"results/{experiment_name}_config.json"
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
        
    save_metrics(test_metrics, vars(args), f"results/{experiment_name}.json")
    
    plot_confusion_matrix(test_metrics['confusion_matrix'], classes=["1", "2", "3", "4"],
                          title=f"Confusion Matrix ({args.model_type})",
                          save_path=f"results/confmat_{experiment_name}.png")
                          
    plot_loss_curve(history, save_path=f"results/loss_curve_{experiment_name}.png")
    print("Experiment complete.")

if __name__ == "__main__":
    main()
