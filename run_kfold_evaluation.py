import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import torch
from torch.utils.data import TensorDataset, DataLoader

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.preprocess_us_accidents import (
    load_full_dataset, 
    stratified_sample, 
    clean_and_engineer_features, 
    fit_feature_transformers,
    transform_with_preprocessors
)
from utils.metrics import evaluate_metrics
from utils.common import set_seed

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run K-Fold Cross Validation")
    parser.add_argument("--data_path", type=str, default="data/raw/US_Accidents_March23.csv")
    parser.add_argument("--sample_frac", type=float, default=0.1)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--model_type", type=str, default="rf", choices=["rf", "xgb", "catboost", "lgbm"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def train_evaluate_fold(X_train, y_train, X_val, y_val, model_type, seed):
    """Trains and evaluates a model for a single fold."""
    
    # Initialize Model
    if model_type == "rf":
        model = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    elif model_type == "xgb":
        from xgboost import XGBClassifier
        model = XGBClassifier(n_estimators=100, random_state=seed, n_jobs=-1, eval_metric='mlogloss')
    elif model_type == "catboost":
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(iterations=100, random_seed=seed, verbose=0, allow_writing_files=False)
    elif model_type == "lgbm":
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(n_estimators=100, random_state=seed, n_jobs=-1, verbose=-1)
        
    # Train
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_val)
    metrics = evaluate_metrics(y_val, y_pred)
    return metrics

def main():
    args = parse_arguments()
    set_seed(args.seed)
    print(f"Running {args.folds}-Fold CV for {args.model_type.upper()}...")
    
    # 1. Load Data
    if not os.path.exists(args.data_path):
        # Fallback
        if os.path.exists("US_Accidents_small.csv"): args.data_path = "US_Accidents_small.csv"
        elif os.path.exists("US_Accidents_March23.csv"): args.data_path = "US_Accidents_March23.csv"
        
    df = load_full_dataset(args.data_path)
    df = stratified_sample(df, frac=args.sample_frac)
    df = clean_and_engineer_features(df)
    
    # 2. K-Fold Setup
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    
    fold_results = []
    
    X = df  # We will transform inside the loop to avoid leakage (fit on train only)
    y = df["Severity"].values
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold+1}/{args.folds}...")
        
        df_train = df.iloc[train_idx]
        df_val = df.iloc[val_idx]
        
        # Fit Transformers on Train
        bundle = fit_feature_transformers(df_train)
        
        # Transform
        X_t_train, X_w_train, X_r_train, X_s_train, y_train_t = transform_with_preprocessors(df_train, bundle, device='cpu')
        X_t_val, X_w_val, X_r_val, X_s_val, y_val_t = transform_with_preprocessors(df_val, bundle, device='cpu')
        
        # Concatenate for ML models
        X_train_np = np.concatenate([X_t_train.numpy(), X_w_train.numpy(), X_r_train.numpy(), X_s_train.numpy()], axis=1)
        X_val_np = np.concatenate([X_t_val.numpy(), X_w_val.numpy(), X_r_val.numpy(), X_s_val.numpy()], axis=1)
        y_train_np = y_train_t.numpy()
        y_val_np = y_val_t.numpy()
        
        # Train & Eval
        metrics = train_evaluate_fold(X_train_np, y_train_np, X_val_np, y_val_np, args.model_type, args.seed)
        
        print(f"  Acc: {metrics['accuracy']:.4f}, Macro F1: {metrics['macro_f1']:.4f}")
        fold_results.append(metrics)
        
    # 3. Aggregate Results
    accuracies = [m['accuracy'] for m in fold_results]
    macro_f1s = [m['macro_f1'] for m in fold_results]
    
    summary = {
        "model": args.model_type,
        "folds": args.folds,
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "macro_f1_mean": float(np.mean(macro_f1s)),
        "macro_f1_std": float(np.std(macro_f1s)),
        "fold_details": fold_results
    }
    
    print("\nK-Fold Summary:")
    print(f"Accuracy: {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}")
    print(f"Macro F1: {summary['macro_f1_mean']:.4f} ± {summary['macro_f1_std']:.4f}")
    
    # Save
    save_path = f"results/kfold_{args.model_type}.json"
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"Results saved to {save_path}")

if __name__ == "__main__":
    main()
