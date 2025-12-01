import argparse
import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.preprocess_us_accidents import (
    load_full_dataset, 
    stratified_sample, 
    clean_and_engineer_features, 
    fit_feature_transformers,
    transform_with_preprocessors,
    time_based_split,
    load_preprocessors
)
from models.sev4_boost.crash_severity_boosted import CrashSeverityBoosted
from utils.metrics import evaluate_metrics, save_metrics

def eval_sev4_boost(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating Sev4 Boost Model on {device}")
    
    # 1. Load Data (Test Set)
    # Ideally we load the same split. 
    # For consistency, we repeat the split logic.
    data_path = "data/raw/US_Accidents_March23.csv"
    if not os.path.exists(data_path):
         data_path = "US_Accidents_March23.csv"
         
    df = load_full_dataset(data_path)
    df = stratified_sample(df, frac=0.1)
    df = clean_and_engineer_features(df)
    _, _, df_test = time_based_split(df)
    
    # Load Preprocessors
    bundle = load_preprocessors("models/sev4_boost/preprocessors.joblib")
    
    X_t_test, X_w_test, X_r_test, X_s_test, y_test = transform_with_preprocessors(df_test, bundle, device)
    
    test_dataset = TensorDataset(X_t_test, X_w_test, X_r_test, X_s_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    
    # 2. Load Model
    input_dims = {
        'temporal': X_t_test.shape[1],
        'weather': X_w_test.shape[1],
        'road': X_r_test.shape[1],
        'spatial': X_s_test.shape[1]
    }
    
    model = CrashSeverityBoosted(input_dims, num_classes=4).to(device)
    model.load_state_dict(torch.load("models/sev4_boost/best_model.pt", map_location=device))
    model.eval()
    
    # 3. Inference
    all_preds = []
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for xt, xw, xr, xs, y in test_loader:
            inputs = {'temporal': xt, 'weather': xw, 'road': xr, 'spatial': xs}
            out = model(inputs)
            probs = torch.softmax(out, dim=1)
            preds = torch.argmax(out, dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            
    # 4. Metrics
    metrics = evaluate_metrics(all_targets, all_preds, y_probs=all_probs)
    
    print("\n=== Evaluation Results ===")
    print(metrics['classification_report'])
    
    # Save
    os.makedirs("results/sev4_boost", exist_ok=True)
    save_metrics(metrics, config, "results/sev4_boost/eval_metrics.json")
    print("Metrics saved to results/sev4_boost/eval_metrics.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    eval_sev4_boost(args.config)
