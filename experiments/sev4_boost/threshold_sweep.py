import argparse
import os
import sys
import json
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.preprocess_us_accidents import (
    load_full_dataset, 
    stratified_sample, 
    clean_and_engineer_features, 
    transform_with_preprocessors,
    time_based_split,
    load_preprocessors
)
from models.sev4_boost.crash_severity_boosted import CrashSeverityBoosted
from utils.metrics import evaluate_metrics

def run_threshold_sweep(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    thresholds = config.get("threshold_list", [0.1, 0.2, 0.3, 0.4, 0.5])
    print(f"Running Threshold Sweep on {device} for thresholds: {thresholds}")
    
    # 1. Load Data & Model (Similar to Eval)
    data_path = "data/raw/US_Accidents_March23.csv"
    if not os.path.exists(data_path):
         data_path = "US_Accidents_March23.csv"
         
    df = load_full_dataset(data_path)
    df = stratified_sample(df, frac=0.1)
    df = clean_and_engineer_features(df)
    _, _, df_test = time_based_split(df)
    
    bundle = load_preprocessors("models/sev4_boost/preprocessors.joblib")
    X_t_test, X_w_test, X_r_test, X_s_test, y_test = transform_with_preprocessors(df_test, bundle, device)
    
    test_dataset = TensorDataset(X_t_test, X_w_test, X_r_test, X_s_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    
    input_dims = {
        'temporal': X_t_test.shape[1],
        'weather': X_w_test.shape[1],
        'road': X_r_test.shape[1],
        'spatial': X_s_test.shape[1]
    }
    
    model = CrashSeverityBoosted(input_dims, num_classes=4).to(device)
    model.load_state_dict(torch.load("models/sev4_boost/best_model.pt", map_location=device))
    model.eval()
    
    # 2. Get Probabilities
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for xt, xw, xr, xs, y in test_loader:
            inputs = {'temporal': xt, 'weather': xw, 'road': xr, 'spatial': xs}
            out = model(inputs)
            probs = torch.softmax(out, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    
    # 3. Sweep
    results = []
    
    for t in thresholds:
        # Custom Threshold Logic for Class 3 (Severity 4)
        # If Prob(Sev4) >= t, predict Sev4. Else argmax of others.
        
        preds = []
        for i in range(len(all_probs)):
            p_sev4 = all_probs[i, 3]
            if p_sev4 >= t:
                preds.append(3)
            else:
                # Argmax of remaining classes (0, 1, 2)
                # We mask index 3 to be safe
                p_others = all_probs[i].copy()
                p_others[3] = -1
                preds.append(np.argmax(p_others))
                
        metrics = evaluate_metrics(all_targets, preds)
        
        # Extract Key Metrics
        sev4_metrics = metrics['classification_report'].get('3', {})
        res = {
            "threshold": t,
            "accuracy": metrics['accuracy'],
            "macro_f1": metrics['macro_f1'],
            "sev4_recall": sev4_metrics.get('recall', 0),
            "sev4_precision": sev4_metrics.get('precision', 0),
            "sev4_f1": sev4_metrics.get('f1-score', 0)
        }
        results.append(res)
        print(f"T={t:.2f} | Acc={res['accuracy']:.4f} | Sev4 Recall={res['sev4_recall']:.4f}")
        
    # 4. Save Results
    df_res = pd.DataFrame(results)
    os.makedirs("results/sev4_boost", exist_ok=True)
    df_res.to_csv("results/sev4_boost/threshold_results.csv", index=False)
    print("Saved results/sev4_boost/threshold_results.csv")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df_res['threshold'], df_res['sev4_recall'], marker='o', label='Sev4 Recall')
    plt.plot(df_res['threshold'], df_res['sev4_precision'], marker='s', label='Sev4 Precision')
    plt.plot(df_res['threshold'], df_res['accuracy'], marker='^', linestyle='--', label='Accuracy')
    plt.xlabel("Threshold (Severity 4)")
    plt.ylabel("Score")
    plt.title("Threshold Sweep: Impact on Fail-Safe Performance")
    plt.grid(True)
    plt.legend()
    plt.savefig("results/sev4_boost/threshold_curve.png")
    print("Saved results/sev4_boost/threshold_curve.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    run_threshold_sweep(args.config)
