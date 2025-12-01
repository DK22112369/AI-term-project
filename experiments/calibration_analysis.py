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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocess_us_accidents import (
    load_full_dataset, 
    clean_and_engineer_features, 
    transform_with_preprocessors,
    time_based_split,
    load_preprocessors
)
from models.crash_severity_net import CrashSeverityNet, infer_input_dims
from utils.metrics import brier_score, reliability_curve, expected_cost

def run_calibration_analysis(model_path, preprocessor_path, data_path, output_dir, device="cpu"):
    print(f"Running Calibration Analysis on {device}")
    print(f"Model: {model_path}")
    print(f"Data: {data_path} (FULL DATASET - NO SAMPLING)")
    
    # 1. Load Data (FULL DATASET)
    if not os.path.exists(data_path):
         # Fallback logic if full path not found, but user insisted on full data.
         # We will try the standard name in current dir if the provided path fails.
         if os.path.exists("US_Accidents_March23.csv"):
             data_path = "US_Accidents_March23.csv"
         else:
             print(f"Error: Data file not found at {data_path}")
             return

    df = load_full_dataset(data_path)
    # NO SAMPLING: df = stratified_sample(df, frac=0.1) <--- REMOVED
    df = clean_and_engineer_features(df)
    
    # Time Split (Official)
    _, _, df_test = time_based_split(df)
    print(f"Test Set Size: {len(df_test)}")
    
    # Load Preprocessors
    bundle = load_preprocessors(preprocessor_path)
    
    # Transform
    X_t_test, X_w_test, X_r_test, X_s_test, y_test = transform_with_preprocessors(df_test, bundle, device)
    
    test_dataset = TensorDataset(X_t_test, X_w_test, X_r_test, X_s_test, y_test)
    # Batch size can be large for inference
    test_loader = DataLoader(test_dataset, batch_size=4096, shuffle=False)
    
    # 2. Load Model
    checkpoint = torch.load(model_path, map_location=device)
    input_dims = infer_input_dims(checkpoint)
    num_classes = 4
    
    model = CrashSeverityNet(input_dims, num_classes).to(device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    # 3. Inference
    all_probs = []
    all_preds = []
    all_targets = []
    
    print("Running Inference...")
    with torch.no_grad():
        for xt, xw, xr, xs, y in test_loader:
            inputs = {'temporal': xt, 'weather': xw, 'road': xr, 'spatial': xs}
            out = model(inputs)
            probs = torch.softmax(out, dim=1)
            preds = torch.argmax(out, dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    
    # 4. Calibration Metrics (Severity 4 Focus)
    print("Calculating Calibration Metrics...")
    
    # Brier Score
    sev4_brier = brier_score(all_targets, all_probs, positive_class=3)
    
    # Reliability Curve
    prob_pred, prob_true = reliability_curve(all_targets, all_probs, n_bins=10, positive_class=3)
    
    # Expected Cost
    cost_matrix = {
        ("fatal", "FN"): 100.0,
        ("fatal", "FP"): 1.0,
        ("nonfatal", "misclass"): 0.1
    }
    exp_cost = expected_cost(all_targets, all_preds, cost_matrix)
    
    # 5. Save Results
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = {
        "sev4_brier_score": sev4_brier,
        "expected_cost": exp_cost,
        "cost_matrix": str(cost_matrix)
    }
    
    with open(os.path.join(output_dir, "calibration_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Save Curve Data
    df_curve = pd.DataFrame({
        "mean_predicted_probability": prob_pred,
        "fraction_of_positives": prob_true
    })
    df_curve.to_csv(os.path.join(output_dir, "reliability_curve.csv"), index=False)
    
    # Plot Reliability Diagram
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
    plt.plot(prob_pred, prob_true, "s-", label="Model (Sev 4)")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Reliability Diagram (Severity 4)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "reliability_plot.png"))
    
    print("\n=== Calibration Results ===")
    print(f"Severity 4 Brier Score: {sev4_brier:.4f}")
    print(f"Expected Cost: {exp_cost:.4f}")
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--preprocessor_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/raw/US_Accidents_March23.csv")
    parser.add_argument("--output_dir", type=str, default="results/calibration")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    run_calibration_analysis(
        args.model_path, 
        args.preprocessor_path, 
        args.data_path, 
        args.output_dir, 
        args.device
    )
