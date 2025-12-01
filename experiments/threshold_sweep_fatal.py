import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import joblib
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score, confusion_matrix

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.crash_severity_net import CrashSeverityNet, infer_input_dims
from data.preprocess_us_accidents import transform_with_preprocessors, load_full_dataset, clean_and_engineer_features, time_based_split

def parse_arguments():
    parser = argparse.ArgumentParser(description="Threshold Sweep for Fatal Accident Detection")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--preprocessor_path", type=str, required=True, help="Path to preprocessor bundle (.joblib)")
    parser.add_argument("--data_path", type=str, default="US_Accidents_small.csv")
    parser.add_argument("--output_dir", type=str, default="results/threshold_sweep")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def get_predictions(model, df, bundle, device):
    """
    Get probabilities for the dataset using the model.
    """
    model.eval()
    
    # Transform data
    # Note: For large datasets, we should use a DataLoader. 
    # Here we assume the test set fits in memory for simplicity of the sweep script.
    print("Transforming data...")
    X_t, X_w, X_r, X_s, y = transform_with_preprocessors(df, bundle, device=device)
    
    inputs = {'temporal': X_t, 'weather': X_w, 'road': X_r, 'spatial': X_s}
    
    print("Running inference...")
    with torch.no_grad():
        logits = model(inputs)
        probs = torch.softmax(logits, dim=1)
        
    return probs.cpu().numpy(), y.cpu().numpy()

def sweep_thresholds(probs, targets, class_idx=3):
    """
    Sweep thresholds for a specific class (default 3 for Severity 4 - 0-indexed).
    """
    thresholds = np.arange(0.0, 1.01, 0.05)
    results = []
    
    print(f"Sweeping thresholds for Class {class_idx} (Severity 4)...")
    
    for t in thresholds:
        # Decision Rule:
        # If P(Class 4) > t, predict 4.
        # Else, predict argmax of other classes (or just argmax of all if t is low, but strictly:
        # standard approach is: if p[4] > t then 4 else argmax(p[0..3]))
        
        # Vectorized implementation
        preds = np.argmax(probs, axis=1) # Default argmax
        
        # Apply threshold override
        # Indices where prob of class 4 > t
        override_indices = probs[:, class_idx] > t
        preds[override_indices] = class_idx
        
        # Metrics
        acc = accuracy_score(targets, preds)
        
        # Binary metrics for Class 4 vs Rest
        binary_targets = (targets == class_idx).astype(int)
        binary_preds = (preds == class_idx).astype(int)
        
        recall = recall_score(binary_targets, binary_preds, zero_division=0)
        precision = precision_score(binary_targets, binary_preds, zero_division=0)
        f2 = fbeta_score(binary_targets, binary_preds, beta=2, zero_division=0)
        
        results.append({
            "threshold": t,
            "accuracy": acc,
            "fatal_recall": recall,
            "fatal_precision": precision,
            "fatal_f2": f2
        })
        
    return pd.DataFrame(results)

def plot_results(df_results, output_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(df_results["threshold"], df_results["fatal_recall"], label="Fatal Recall", marker='o')
    plt.plot(df_results["threshold"], df_results["accuracy"], label="Overall Accuracy", marker='s')
    plt.plot(df_results["threshold"], df_results["fatal_f2"], label="Fatal F2-Score", marker='^')
    
    plt.xlabel("Threshold (Probability of Fatal Class)")
    plt.ylabel("Score")
    plt.title("Threshold Sweep: Accuracy vs Fatal Recall Trade-off")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/threshold_tradeoff.png")
    print(f"Plot saved to {output_dir}/threshold_tradeoff.png")

def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Data
    print(f"Loading data from {args.data_path}...")
    df = load_full_dataset(args.data_path)
    df = clean_and_engineer_features(df)
    # Use only Test set (Time-based split)
    _, _, df_test = time_based_split(df)
    print(f"Test set size: {len(df_test)}")
    
    # 2. Load Preprocessors
    print(f"Loading preprocessors from {args.preprocessor_path}...")
    bundle = joblib.load(args.preprocessor_path)
    
    # 3. Load Model
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=args.device)
    input_dims = infer_input_dims(checkpoint)
    model = CrashSeverityNet(input_dims, num_classes=4).to(args.device)
    model.load_state_dict(checkpoint)
    
    # 4. Get Predictions
    probs, targets = get_predictions(model, df_test, bundle, args.device)
    
    # 5. Sweep
    df_results = sweep_thresholds(probs, targets, class_idx=3) # 0-indexed, so 3 is Severity 4
    
    # 6. Save Results
    csv_path = f"{args.output_dir}/threshold_sweep_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # 7. Plot
    plot_results(df_results, args.output_dir)

if __name__ == "__main__":
    main()
