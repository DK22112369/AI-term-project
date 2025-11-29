import argparse
import os
import sys
import torch
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.early_fusion_mlp import EarlyFusionMLP
from data.preprocess_us_accidents import load_preprocessors, transform_with_preprocessors, load_full_dataset, stratified_sample, clean_and_engineer_features

def parse_arguments():
    parser = argparse.ArgumentParser(description="Explain Model with SHAP")
    parser.add_argument("--model_path", type=str, required=True, help="Path to .pt model file")
    parser.add_argument("--preprocessor_path", type=str, required=True, help="Path to preprocessors .joblib")
    parser.add_argument("--data_path", type=str, default="data/raw/US_Accidents_March23.csv")
    parser.add_argument("--sample_size", type=int, default=100, help="Number of background samples for SHAP")
    return parser.parse_args()

def main():
    args = parse_arguments()
    device = "cpu" # SHAP is often easier/safer on CPU for smaller samples
    
    # 1. Load Data (Small sample for explanation)
    print("Loading data for explanation...")
    if not os.path.exists(args.data_path):
        # Fallback
        if os.path.exists("US_Accidents_small.csv"): args.data_path = "US_Accidents_small.csv"
        elif os.path.exists("US_Accidents_March23.csv"): args.data_path = "US_Accidents_March23.csv"
        
    df = load_full_dataset(args.data_path)
    df = stratified_sample(df, frac=0.01) # Very small sample
    df = clean_and_engineer_features(df)
    
    # 2. Load Preprocessors
    bundle = load_preprocessors(args.preprocessor_path)
    
    # 3. Transform
    X_d, X_e, X_t, y = transform_with_preprocessors(df, bundle, device=device)
    
    # Concatenate for EarlyFusionMLP (SHAP works best with single input tensor)
    # Note: For CrashSeverityNet (multi-input), SHAP DeepExplainer needs list of inputs.
    # Here we assume EarlyFusionMLP for simplicity as it's easier to explain.
    X = torch.cat([X_d, X_e, X_t], dim=1)
    
    # 4. Load Model
    # We need to infer dimensions from X
    input_dim = X.shape[1]
    num_classes = 4
    model = EarlyFusionMLP(input_dim, num_classes).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    # 5. SHAP Analysis
    print("Running SHAP analysis...")
    # Use a background dataset (e.g., 100 random samples)
    background = X[:args.sample_size]
    test_samples = X[args.sample_size:args.sample_size+10] # Explain next 10 samples
    
    explainer = shap.DeepExplainer(model.mlp, background)
    shap_values = explainer.shap_values(test_samples)
    
    # 6. Plot
    print("Saving SHAP summary plot...")
    # Feature names are hard to get perfectly from OHE, but we can try generic names or indices
    # For now, we just let SHAP use indices or generic names
    
    save_path = "results/shap_summary.png"
    os.makedirs("results", exist_ok=True)
    
    plt.figure()
    # shap_values is a list of arrays (one for each class). We plot for Class 1 (Severity 2 - Majority) or Class 3 (Severity 4)
    # Let's plot for Severity 4 (Index 3)
    shap.summary_plot(shap_values[3], test_samples.numpy(), show=False)
    plt.title("SHAP Summary for Severity 4")
    plt.savefig(save_path)
    print(f"SHAP summary saved to {save_path}")

if __name__ == "__main__":
    main()
