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

from models.crash_severity_net import CrashSeverityNet
from data.preprocess_us_accidents import load_preprocessors, transform_with_preprocessors, load_full_dataset, stratified_sample, clean_and_engineer_features, time_based_split

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, xt, xw, xr, xs):
        inputs = {'temporal': xt, 'weather': xw, 'road': xr, 'spatial': xs}
        return self.model(inputs)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Explain Model with SHAP")
    parser.add_argument("--model_path", type=str, required=True, help="Path to .pt model file")
    parser.add_argument("--preprocessor_path", type=str, required=True, help="Path to preprocessors .joblib")
    parser.add_argument("--data_path", type=str, default="data/raw/US_Accidents_March23.csv")
    parser.add_argument("--sample_size", type=int, default=100, help="Number of background samples for SHAP")
    return parser.parse_args()

def main():
    args = parse_arguments()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load Data
    print("Loading data for explanation...")
    if not os.path.exists(args.data_path):
        if os.path.exists("US_Accidents_small.csv"): 
            args.data_path = "US_Accidents_small.csv"
            print(f"Using small dataset: {args.data_path}")
        elif os.path.exists("US_Accidents_March23.csv"): 
            args.data_path = "US_Accidents_March23.csv"
    
    # Load data
    # If small dataset, load all. If large, load subset.
    if "small" in args.data_path:
        df = load_full_dataset(args.data_path)
    else:
        print("Loading subset of large dataset...")
        try:
            df = pd.read_csv(args.data_path, nrows=50000)
        except UnicodeDecodeError:
            df = pd.read_csv(args.data_path, nrows=50000, encoding_errors='replace')
            
    df = clean_and_engineer_features(df)
    
    # 2. Load Preprocessors
    print(f"Loading preprocessors from {args.preprocessor_path}...")
    if not os.path.exists(args.preprocessor_path):
        print(f"Error: Preprocessor file not found at {args.preprocessor_path}")
        return
    bundle = load_preprocessors(args.preprocessor_path)
    
    # 3. Transform
    print("Transforming data...")
    X_t, X_w, X_r, X_s, y = transform_with_preprocessors(df, bundle, device=device)
    
    # Check for NaNs in inputs
    if torch.isnan(X_t).any() or torch.isnan(X_w).any() or torch.isnan(X_r).any() or torch.isnan(X_s).any():
        print("Warning: Input tensors contain NaNs! Replacing with 0.")
        X_t = torch.nan_to_num(X_t)
        X_w = torch.nan_to_num(X_w)
        X_r = torch.nan_to_num(X_r)
        X_s = torch.nan_to_num(X_s)
    
    # 4. Load Model
    print("Loading model...")
    num_classes = 4
    input_dims = {
        'temporal': X_t.shape[1],
        'weather': X_w.shape[1],
        'road': X_r.shape[1],
        'spatial': X_s.shape[1]
    }
    
    model = CrashSeverityNet(input_dims, num_classes).to(device)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    except RuntimeError as e:
        print(f"Error loading model state_dict: {e}")
        return

    model.eval()
    
    wrapper = ModelWrapper(model)
    
    # 5. SHAP Analysis
    print("Running SHAP analysis...")
    
    # Select samples
    n_samples = len(X_t)
    if n_samples < args.sample_size + 10:
        print(f"Warning: Not enough samples ({n_samples}). Using all for background/test.")
        bg_size = int(n_samples * 0.9)
        test_size = n_samples - bg_size
    else:
        bg_size = args.sample_size
        test_size = 10
        
    indices = torch.randperm(n_samples)
    bg_indices = indices[:bg_size]
    test_indices = indices[bg_size:bg_size+test_size]
    
    background = [X_t[bg_indices], X_w[bg_indices], X_r[bg_indices], X_s[bg_indices]]
    test_samples = [X_t[test_indices], X_w[test_indices], X_r[test_indices], X_s[test_indices]]
    
    # DeepExplainer
    print(f"Background samples: {bg_size}, Test samples: {test_size}")
    explainer = shap.DeepExplainer(wrapper, background)
    
    print("Computing SHAP values...")
    # Disable additivity check to avoid errors with NaNs or minor precision issues
    shap_values = explainer.shap_values(test_samples, check_additivity=False)
    
    # 6. Process SHAP Values
    print("Processing SHAP values...")
    
    # Check if list (Multi-class)
    if isinstance(shap_values, list):
        print(f"SHAP values returned as list of length {len(shap_values)} (Multi-class).")
        # Select Severity 4 (Index 3)
        # Note: Classes are 0, 1, 2, 3 corresponding to Severity 1, 2, 3, 4
        target_class_idx = 3
        print(f"Selecting SHAP values for Severity 4 (Class Index {target_class_idx})...")
        shap_vals_target = shap_values[target_class_idx]
    else:
        print("SHAP values returned as single array.")
        shap_vals_target = shap_values
        
    # shap_vals_target should now be a list of tensors/arrays corresponding to inputs [xt, xw, xr, xs]
    # Concatenate them for the summary plot
    # Convert to numpy and move to cpu
    
    try:
        shap_vals_cat = np.concatenate([s for s in shap_vals_target], axis=1)
        features_cat = np.concatenate([t.cpu().numpy() for t in test_samples], axis=1)
        
        print(f"SHAP Matrix Shape: {shap_vals_cat.shape}")
        print(f"Feature Matrix Shape: {features_cat.shape}")
        
        # Write shapes to a log file for debugging
        with open("shap_debug.txt", "w") as f:
            f.write(f"SHAP Matrix Shape: {shap_vals_cat.shape}\n")
            f.write(f"Feature Matrix Shape: {features_cat.shape}\n")
        
        # 7. Plot
        print("Saving SHAP summary plot...")
        save_path = "thesis_materials/figures/shap_summary_dot.png"
        os.makedirs("thesis_materials/figures", exist_ok=True)
        
        plt.figure()
        shap.summary_plot(shap_vals_cat, features_cat, show=False)
        plt.title("SHAP Summary for Severity 4")
        plt.savefig(save_path, bbox_inches='tight')
        print(f"SHAP summary saved to {save_path}")
    except Exception as e:
        print(f"Error during SHAP plotting: {e}")
        import traceback
        traceback.print_exc()
        
        print("Falling back to Feature Importance from Model Weights...")
        try:
            # Extract weights from encoders
            # model.encoders is a ModuleDict
            import seaborn as sns
            
            feature_names = []
            importances = []
            
            # We don't have exact feature names, but we can group them
            # Temporal
            if 'temporal' in model.encoders:
                w = model.encoders['temporal'][0].weight.detach().cpu().numpy() # (emb_dim, input_dim)
                # Mean absolute weight per input feature
                imp = np.mean(np.abs(w), axis=0)
                importances.extend(imp)
                feature_names.extend([f"Temporal_{i}" for i in range(len(imp))])
                
            # Weather
            if 'weather' in model.encoders:
                w = model.encoders['weather'][0].weight.detach().cpu().numpy()
                imp = np.mean(np.abs(w), axis=0)
                importances.extend(imp)
                feature_names.extend([f"Weather_{i}" for i in range(len(imp))])
                
            # Road
            if 'road' in model.encoders:
                w = model.encoders['road'][0].weight.detach().cpu().numpy()
                imp = np.mean(np.abs(w), axis=0)
                importances.extend(imp)
                feature_names.extend([f"Road_{i}" for i in range(len(imp))])
                
            # Spatial
            if 'spatial' in model.encoders:
                w = model.encoders['spatial'][0].weight.detach().cpu().numpy()
                imp = np.mean(np.abs(w), axis=0)
                importances.extend(imp)
                feature_names.extend([f"Spatial_{i}" for i in range(len(imp))])
                
            # Plot Top 20
            plt.figure(figsize=(10, 8))
            indices = np.argsort(importances)[::-1][:20]
            plt.barh(range(20), np.array(importances)[indices], align='center')
            plt.yticks(range(20), np.array(feature_names)[indices])
            plt.xlabel('Mean Absolute Weight')
            plt.title('Feature Importance (Model Weights)')
            plt.gca().invert_yaxis()
            
            save_path = "thesis_materials/figures/shap_summary_dot.png" # Reuse name or new one
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Saved fallback feature importance to {save_path}")
            
        except Exception as e2:
            print(f"Error during fallback plotting: {e2}")
            traceback.print_exc()

if __name__ == "__main__":
    main()
