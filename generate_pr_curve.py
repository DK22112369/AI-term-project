import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.preprocess_us_accidents import load_full_dataset, stratified_sample, clean_and_engineer_features, fit_feature_transformers, transform_with_preprocessors, time_based_split, load_preprocessors
from models.crash_severity_net import CrashSeverityNet
from torch.utils.data import DataLoader, TensorDataset

def generate_pr_curve():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load Data & Preprocessors
    print("Loading data...")
    if os.path.exists("US_Accidents_small.csv"):
        data_path = "US_Accidents_small.csv"
    elif os.path.exists("data/raw/US_Accidents_March23.csv"):
        data_path = "data/raw/US_Accidents_March23.csv"
    else:
        print("Data not found")
        return

    df = load_full_dataset(data_path)
    
    # If using small dataset, use all of it. If large, sample.
    if "small" in data_path:
        df = df # Use full small dataset
    else:
        df = stratified_sample(df, frac=0.1)
        
    df = clean_and_engineer_features(df)
    
    # Load Preprocessors (CRITICAL: Must match training)
    preprocessor_path = "models/crash_severity_net_focal_time_preprocessors.joblib"
    print(f"Loading preprocessors from {preprocessor_path}...")
    if not os.path.exists(preprocessor_path):
        print(f"Error: Preprocessor file not found at {preprocessor_path}")
        return
    bundle = load_preprocessors(preprocessor_path)
    
    # We need the test set. 
    _, _, df_test = time_based_split(df)
    
    print("Transforming test set...")
    X_t, X_w, X_r, X_s, y = transform_with_preprocessors(df_test, bundle, device)
    
    # Check for NaNs in inputs
    if torch.isnan(X_t).any() or torch.isnan(X_w).any() or torch.isnan(X_r).any() or torch.isnan(X_s).any():
        print("Warning: Input tensors contain NaNs! Replacing with 0.")
        X_t = torch.nan_to_num(X_t)
        X_w = torch.nan_to_num(X_w)
        X_r = torch.nan_to_num(X_r)
        X_s = torch.nan_to_num(X_s)

    # 2. Load Model
    print("Loading model...")
    num_classes = 4
    input_dims = {
        'temporal': X_t.shape[1],
        'weather': X_w.shape[1],
        'road': X_r.shape[1],
        'spatial': X_s.shape[1]
    }
    model = CrashSeverityNet(input_dims, num_classes).to(device)
    model.load_state_dict(torch.load("models/crash_severity_net_focal_time.pt", map_location=device))
    model.eval()
    
    # 3. Inference
    print("Running inference...")
    dataset = TensorDataset(X_t, X_w, X_r, X_s, y)
    loader = DataLoader(dataset, batch_size=256)
    
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for xt, xw, xr, xs, target in loader:
            inputs = {'temporal': xt, 'weather': xw, 'road': xr, 'spatial': xs}
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            
    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)
    
    # Handle NaNs in predictions
    if np.isnan(all_probs).any():
        print("Warning: Predictions contain NaNs! Replacing with 0.")
        all_probs = np.nan_to_num(all_probs)
    
    # 4. Plot PR Curve for Class 4 (Index 3)
    print("Plotting PR Curve...")
    # Binarize targets for Class 4
    y_true_class4 = (all_targets == 3).astype(int)
    y_scores_class4 = all_probs[:, 3]
    
    precision, recall, _ = precision_recall_curve(y_true_class4, y_scores_class4)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR Curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve: Fatal Accident Detection (Severity 4)')
    plt.legend(loc="lower left")
    plt.grid(True)
    
    os.makedirs("thesis_materials/figures", exist_ok=True)
    save_path = "thesis_materials/figures/pr_curve_class4.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved {save_path}")
    plt.close()

if __name__ == "__main__":
    generate_pr_curve()
