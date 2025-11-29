import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os
import sys
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_recall_curve, auc

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.crash_severity_net import CrashSeverityNet
from data.preprocess_us_accidents import load_full_dataset, clean_and_engineer_features, transform_with_preprocessors, load_preprocessors

def set_academic_style():
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 14

def plot_recall_comparison():
    # Data from results_summary.md
    data = {
        'Model': ['CatBoost (Baseline)', 'CatBoost (Baseline)', 
                  'CrashSeverityNet (Ours)', 'CrashSeverityNet (Ours)'],
        'Metric': ['Accuracy', 'Severity 4 Recall', 
                   'Accuracy', 'Severity 4 Recall'],
        'Value': [0.806, 0.030, 
                  0.592, 0.686]
    }
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(10, 6))
    
    # Custom palette: Grey for Baseline, Red for Ours (Safety)
    colors = ["#95a5a6", "#e74c3c"] 
    
    ax = sns.barplot(x='Metric', y='Value', hue='Model', data=df, palette=colors)
    
    # Add values
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 0.01,
                f'{height:.1%}', ha="center", fontsize=12, weight='bold')
                
    plt.title('Performance Comparison: Baseline vs. Proposed Model')
    plt.ylim(0, 1.0)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.tight_layout()
    plt.savefig("final_thesis_data/figures/fig1_recall_comparison_bar.png", dpi=300)
    print("Saved fig1_recall_comparison_bar.png")
    plt.close()

def plot_confusion_matrices():
    # Load data
    with open("results/baseline_catboost.json", 'r') as f:
        res = json.load(f)
        if 'metrics' in res:
            cm_base = np.array(res['metrics']['confusion_matrix'])
        else:
            cm_base = np.array(res['confusion_matrix'])
    
    # For Ours, we use the best model's CM
    with open("results/crash_severity_net_ce_weighted.json", 'r') as f:
        res = json.load(f)
        if 'metrics' in res:
            cm_ours = np.array(res['metrics']['confusion_matrix'])
        else:
            cm_ours = np.array(res['confusion_matrix'])
            
    # Normalize
    cm_base_norm = cm_base.astype('float') / cm_base.sum(axis=1)[:, np.newaxis]
    cm_ours_norm = cm_ours.astype('float') / cm_ours.sum(axis=1)[:, np.newaxis]
    
    classes = ['Sev 1', 'Sev 2', 'Sev 3', 'Sev 4']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    sns.heatmap(cm_base_norm, annot=True, fmt='.2f', cmap='Blues', ax=axes[0],
                xticklabels=classes, yticklabels=classes, cbar=False)
    axes[0].set_title('Baseline (CatBoost)')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    sns.heatmap(cm_ours_norm, annot=True, fmt='.2f', cmap='Reds', ax=axes[1],
                xticklabels=classes, yticklabels=classes, cbar=False)
    axes[1].set_title('Proposed (CrashSeverityNet)')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig("final_thesis_data/figures/fig2_confusion_matrices.png", dpi=300)
    print("Saved fig2_confusion_matrices.png")
    plt.close()

def plot_pr_curve_threshold():
    # Use retrained model for demonstration of curve and threshold
    model_path = "models/crash_severity_net_ce_weighted_time.pt"
    prep_path = "models/crash_severity_net_ce_weighted_time_preprocessors.joblib"
    
    if not os.path.exists(model_path):
        print("Retrained model not found, skipping PR curve.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Data (Small)
    df = load_full_dataset("US_Accidents_small.csv")
    df = clean_and_engineer_features(df)
    bundle = load_preprocessors(prep_path)
    X_t, X_w, X_r, X_s, y = transform_with_preprocessors(df, bundle, device=device)
    
    # Handle NaNs
    if torch.isnan(X_t).any() or torch.isnan(X_w).any() or torch.isnan(X_r).any() or torch.isnan(X_s).any():
        print("Warning: Input tensors contain NaNs! Replacing with 0.")
        X_t = torch.nan_to_num(X_t)
        X_w = torch.nan_to_num(X_w)
        X_r = torch.nan_to_num(X_r)
        X_s = torch.nan_to_num(X_s)
    
    # Load Model
    state_dict = torch.load(model_path, map_location=device)
    # Infer dims from state_dict
    input_dims = {}
    if "encoders.temporal.0.weight" in state_dict:
        input_dims['temporal'] = state_dict["encoders.temporal.0.weight"].shape[1]
    elif "encoders.temporal.net.0.weight" in state_dict:
        input_dims['temporal'] = state_dict["encoders.temporal.net.0.weight"].shape[1]
        
    if "encoders.weather.0.weight" in state_dict:
        input_dims['weather'] = state_dict["encoders.weather.0.weight"].shape[1]
    elif "encoders.weather.net.0.weight" in state_dict:
        input_dims['weather'] = state_dict["encoders.weather.net.0.weight"].shape[1]
        
    if "encoders.road.0.weight" in state_dict:
        input_dims['road'] = state_dict["encoders.road.0.weight"].shape[1]
    elif "encoders.road.net.0.weight" in state_dict:
        input_dims['road'] = state_dict["encoders.road.net.0.weight"].shape[1]
        
    if "encoders.spatial.0.weight" in state_dict:
        input_dims['spatial'] = state_dict["encoders.spatial.0.weight"].shape[1]
    elif "encoders.spatial.net.0.weight" in state_dict:
        input_dims['spatial'] = state_dict["encoders.spatial.net.0.weight"].shape[1]
    model = CrashSeverityNet(input_dims, 4).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Inference
    dataset = TensorDataset(X_t, X_w, X_r, X_s, y)
    loader = DataLoader(dataset, batch_size=256)
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for xt, xw, xr, xs, target in loader:
            inputs = {'temporal': xt, 'weather': xw, 'road': xr, 'spatial': xs}
            logits = model(inputs)
            all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            all_targets.append(target.cpu().numpy())
            
    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)
    
    # PR Curve Class 4
    y_true = (all_targets == 3).astype(int)
    y_scores = all_probs[:, 3]
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkred', lw=2, label=f'Class 4 (AUC = {pr_auc:.2f})')
    
    # Mark T=0.29
    # Find index closest to T=0.29
    # thresholds array is shorter than p/r by 1
    idx = np.argmin(np.abs(thresholds - 0.29))
    plt.plot(recall[idx], precision[idx], 'r*', markersize=15, label='Optimized T=0.29')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Severity 4)')
    plt.legend()
    plt.grid(True)
    plt.savefig("final_thesis_data/figures/fig3_pr_curve_threshold.png", dpi=300)
    print("Saved fig3_pr_curve_threshold.png")
    plt.close()

def main():
    set_academic_style()
    plot_recall_comparison()
    plot_confusion_matrices()
    plot_pr_curve_threshold()

if __name__ == "__main__":
    main()
