import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def set_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("paper", font_scale=1.5)
    plt.rcParams['font.family'] = 'serif'

import json

def load_metrics():
    with open("results/baseline_catboost.json", 'r') as f:
        baseline = json.load(f)['metrics']
    with open("results/crash_severity_net_focal_time.json", 'r') as f:
        ours = json.load(f)['metrics']
    return baseline, ours

def plot_recall_comparison(baseline, ours):
    models = ['CatBoost (Baseline)', 'CrashSeverityNet (Ours)']
    
    # Get Recall for Class 4 (Index 3 in classification report if keys are strings "0","1" etc or "1","2"...)
    # Report keys are usually "1", "2", "3", "4"
    # But we subtracted 1 in preprocessing, so keys are "0", "1", "2", "3"
    b_recall = baseline['classification_report']['3']['recall']
    o_recall = ours['classification_report']['3']['recall']
    
    recall_scores = [b_recall, o_recall]
    colors = ['#bdc3c7', '#e74c3c'] # Grey for baseline, Red for ours

    plt.figure(figsize=(8, 6))
    bars = plt.bar(models, recall_scores, color=colors, width=0.6)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.1%}',
                 ha='center', va='bottom', fontsize=16, fontweight='bold')

    plt.ylim(0, 1.0)
    plt.ylabel('Recall (Sensitivity) for Severity 4', fontsize=14)
    plt.title('Recall Comparison: Detecting Fatal Accidents', fontsize=16, fontweight='bold', pad=20)
    
    # Add gap annotation
    gap = o_recall - b_recall
    plt.annotate('', xy=(1, o_recall), xytext=(1, b_recall),
                 arrowprops=dict(arrowstyle='<->', color='black'))
    plt.text(1.05, (o_recall + b_recall)/2, f'+{gap:.1%} Improvement', va='center', fontsize=12, fontweight='bold', color='#e74c3c')

    plt.tight_layout()
    os.makedirs('thesis_materials/figures', exist_ok=True)
    save_path = 'thesis_materials/figures/recall_comparison.png'
    plt.savefig(save_path, dpi=300)
    print(f"Saved {save_path}")
    plt.close()

def plot_confusion_matrices(baseline, ours):
    # Get Confusion Matrices
    b_cm = np.array(baseline['confusion_matrix'])
    o_cm = np.array(ours['confusion_matrix'])
    
    # Normalize by row (True labels)
    b_cm_norm = b_cm.astype('float') / b_cm.sum(axis=1)[:, np.newaxis]
    o_cm_norm = o_cm.astype('float') / o_cm.sum(axis=1)[:, np.newaxis]
    
    # Class 4 Row (Index 3)
    # [Missed (Sum of others), Detected (Class 4)]
    # Actually, let's just show the full row for Class 4? 
    # Or just Missed vs Detected as before.
    # Missed = 1 - Recall
    # Detected = Recall
    
    b_recall = b_cm_norm[3, 3]
    o_recall = o_cm_norm[3, 3]
    
    cb_row = np.array([[1 - b_recall, b_recall]])
    ours_row = np.array([[1 - o_recall, o_recall]])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # CatBoost
    sns.heatmap(cb_row, annot=True, fmt='.1%', cmap='Reds', cbar=False, ax=ax1,
                xticklabels=['Missed', 'Detected'], yticklabels=['Severity 4'])
    ax1.set_title('CatBoost (Baseline)', fontsize=14)
    ax1.set_xlabel('Prediction')
    
    # Ours
    sns.heatmap(ours_row, annot=True, fmt='.1%', cmap='Greens', cbar=False, ax=ax2,
                xticklabels=['Missed', 'Detected'], yticklabels=['Severity 4'])
    ax2.set_title('CrashSeverityNet (Ours)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Prediction')
    
    plt.suptitle('Confusion Analysis: Fatal Accident Detection', fontsize=16, y=1.05)
    plt.tight_layout()
    
    save_path = 'thesis_materials/figures/confusion_matrix.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved {save_path}")
    plt.close()

if __name__ == "__main__":
    print(f"Current Working Directory: {os.getcwd()}")
    set_style()
    baseline, ours = load_metrics()
    plot_recall_comparison(baseline, ours)
    plot_confusion_matrices(baseline, ours)
