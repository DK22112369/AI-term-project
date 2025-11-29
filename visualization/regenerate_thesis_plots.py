import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os

def plot_confusion_matrix(cm, classes, title, filename):
    # Normalize
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Saved {filename}")
    plt.close()

def plot_performance_comparison():
    # Metrics requested by user
    # Baseline (CatBoost)
    # We can load from file or hardcode if we know them.
    # From results_summary.md: CatBoost Acc 80.6%, Sev 4 Recall 3.0%
    
    # Best Model (CrashSeverityNet)
    # User specified: Acc 59.2%, Recall 68.6%
    
    data = {
        'Model': ['CatBoost (Baseline)', 'CatBoost (Baseline)', 
                  'CrashSeverityNet (Ours)', 'CrashSeverityNet (Ours)'],
        'Metric': ['Accuracy', 'Severity 4 Recall', 
                   'Accuracy', 'Severity 4 Recall'],
        'Value': [0.806, 0.030, 
                  0.592, 0.686] # User's numbers
    }
    
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Custom palette
    colors = ["#95a5a6", "#3498db"] # Grey for Baseline, Blue for Ours
    
    ax = sns.barplot(x='Metric', y='Value', hue='Model', data=df, palette=colors)
    
    # Add values on top
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 0.01,
                f'{height:.1%}', ha="center", fontsize=12, weight='bold')
                
    plt.title('Performance Comparison: Baseline vs. Proposed Model', fontsize=14)
    plt.ylim(0, 1.0)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    plt.tight_layout()
    
    save_path = "thesis_materials/figures/recall_comparison.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved {save_path}")
    plt.close()

def main():
    os.makedirs("thesis_materials/figures", exist_ok=True)
    
    # 1. Plot Bar Chart (Performance Comparison)
    plot_performance_comparison()
    
    # 2. Plot Confusion Matrix
    # Load actual matrix from file
    json_path = "results/crash_severity_net_ce_weighted.json"
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            res = json.load(f)
            # Check if metrics key exists
            if 'metrics' in res:
                cm = np.array(res['metrics']['confusion_matrix'])
            else:
                # Fallback or direct access
                cm = np.array(res.get('confusion_matrix', []))
            
        classes = ['Sev 1', 'Sev 2', 'Sev 3', 'Sev 4']
        plot_confusion_matrix(cm, classes, 
                              "Confusion Matrix (CrashSeverityNet)", 
                              "thesis_materials/figures/confusion_matrix.png")
    else:
        print(f"Error: {json_path} not found.")

if __name__ == "__main__":
    main()
