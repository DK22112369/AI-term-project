import json
import os
import sys
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def generate_results_summary():
    # Load Data
    with open("results/baseline_catboost.json", 'r') as f:
        baseline = json.load(f)['metrics']
    with open("results/crash_severity_net_focal_time.json", 'r') as f:
        ours = json.load(f)['metrics']
        
    # Helper to get metrics for a class
    def get_class_metrics(report, cls):
        if cls in report:
            return report[cls]
        return {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0}

    # Markdown Content
    md_content = "# Thesis Results Summary\n\n"
    
    # 1. Overall Comparison
    md_content += "## 1. Overall Performance Comparison\n\n"
    md_content += "| Model | Accuracy | Macro F1 | Weighted F1 |\n"
    md_content += "| :--- | :--- | :--- | :--- |\n"
    md_content += f"| CatBoost (Baseline) | {baseline['accuracy']:.4f} | {baseline['macro_f1']:.4f} | {baseline['weighted_f1']:.4f} |\n"
    md_content += f"| CrashSeverityNet (Ours) | {ours['accuracy']:.4f} | {ours['macro_f1']:.4f} | {ours['weighted_f1']:.4f} |\n\n"
    
    # 2. Per-Class Comparison
    md_content += "## 2. Per-Class Performance (Precision / Recall / F1)\n\n"
    md_content += "| Class | Metric | CatBoost | CrashSeverityNet | Gap |\n"
    md_content += "| :--- | :--- | :--- | :--- | :--- |\n"
    
    classes = ["1", "2", "3", "4"]
    metrics = ["precision", "recall", "f1-score"]
    
    for cls in classes:
        # Map Severity 1..4 to Keys "0".."3"
        key = str(int(cls) - 1)
        b_metrics = get_class_metrics(baseline['classification_report'], key)
        o_metrics = get_class_metrics(ours['classification_report'], key)
        
        for m in metrics:
            gap = o_metrics[m] - b_metrics[m]
            gap_str = f"{gap:+.4f}"
            if m == "recall" and cls == "4":
                gap_str = f"**{gap_str}**" # Highlight
            
            md_content += f"| Severity {cls} | {m.capitalize()} | {b_metrics[m]:.4f} | {o_metrics[m]:.4f} | {gap_str} |\n"
        md_content += "| | | | | |\n"

    # 3. Fatal Accident Detection
    md_content += "## 3. Fatal Accident Detection (Severity 4)\n\n"
    
    # Severity 4 is Key "3"
    b_recall = get_class_metrics(baseline['classification_report'], "3")['recall']
    o_recall = get_class_metrics(ours['classification_report'], "3")['recall']
    
    # Confusion Matrix Counts for Class 4 (Assuming 4th row/col, index 3)
    # Note: JSON saves CM as list of lists
    b_cm = baseline['confusion_matrix']
    o_cm = ours['confusion_matrix']
    
    # True Positives for Class 4 (Index 3)
    b_tp = b_cm[3][3]
    o_tp = o_cm[3][3]
    
    # Total Actual Class 4
    b_total = sum(b_cm[3])
    o_total = sum(o_cm[3])
    
    md_content += f"- **CatBoost Recall**: {b_recall:.2%} ({b_tp} detected out of {b_total})\n"
    md_content += f"- **CrashSeverityNet Recall**: {o_recall:.2%} ({o_tp} detected out of {o_total})\n"
    md_content += f"- **Improvement**: +{o_recall - b_recall:.2%}\n\n"
    
    print(f"Writing to {os.path.abspath('thesis_materials/results_summary.md')}")
    try:
        with open("thesis_materials/results_summary.md", 'w') as f:
            f.write(md_content)
        print("Generated thesis_materials/results_summary.md")
    except Exception as e:
        print(f"Error writing file: {e}")

if __name__ == "__main__":
    try:
        generate_results_summary()
    except Exception as e:
        print(f"Script failed: {e}")
