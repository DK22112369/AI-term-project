import json
import os

def print_metrics(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    with open(file_path, 'r') as f:
        data = json.load(f)
        
    metrics = data.get("metrics", {})
    clf_report = metrics.get("classification_report", {})
    
    print(f"\n--- Results for {os.path.basename(file_path)} ---")
    print(f"Accuracy: {metrics.get('accuracy', 'N/A')}")
    print(f"Macro F1: {metrics.get('macro_f1', 'N/A')}")
    print(f"Weighted F1: {metrics.get('weighted_f1', 'N/A')}")
    
    if "3" in clf_report: # Severity 4 (0-indexed 3)
        sev4 = clf_report["3"]
        print(f"Severity 4 Recall: {sev4.get('recall', 'N/A')}")
        print(f"Severity 4 Precision: {sev4.get('precision', 'N/A')}")
        print(f"Severity 4 F1: {sev4.get('f1-score', 'N/A')}")
        print(f"Severity 4 Support: {sev4.get('support', 'N/A')}")
    else:
        print("Severity 4 metrics not found in report.")

files = [
    "results/baseline_catboost_time.json",
    "results/baseline_rf.json",
    "results/crash_severity_net_ce_weighted_time.json"
]

for f in files:
    print_metrics(f)
