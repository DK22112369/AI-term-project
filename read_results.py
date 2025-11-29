import json
import os
import pandas as pd

def print_metrics(path, name):
    print(f"--- {name} ---")
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    with open(path, 'r') as f:
        data = json.load(f)
    
    metrics = data.get('metrics', {})
    report = metrics.get('classification_report', {})
    cm = metrics.get('confusion_matrix', [])
    
    # Severity 4 is usually class '3' (0-indexed) or '4' (1-indexed)
    # The report keys are strings "0", "1", "2", "3" or "1", "2", "3", "4"
    
    target_class = "4" if "4" in report else "3"
    
    if target_class in report:
        print(f"Severity 4 Metrics:")
        print(f"  Precision: {report[target_class]['precision']:.4f}")
        print(f"  Recall:    {report[target_class]['recall']:.4f}")
        print(f"  F1-Score:  {report[target_class]['f1-score']:.4f}")
        print(f"  Support:   {report[target_class]['support']}")
    else:
        print("Severity 4 not found in report keys:", report.keys())
        
    print("Confusion Matrix:")
    for row in cm:
        print(row)
    print("\n")

print_metrics("results/baseline_catboost_time.json", "CatBoost (Time Split)")
print_metrics("results/crash_severity_net_ce_time.json", "CrashSeverityNet (CE + Time Split)")
print_metrics("results/crash_severity_net_focal.json", "CrashSeverityNet (Focal)")
