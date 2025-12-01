import pandas as pd
import sys
import os

def analyze_sweep(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    # 1. Baseline (Threshold ~ 0.5)
    baseline = df.iloc[(df['threshold'] - 0.5).abs().argsort()[:1]].iloc[0]
    
    # 2. Fail-Safe Point (e.g., Threshold ~ 0.3 or Max F2)
    # Let's pick Threshold = 0.3 as a representative point
    failsafe = df.iloc[(df['threshold'] - 0.3).abs().argsort()[:1]].iloc[0]
    
    # 3. Max F2 Point
    max_f2_idx = df['fatal_f2'].idxmax()
    max_f2 = df.iloc[max_f2_idx]
    
    print("\n=== Threshold Sweep Analysis ===")
    print(f"Baseline (t={baseline['threshold']:.2f}):")
    print(f"  Accuracy: {baseline['accuracy']:.4f}")
    print(f"  Fatal Recall: {baseline['fatal_recall']:.4f}")
    print(f"  Fatal F2: {baseline['fatal_f2']:.4f}")
    
    print(f"\nFail-Safe (t={failsafe['threshold']:.2f}):")
    print(f"  Accuracy: {failsafe['accuracy']:.4f}")
    print(f"  Fatal Recall: {failsafe['fatal_recall']:.4f}")
    print(f"  Fatal F2: {failsafe['fatal_f2']:.4f}")
    
    print(f"\nMax F2 (t={max_f2['threshold']:.2f}):")
    print(f"  Accuracy: {max_f2['accuracy']:.4f}")
    print(f"  Fatal Recall: {max_f2['fatal_recall']:.4f}")
    print(f"  Fatal F2: {max_f2['fatal_f2']:.4f}")
    print(f"  Fatal F2: {max_f2['fatal_f2']:.4f}")
    print("================================")

    # Save to JSON
    import json
    results = {
        "baseline_0.5": {
            "threshold": float(baseline['threshold']),
            "accuracy": float(baseline['accuracy']),
            "recall_class_4": float(baseline['fatal_recall'])
        },
        "failsafe": {
            "threshold": float(failsafe['threshold']),
            "accuracy": float(failsafe['accuracy']),
            "recall_class_4": float(failsafe['fatal_recall'])
        },
        "max_f2": {
            "threshold": float(max_f2['threshold']),
            "accuracy": float(max_f2['accuracy']),
            "recall_class_4": float(max_f2['fatal_recall'])
        }
    }
    with open("results/threshold_sweep/threshold_analysis.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Saved analysis to results/threshold_sweep/threshold_analysis.json")

if __name__ == "__main__":
    analyze_sweep("results/threshold_sweep/threshold_sweep_results.csv")
