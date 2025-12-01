import json
import os
import pandas as pd
import glob

def load_metrics(result_dir, experiment_ids):
    results = []
    for exp_id in experiment_ids:
        file_path = os.path.join(result_dir, f"{exp_id}.json")
        if not os.path.exists(file_path):
            print(f"Warning: Result file not found for {exp_id}")
            continue
            
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        metrics = data.get("metrics", {})
        clf_report = metrics.get("classification_report", {})
        
        # Extract Severity 4 (Fatal) metrics
        # Note: Class indices are strings in JSON
        sev4 = clf_report.get("3", {}) # 0-indexed 3 is Severity 4
        
        results.append({
            "Model ID": exp_id,
            "Accuracy": metrics.get("accuracy", 0),
            "Macro F1": metrics.get("macro_f1", 0),
            "Weighted F1": metrics.get("weighted_f1", 0),
            "Fatal Recall": sev4.get("recall", 0),
            "Fatal Precision": sev4.get("precision", 0),
            "Fatal F1": sev4.get("f1-score", 0)
        })
        
    return pd.DataFrame(results)

def main():
    config_path = "configs/exp_official_grid.json"
    result_dir = "results"
    
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    exp_ids = [exp["id"] for exp in config["experiments"]]
    
    df = load_metrics(result_dir, exp_ids)
    
    if df.empty:
        print("No results found.")
        return

    # Sort by Fatal Recall descending
    df = df.sort_values("Fatal Recall", ascending=False)
    
    # Save to CSV
    df.to_csv("results/summary_official_grid.csv", index=False)
    
    print("\n### Official Experiment Grid Results")
    print(df.to_markdown(index=False, floatfmt=".4f"))
    print("\n")

if __name__ == "__main__":
    main()
