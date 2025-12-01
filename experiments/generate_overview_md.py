import json
import pandas as pd
import os

def generate_overview():
    print(f"Current CWD: {os.getcwd()}")
    print(f"Checking for results/summary_official_grid.csv: {os.path.exists('results/summary_official_grid.csv')}")
    print("Generating Experiment Results Overview...")
    
    # 1. Official Grid Results
    official_md = ""
    if os.path.exists("results/summary_official_grid.csv"):
        df = pd.read_csv("results/summary_official_grid.csv")
        official_md = df.to_markdown(index=False)
    else:
        official_md = "Results not found."

    # 2. Imbalance Grid Results
    imbalance_md = ""
    if os.path.exists("results/summary_imbalance_failsafe.csv"):
        df = pd.read_csv("results/summary_imbalance_failsafe.csv")
        imbalance_md = df.to_markdown(index=False)
    else:
        imbalance_md = "Results not found."

    # 3. Threshold Sweep
    sweep_md = ""
    if os.path.exists("results/threshold_sweep/threshold_analysis.json"):
        with open("results/threshold_sweep/threshold_analysis.json", 'r') as f:
            sweep_data = json.load(f)
            baseline = sweep_data.get("baseline_0.5", {})
            failsafe = sweep_data.get("failsafe", {})
            
            sweep_md = f"""
- **Baseline Threshold** ($\\tau \\approx 0.5$):
    - Accuracy: {baseline.get('accuracy', 0):.4f}
    - Fatal Recall: {baseline.get('recall_class_4', 0):.4f}
- **Fail-Safe Threshold** ($\\tau \\approx {failsafe.get('threshold', 0):.4f}$):
    - Accuracy: {failsafe.get('accuracy', 0):.4f}
    - Fatal Recall: {failsafe.get('recall_class_4', 0):.4f}
- **Trade-off**: Lowering the threshold increases Fatal Recall while decreasing Accuracy.
"""
    else:
        sweep_md = "Results not found."

    # 4. SHAP
    shap_md = ""
    if os.path.exists("results/shap_fatal/group_importance_sev4.json"):
        with open("results/shap_fatal/group_importance_sev4.json", 'r') as f:
            group_imp = json.load(f)
            shap_md += "### Group Importance\n"
            for k, v in group_imp.items():
                shap_md += f"- **{k}**: {v:.4f}\n"
    
    if os.path.exists("results/shap_fatal/shap_feature_importance_sev4.csv"):
        df_shap = pd.read_csv("results/shap_fatal/shap_feature_importance_sev4.csv")
        top_features = df_shap.head(5)
        shap_md += "\n### Top Features\n"
        for _, row in top_features.iterrows():
            shap_md += f"1. **{row['feature_name']}**: {row['mean_abs_shap']:.4f}\n"
    else:
        shap_md += "Results not found."

    # Combine into Markdown
    content = f"""# Experiment Results Overview

## 1. Official Baseline vs CrashSeverityNet (10% sample, time split)
**Configuration**: `configs/exp_official_grid.json`
**Split Strategy**: Time-based (Chronological)
**Sample Fraction**: 10%

### Key Results
{official_md}

### Analysis
- **Baseline Performance**: Random Forest and CatBoost provide strong baselines.
- **CSN Improvement**: CrashSeverityNet with Weighted Loss aims to improve Fatal Recall.

## 2. Imbalance & Fail-Safe Strategies
**Configuration**: `configs/exp_imbalance_failsafe.json`

### Key Results
{imbalance_md}

### Analysis
- **Best for Fatal Recall**: Weighted Loss and Focal Loss typically improve recall for minority classes (Severity 3 & 4).
- **Trade-offs**: Higher recall often comes at the cost of overall accuracy.

## 3. Fail-Safe Threshold Sweep
**Script**: `experiments/threshold_sweep_fatal.py`

### Summary
{sweep_md}

## 4. SHAP-based Explanation (Severity 4)
**Target**: Severity 4 (Fatal)

{shap_md}

## 5. How to Reproduce
```bash
# Official grid
python experiments/run_model_grid.py --config configs/exp_official_grid.json
python experiments/summary_official_grid.py

# Imbalance/failsafe grid
python experiments/run_model_grid.py --config configs/exp_imbalance_failsafe.json
python experiments/summary_imbalance_failsafe.py

# Threshold sweep
python experiments/threshold_sweep_fatal.py --model_path models/csn_ce_weighted_official.pt --preprocessor_path models/csn_ce_weighted_official_preprocessors.joblib
python experiments/analyze_threshold_sweep.py

# SHAP (Fatal)
python analysis/shap_crash_severity.py \\
  --model_path models/csn_ce_weighted_official.pt \\
  --preprocessor_path models/csn_ce_weighted_official_preprocessors.joblib \\
  --target_severity 4 \\
  --output_dir results/shap_fatal/
```
"""

    with open("docs/experiment_results_overview.md", "w") as f:
        f.write(content)
    
    print("Overview generated at docs/experiment_results_overview.md")

if __name__ == "__main__":
    generate_overview()
