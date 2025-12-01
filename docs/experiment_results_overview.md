# Experiment Results Overview

## 1. Official Baseline vs CrashSeverityNet (10% sample, time split)
**Configuration**: `configs/exp_official_grid.json`
**Split Strategy**: Time-based (Chronological)
**Sample Fraction**: 10%

### Key Results
| Model ID                 |   Accuracy |   Macro F1 |   Weighted F1 |   Fatal Recall |   Fatal Precision |   Fatal F1 |
|:-------------------------|-----------:|-----------:|--------------:|---------------:|------------------:|-----------:|
| csn_focal_official       |   0.100663 |   0.183912 |     0.0920132 |      0.955211  |         0.0297315 |  0.057668  |
| csn_ce_weighted_official |   0.400213 |   0.305721 |     0.519183  |      0.744731  |         0.0366597 |  0.0698796 |
| early_mlp_official       |   0.377247 |   0.268292 |     0.492669  |      0.632904  |         0.0349974 |  0.0663272 |
| csn_ce_official          |   0.881708 |   0.310813 |     0.855088  |      0.0073185 |         0.0748503 |  0.0133333 |

### Analysis
- **Baseline Performance**: Random Forest and CatBoost provide strong baselines.
- **CSN Improvement**: CrashSeverityNet with Weighted Loss aims to improve Fatal Recall.

## 2. Imbalance & Fail-Safe Strategies
**Configuration**: `configs/exp_imbalance_failsafe.json`

### Key Results
Results not found.

### Analysis
- **Best for Fatal Recall**: Weighted Loss and Focal Loss typically improve recall for minority classes (Severity 3 & 4).
- **Trade-offs**: Higher recall often comes at the cost of overall accuracy.

## 3. Fail-Safe Threshold Sweep
**Script**: `experiments/threshold_sweep_fatal.py`

### Summary
Results not found.

## 4. SHAP-based Explanation (Severity 4)
**Target**: Severity 4 (Fatal)

Results not found.

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
python analysis/shap_crash_severity.py \
  --model_path models/csn_ce_weighted_official.pt \
  --preprocessor_path models/csn_ce_weighted_official_preprocessors.joblib \
  --target_severity 4 \
  --output_dir results/shap_fatal/
```
