# Experiment Guide

This guide explains how to run various experiments for the Crash Severity Prediction project, including deep learning models, machine learning baselines, and analysis tools.

## 1. Deep Learning Models
Use `train_crash_severity_net.py` to train deep learning models.

### Options
- **`--model_type`**:
    - `crash_severity_net`: The default 3-stream Late Fusion model.
    - `early_mlp`: A single-stream Early Fusion MLP baseline.
- **`--loss_type`**:
    - `ce`: Standard Cross Entropy Loss.
    - `ce_weighted`: Class-Weighted Cross Entropy (handles imbalance).
    - `focal`: Focal Loss (handles imbalance).
- **`--gamma`**:
    - Gamma parameter for Focal Loss (default: 2.0).
- **`--use_sampler`**:
    - Add this flag to use `WeightedRandomSampler` for oversampling minority classes.
- **`--split_strategy`**:
    - `random`: Stratified random split (default).
    - `time`: Time-based split (past -> future).
- **`--seed`**:
    - Random seed for reproducibility (default: 42).

### Examples
**Run Default Model (Weighted Loss, Time Split):**
```bash
python train_crash_severity_net.py --loss_type ce_weighted --split_strategy time --epochs 20
```

**Run TabTransformer (Advanced Architecture):**
```bash
# Note: Currently uses EarlyFusionMLP as placeholder for OHE data
python train_crash_severity_net.py --model_type early_mlp --epochs 20
```

**Run with SMOTE-NC (Advanced Sampling):**
```bash
# Note: Requires imbalanced-learn
python train_crash_severity_net.py --use_sampler --epochs 20
```

## 2. Baseline ML Models
Use `baselines/train_baseline_ml.py` to train traditional ML models.

### Options
- **`--model_type`**:
    - `rf`: Random Forest Classifier.
    - `xgb`: XGBoost Classifier.
    - `catboost`: CatBoost Classifier (Recommended SOTA).
    - `lgbm`: LightGBM Classifier (Fast SOTA).
- **`--split_strategy`**: `random` or `time`.

### Examples
**Run CatBoost (SOTA Baseline):**
```bash
python baselines/train_baseline_ml.py --model_type catboost --split_strategy time
```

## 3. Rigorous Evaluation (K-Fold CV)
Run 5-Fold Cross Validation to get statistically significant results (Mean ± Std).

```bash
python run_kfold_evaluation.py --model_type catboost --folds 5
```

## 4. Analysis & Inference

### SHAP Analysis (Explainability)
Generate SHAP summary plots for Deep Learning models.
```bash
python analysis/explain_model.py \
  --model_path models/crash_severity_net_ce_weighted.pt \
  --preprocessor_path models/crash_severity_net_ce_weighted_preprocessors.joblib
```

### Real-time Inference
Predict severity for a single sample (JSON input).
```bash
python inference/predict.py \
  --model_path models/crash_severity_net_ce_weighted.pt \
  --preprocessor_path models/crash_severity_net_ce_weighted_preprocessors.joblib \
  --input_json '{"Start_Lat": 39.0, "Start_Lng": -84.0, "Distance(mi)": 0.1, ...}'
```

## 4. Results Structure
All results are saved in the `results/` directory:
- **Metrics**: `[name].json` (Accuracy, F1, etc.)
- **Config**: `[name]_config.json` (Hyperparameters)
- **Feature Importance**: `[name]_feature_importance.csv` (for RF/XGB)
- **Confusion Matrices**: `confmat_[name].png`
- **Loss Curves**: `loss_curve_[name].png`
- **SHAP Plots**: `shap_summary.png`
