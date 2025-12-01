# Project Report: CrashSeverityNet (Mk.2 Upgrade)

## 1. Project Summary
This project implements **CrashSeverityNet**, a deep learning framework designed to predict traffic accident severity (1-4) using the US Accidents dataset. The core objective is to overcome the "Accuracy Paradox" in imbalanced data by maximizing the **Recall of Fatal Accidents (Severity 4)** through a Group-wise Late Fusion architecture and Fail-Safe loss functions.

## 2. Directory Structure
```
project_root/
├── data/                   # Data loading & Preprocessing (US Accidents)
├── models/                 # Deep Learning Models (CrashSeverityNet, EarlyFusion, TabTransformer)
├── scripts/                # Main Execution Scripts (Train, Eval, Plot)
├── experiments/            # Hyperparameter Search & One-off Experiments
├── analysis/               # Explainability & Analysis (SHAP)
├── final_thesis_data/      # Generated Thesis Materials (Metrics, Figures, Specs)
├── thesis_materials/       # Legacy Results & Summaries
├── docs/                   # Documentation & Papers (KSAE Draft, Plans)
└── results/                # Experiment Outputs (Logs, Checkpoints, JSONs)
```

## 3. Key Features
- **Model**: `CrashSeverityNet` (Group-wise Late Fusion MLP).
- **Baselines**: `CatBoost`, `RandomForest`, `EarlyFusionMLP`.
- **Imbalance Handling**:
    - Weighted Cross Entropy Loss.
    - Focal Loss (for hard mining).
    - Threshold Calibration (Fail-Safe optimization).
- **Explainability**: SHAP (SHapley Additive exPlanations) analysis for feature importance.
- **Deployment**: Inference script for real-time prediction.

## 4. Key Python Files
| File | Role | Status |
| :--- | :--- | :--- |
| `scripts/train.py` | Main training loop (Train/Val/Test split, Logging). | **Active** |
| `models/crash_severity_net.py` | Definition of the Late Fusion architecture. | **Active** |
| `data/preprocess_us_accidents.py` | Cleaning, Feature Engineering, Scaling. | **Active** |
| `scripts/evaluate_kfold.py` | Robust performance verification via Cross-Validation. | **Active** |
| `scripts/calibrate.py` | Threshold optimization for Safety Recall. | **Active** |
| `analysis/shap_crash_severity.py` | SHAP-based explainability analysis. | *Skeleton* |

## 5. Implementation Status
- **Completed (Mk.1)**:
    - Core model implementation & training pipeline.
    - Quantitative evaluation & comparison with baselines.
    - Thesis data generation (Plots, Metrics, Specs).
- **In Progress (Mk.2)**:
    - KSAE Full Paper Draft (Korean).
    - Advanced SHAP Analysis.
    - Long-term Journal Extension Plan.
