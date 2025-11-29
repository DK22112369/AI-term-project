# CrashSeverityNet: Traffic Accident Severity Prediction

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive deep learning framework for predicting traffic accident severity using the US Accidents dataset. This project implements group-wise late fusion architecture (CrashSeverityNet) alongside state-of-the-art baselines and advanced class imbalance handling techniques.

---

## 🚨 Important: Dataset Notice

**This repository contains CODE ONLY, not the dataset.**

- The US Accidents dataset must be downloaded separately from **[Kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)**.
- Per Kaggle's Terms of Use, we **cannot redistribute the original data files** in this repository.
- After downloading, place the dataset in a local directory and specify the path using the `--data_path` argument when running scripts.

**Example**:
```bash
python train_crash_severity_net.py --data_path /path/to/US_Accidents_March23.csv
```

---

## 📋 Project Overview

### Features
- ✅ **Novel Architecture**: Group-wise Late Fusion (CrashSeverityNet) for multi-modal tabular data
- ✅ **Advanced Baselines**: RandomForest, XGBoost, CatBoost, LightGBM, TabTransformer
- ✅ **Class Imbalance Handling**: Weighted Loss, Focal Loss, SMOTE-NC, WeightedRandomSampler
- ✅ **Rigorous Evaluation**: K-Fold Cross Validation, Time-based Splitting
- ✅ **Explainability**: SHAP analysis for model interpretability
- ✅ **Production-Ready**: Real-time inference API with saved preprocessors

### Key Results (Preliminary - 1% Sample)
- **CatBoost Baseline**: 89.7% accuracy
- **TabTransformer**: 78.3% accuracy
- **SMOTE-NC + EarlyMLP**: 62.1% accuracy with significantly improved minority class recall

---

## 🗂️ Repository Structure

```
CrashSeverityNet/
├── data/
│   └── preprocess_us_accidents.py      # Data loading, cleaning, feature engineering
├── models/
│   ├── crash_severity_net.py          # Proposed Late Fusion model
│   ├── early_fusion_mlp.py            # Early Fusion baseline
│   ├── tab_transformer.py              # TabTransformer implementation
│   └── losses.py                       # Focal Loss
├── baselines/
│   └── train_baseline_ml.py           # RF, XGB, CatBoost, LGBM training
├── utils/
│   ├── common.py                       # Reproducibility utilities (set_seed)
│   └── metrics.py                      # Evaluation metrics
├── visualization/
│   └── plots.py                        # Confusion matrix, loss curves
├── analysis/
│   └── explain_model.py                # SHAP explainability
├── inference/
│   └── predict.py                      # Real-time inference API
├── train_crash_severity_net.py         # Main training script
├── run_kfold_evaluation.py             # K-Fold Cross Validation
└── docs/
    ├── experiment_guide.md             # How to run experiments
    ├── literature_gap_analysis.md      # Comparison with SOTA research
    └── thesis_summary.md               # Comprehensive research documentation
```

---

## 🛠️ Installation

### Requirements
- Python 3.9+
- CUDA 11.8+ (optional, for GPU acceleration)

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/CrashSeverityNet.git
cd CrashSeverityNet

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install torch torchvision scikit-learn pandas numpy matplotlib seaborn tqdm
pip install xgboost catboost lightgbm imbalanced-learn shap joblib
```

---

## 🚀 Quick Start

### 1. Download Dataset
Download the US Accidents dataset from [Kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents) and place it in a local directory.

### 2. Train CrashSeverityNet (Proposed Model)
```bash
python train_crash_severity_net.py \
  --model_type crash_severity_net \
  --loss_type ce_weighted \
  --split_strategy time \
  --epochs 20 \
  --sample_frac 0.1 \
  --data_path /path/to/US_Accidents_March23.csv
```

### 3. Train CatBoost Baseline
```bash
python baselines/train_baseline_ml.py \
  --model_type catboost \
  --split_strategy time \
  --sample_frac 0.1 \
  --data_path /path/to/US_Accidents_March23.csv
```

### 4. Run K-Fold Cross Validation
```bash
python run_kfold_evaluation.py \
  --model_type catboost \
  --folds 5 \
  --sample_frac 0.1 \
  --data_path /path/to/US_Accidents_March23.csv
```

### 5. Run Inference
```bash
python inference/predict.py \
  --model_path models/crash_severity_net_ce_weighted_time.pt \
  --preprocessor_path models/crash_severity_net_ce_weighted_time_preprocessors.joblib \
  --input_json '{"Start_Lat": 39.0, "Start_Lng": -84.0, "Temperature(F)": 55.0, ...}'
```

---

## 📚 Documentation

- **[Experiment Guide](docs/experiment_guide.md)**: Detailed instructions for running all experiments
- **[Literature Gap Analysis](docs/literature_gap_analysis.md)**: Comparison with state-of-the-art research
- **[Thesis Summary](docs/thesis_summary.md)**: Comprehensive research planning and methodology document

---

## 🎯 Key Contributions

1. **Group-wise Late Fusion Architecture** tailored for heterogeneous tabular data
2. **Systematic Comparison** of 4 class imbalance handling techniques
3. **SOTA Baselines**: First study comparing CatBoost, LightGBM, and TabTransformer on US Accidents
4. **Rigorous Evaluation**: Time-based splitting + K-Fold CV prevents data leakage
5. **Production-Ready**: Inference API with SHAP explainability

---

## 📝 Citation

If you use this code or methodology in your research, please cite:

```bibtex
@misc{crashseveritynet2025,
  title={CrashSeverityNet: Advanced Deep Learning Framework for Traffic Accident Severity Prediction},
  author={CrashSeverityNet Research Team},
  year={2025},
  publisher={GitHub},
  howpublished={\url{https://github.com/yourusername/CrashSeverityNet}}
}
```

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset**: Sobhan Moosavi et al., "A Countrywide Traffic Accident Dataset" (Kaggle)
- **Frameworks**: PyTorch, scikit-learn, CatBoost, LightGBM, XGBoost
- **Inspiration**: TabTransformer (Huang et al., 2020), Focal Loss (Lin et al., 2017)

---

## ⚠️ Disclaimer

This project is for **research and educational purposes only**. The models and results presented are not intended for use in safety-critical applications without further validation and testing.

---

## 📧 Contact

For questions or collaboration inquiries, please open an issue on GitHub.
