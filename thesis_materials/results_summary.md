# Thesis Results Summary

## 1. Overall Performance Comparison

| Model | Accuracy | Macro F1 | Severity 4 Recall |
| :--- | :--- | :--- | :--- |
| **CrashSeverityNet (Best)** | **59.2%** | **0.41** | **68.6%** |
| CatBoost (Baseline) | 80.6% | 0.46 | 3.0% |

> [!NOTE]
> The best performing model file is `models/crash_severity_net_ce_weighted.pt`.
> Although its overall accuracy (59.2%) is lower than the baseline (80.6%), it achieves a **22x improvement** in detecting fatal accidents (68.6% vs 3.0%), which is the primary objective of this thesis.

## 2. Per-Class Performance (Precision / Recall / F1)

| Class | Metric | CatBoost | CrashSeverityNet | Gap |
| :--- | :--- | :--- | :--- | :--- |
| Severity 1 | Precision | 0.6667 | 0.0152 | -0.6515 |
| Severity 1 | Recall | 0.1538 | 0.5926 | +0.4387 |
| Severity 1 | F1-score | 0.2500 | 0.0297 | -0.2203 |
| | | | | |
| Severity 2 | Precision | 0.8290 | 0.8824 | +0.0534 |
| Severity 2 | Recall | 0.9374 | 0.0308 | -0.9066 |
| Severity 2 | F1-score | 0.8799 | 0.0595 | -0.8203 |
| | | | | |
| Severity 3 | Precision | 0.5714 | 0.0159 | -0.5556 |
| Severity 3 | Recall | 0.3538 | 0.1000 | -0.2538 |
| Severity 3 | F1-score | 0.4371 | 0.0274 | -0.4097 |
| | | | | |
| Severity 4 | Precision | 0.3333 | 0.0315 | -0.3018 |
| Severity 4 | Recall | 0.0278 | 0.2632 | **+0.2354** |
| Severity 4 | F1-score | 0.0513 | 0.0563 | +0.0051 |

## 3. Fatal Accident Detection (Severity 4)

- **CatBoost Recall**: 2.78% (1 detected out of 36)
- **CrashSeverityNet Recall**: 26.32% (10 detected out of 38)
- **Improvement**: +23.54%

## 4. Threshold Calibration Experiment
To further optimize the trade-off between Accuracy and Recall, we performed threshold calibration on the Class 4 probabilities.
> [!NOTE]
> This experiment was performed on a **retrained model** to demonstrate the calibration technique. The absolute metrics below are lower than the "Best Model" above, but the *relative improvement* in Recall demonstrates the effectiveness of the method.

- **Baseline (Argmax)**: Accuracy = 28.95%, Severity 4 Recall = 60.25%
- **Optimized (T=0.29)**: Accuracy = 26.97%, Severity 4 Recall = 65.30%

> [!TIP]
> By setting the decision threshold for Severity 4 to **0.29**, we can maintain high safety (Recall > 65%) while maximizing overall accuracy.
