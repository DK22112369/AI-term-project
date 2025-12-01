# Calibration and Cost Evaluation

This document describes the calibration analysis and cost evaluation metrics used to assess the reliability and safety of the CrashSeverityNet model, with a specific focus on **Severity 4 (Fatal)** crashes.

## 1. Metrics Definitions

### Brier Score
The Brier Score measures the accuracy of probabilistic predictions. It is the mean squared difference between the predicted probability and the actual outcome.
- **Range**: 0 to 1 (Lower is better).
- **Interpretation**: A score of 0 indicates perfect accuracy and certainty. For binary classification (Fatal vs Non-Fatal), it penalizes overconfident wrong predictions heavily.

### Reliability Curve (Calibration Curve)
A plot of the **Mean Predicted Probability** vs. the **Fraction of Positives** (Actual Probability) for binned predictions.
- **Perfect Calibration**: Points lie on the diagonal ($y=x$).
- **Below Diagonal**: Model is **overconfident** (predicts high probability, but event happens less often).
- **Above Diagonal**: Model is **underconfident** (predicts low probability, but event happens more often).

### Expected Cost
A domain-specific metric that assigns different penalties to different types of errors. In safety-critical applications, missing a fatal crash (False Negative) is much worse than a false alarm (False Positive).

**Cost Matrix:**
- **Fatal False Negative (Missed Fatal)**: Cost = **100.0**
- **Fatal False Positive (False Alarm)**: Cost = **1.0**
- **Non-Fatal Misclassification**: Cost = **0.1**

$$ \text{Expected Cost} = \frac{1}{N} \sum_{i=1}^{N} \text{Cost}(y_{true}^{(i)}, y_{pred}^{(i)}) $$

## 2. Usage

To run the calibration analysis on the **FULL dataset** (no sampling):

```bash
python experiments/calibration_analysis.py \
  --model_path models/csn_ce_weighted_official.pt \
  --preprocessor_path models/csn_ce_weighted_official_preprocessors.joblib \
  --output_dir results/calibration
```

### Outputs
The script generates the following in `results/calibration/`:
1.  `calibration_metrics.json`: Contains Brier Score and Expected Cost.
2.  `reliability_curve.csv`: Data points for the reliability plot.
3.  `reliability_plot.png`: Visualization of the reliability curve for Severity 4.

## 3. Interpretation Guide

- **High Brier Score**: The model's probability estimates are unreliable.
- **High Expected Cost**: The model is making too many dangerous errors (Fatal FNs).
- **Reliability Curve Shape**:
    - If the curve is "S-shaped", the model is underconfident at low probs and overconfident at high probs.
    - If the curve is entirely below the diagonal, the model is consistently overconfident about fatal crashes.
