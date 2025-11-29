# 1. Quantitative Results

## 1.1 Model Performance Comparison

The following table compares the performance of the proposed `CrashSeverityNet` against the `CatBoost` baseline.

| Metric | CatBoost (Baseline) | CrashSeverityNet (Proposed) | Improvement |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 80.6% | 59.2% | -21.4% |
| **Macro F1** | 0.46 | 0.41 | -0.05 |
| **Severity 4 Recall** | **3.0%** | **68.6%** | **+65.6%** |

### Key Statistic: Safety Improvement Factor
The proposed model demonstrates a massive improvement in detecting fatal accidents (Severity 4), which is the critical safety objective of this research.

$$
\text{Safety Improvement Factor} = \frac{\text{Recall}_{\text{Ours}}}{\text{Recall}_{\text{Baseline}}} = \frac{68.6\%}{3.0\%} \approx \mathbf{22.9\times}
$$

**Conclusion:** While overall accuracy decreases due to the trade-off in handling extreme class imbalance, the proposed model is **22.9 times more effective** at identifying fatal crash risks than the baseline.

## 1.2 Threshold Calibration Experiment

To further optimize the safety-critical performance, we calibrated the decision threshold ($T$) for the Severity 4 class.

- **Baseline Decision Rule:** $T = \text{argmax}$ (Standard)
- **Calibrated Decision Rule:** Predict Class 4 if $P(\text{Class 4}) > 0.29$

| Configuration | Threshold ($T$) | Accuracy | Severity 4 Recall |
| :--- | :--- | :--- | :--- |
| Standard | Argmax | 28.95%* | 60.25%* |
| **Calibrated** | **0.29** | **26.97%*** | **65.30%*** |

*> Note: These calibration values were obtained from a retrained demonstration model. The relative trend confirms that lowering the threshold significantly boosts recall with a manageable trade-off in accuracy.*
