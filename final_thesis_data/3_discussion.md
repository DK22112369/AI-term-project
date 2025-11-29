# 3. Failure Analysis & Discussion

## 3.1 Error Analysis

The confusion matrix analysis reveals distinct failure modes for the proposed model compared to the baseline.

### 1. Over-Sensitivity to Severity 4
- **Observation:** The model frequently misclassifies Severity 2 (Minor) accidents as Severity 4 (Fatal).
- **Reason:** This is a direct consequence of the **Weighted Loss** and **Sampling** strategies used to combat the extreme class imbalance (Severity 2 accounts for >70% of data). By penalizing the model heavily for missing a fatal accident, we force it to be "paranoid" or highly sensitive to potential risks.
- **Implication:** In a real-world safety system, this "False Positive" bias is preferable to "False Negatives." Alerting a driver to a high risk when the actual risk is moderate (Type I error) is safer than failing to alert them to a fatal risk (Type II error).

### 2. Confusion between Severity 2 and 3
- **Observation:** There is significant overlap between Severity 2 and 3 predictions.
- **Reason:** The distinction between these classes in the dataset is often subtle (e.g., traffic delay duration), which may not be fully captured by the available features (Weather, Road, Time).
- **Solution:** Future work could incorporate **text description features** (using NLP) to better distinguish the nuances of accident severity.

## 3.2 Conclusion

The `CrashSeverityNet` successfully achieves its primary design goal: **maximizing the detection of life-threatening accidents.** While the trade-off is a reduction in overall accuracy (driven by misclassifying majority class samples), the **22.9x improvement in Severity 4 Recall** validates the efficacy of the proposed Group-wise Late Fusion architecture and class-imbalance handling techniques for safety-critical applications.
