# Literature Gap Analysis: Crash Severity Prediction

This document outlines the gaps between the current `CrashSeverityNet` project and state-of-the-art (SOTA) research in traffic accident severity prediction (2023-2024). Addressing these gaps or discussing them as limitations will significantly enhance the quality of a thesis or academic paper.

## 1. Model Architecture: "MLP is too simple" (The Architecture Gap)
The current project uses a **3-Stream MLP (Late Fusion)**. While logical, it is considered a "classic" approach compared to modern tabular deep learning.

*   **SOTA Trends**:
    *   **TabNet (Google)**: Combines the feature selection of Decision Trees with the representation power of DNNs. Often outperforms MLP on tabular data.
    *   **TabTransformer / FT-Transformer**: Applies **Self-Attention** mechanisms (like BERT) to categorical features to learn complex feature interactions.
    *   **Hybrid Models**: Combining CNN (for spatial grid data) or LSTM (for temporal sequences) is common for spatiotemporal analysis.
*   **Recommendation**:
    *   **Defense**: "We focused on interpreting the contribution of distinct feature groups (Driver, Env, Time) via Late Fusion."
    *   **Future Work**: "Future research will incorporate Self-Attention mechanisms (e.g., TabTransformer) to capture feature interactions more effectively."

## 2. Data Utilization: "Ignoring Text Data" (The Unstructured Data Gap)
The current preprocessing drops the `Description` column. Many high-impact papers utilize this unstructured text.

*   **SOTA Trends**:
    *   **NLP Fusion**: Using **BERT**, **RoBERTa**, or **DistilBERT** to generate embeddings from accident descriptions (e.g., "Multi-vehicle collision blocking left lane") and fusing them with tabular features.
    *   Text often contains critical context (rollover, fire, secondary crash) not captured in structured columns.
*   **Recommendation**:
    *   **Limitation**: Acknowledge that excluding text data limits the model's understanding of accident context.

## 3. Class Imbalance: "Lack of Advanced Sampling" (The Sampling Gap)
The project uses `WeightedRandomSampler` (Random Oversampling) and `Focal Loss`. While effective, they are standard techniques.

*   **SOTA Trends**:
    *   **SMOTE-NC**: Synthetic Minority Over-sampling Technique for Nominal and Continuous data. Generates *synthetic* samples rather than duplicating existing ones, reducing overfitting.
    *   **Hybrid Sampling**: Combining Undersampling (for majority class) and SMOTE (for minority class).
    *   **Generative Models (GANs)**: Using Tabular GANs (CTGAN) to generate realistic minority samples.
*   **Recommendation**:
    *   **Defense**: "Weighted Loss and Focal Loss were chosen for computational efficiency during batch training."
    *   **Future Work**: "Investigate SMOTE-NC or GAN-based augmentation."

## 4. Baseline Models: "Beyond XGBoost" (The Baseline Gap)
The project compares against Random Forest and XGBoost.

*   **SOTA Trends**:
    *   **CatBoost**: Handles categorical features natively and often outperforms XGBoost on datasets with many categories (like traffic data).
    *   **LightGBM**: Faster and often more accurate for large datasets.
*   **Recommendation**:
    *   Adding **CatBoost** or **LightGBM** to the baseline comparison would strengthen the evaluation.

## 5. Evaluation Rigor: "Statistical Significance" (The Statistical Gap)
The current evaluation relies on a single Train/Val/Test split.

*   **SOTA Trends**:
    *   **K-Fold Cross Validation**: Reporting average metrics over 5 or 10 folds.
    *   **Statistical Tests**: Using **t-tests** or **Wilcoxon signed-rank tests** to prove that the proposed model's improvement over the baseline is statistically significant, not just due to random seed luck.
*   **Recommendation**:
    *   If K-Fold is too expensive, run the experiment with **5 different random seeds** and report "Mean ± Std".

## Summary for Thesis Discussion
> "While this study successfully demonstrates the efficacy of a Group-wise Late Fusion architecture and Weighted Loss for handling class imbalance, it has limitations. We did not utilize unstructured text data (Description) or advanced synthetic sampling (SMOTE-NC). Future work should explore Transformer-based tabular architectures and multi-modal learning to further improve minority class prediction."
