# 5. Data Integrity & Splitting Strategy Report

## 5.1 Training and Test Set Separation

To ensure the validity of the experimental results and prevent **Data Leakage**, strict separation between the Training and Inference (Test) datasets was enforced throughout the pipeline.

### Splitting Methodology
The dataset was split into three distinct subsets using a **Time-based Splitting Strategy** to simulate real-world forecasting scenarios (predicting future accidents based on past data).

1.  **Training Set (60%)**: Used for model parameter updates.
2.  **Validation Set (20%)**: Used for hyperparameter tuning and Early Stopping.
3.  **Test Set (20%)**: Used **ONLY** for final performance evaluation.

**Verification of Code Logic:**
- The splitting function `time_based_split` sorts the entire dataset by `Start_Time` before dividing indices.
- **Overlap Check**:
    - `Training Period`: Earliest 60% of data.
    - `Test Period`: Latest 20% of data.
    - **Intersection**: $\text{Train} \cap \text{Test} = \emptyset$ (Empty Set).

## 5.2 Preprocessing & Data Leakage Prevention

A critical aspect of robust machine learning is ensuring that information from the Test set does not leak into the Training process via feature engineering.

### Feature Scaling & Encoding
- **Fit**: The `StandardScaler` (for numerical features) and `OneHotEncoder` (for categorical features) were fitted **ONLY on the Training Set**.
- **Transform**: The fitted transformers were then applied to the Validation and Test sets.
- **Handling Unknowns**:
    - Numerical: Scaled using Train set statistics (Mean/Std).
    - Categorical: Unknown categories in Test set (not seen in Train) were handled via `handle_unknown='ignore'` (OneHotEncoder), resulting in all-zero vectors for those specific categories.

### Code Evidence
The following logic from `scripts/train.py` demonstrates this strict separation:

```python
# 1. Split Data (Time-based)
df_train, df_val, df_test = time_based_split(df)

# 2. Fit Transformers ONLY on Train
bundle = fit_feature_transformers(df_train)

# 3. Transform All Sets using the Train-fitted Bundle
X_train, ... = transform_with_preprocessors(df_train, bundle)
X_test, ...  = transform_with_preprocessors(df_test, bundle)
```

## 5.3 Conclusion
The experimental setup strictly adheres to academic standards for data integrity. The model evaluated on the Test set has **never seen** the test samples during training, nor has it been influenced by their global statistics during preprocessing. Therefore, the reported performance metrics reflect the model's true generalization capability.
