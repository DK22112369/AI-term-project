# Cross-Dataset & Region Generalization Plan

This document outlines the strategy for evaluating the robustness and generalization capability of the CrashSeverityNet model trained on US Accidents data.

## 1. Region-Based Generalization (Domain Shift)

**Goal**: Evaluate how well the model generalizes to unseen geographical regions within the US.

### Methodology
- **Split Strategy**: `region`
- **Region A (Train/Val)**: West Coast States (`CA`, `OR`, `WA`, `NV`, `AZ`)
- **Region B (Test)**: Midwest/East Coast States (`NY`, `PA`, `OH`, `MI`, `IL`)
- **Hypothesis**: Performance may drop due to differences in road infrastructure, weather patterns, and reporting standards between regions.

### Usage
To run the region generalization experiment:

```bash
# Train on Region A, Test on Region B
python experiments/run_model_grid.py --config configs/exp_region_generalization.json
```

Results will be saved in `results/` with names like `csn_region_gen_*.json`.

## 2. Cross-Dataset Generalization (FARS)

**Goal**: Evaluate "Zero-Shot" performance on the NHTSA Fatality Analysis Reporting System (FARS) dataset.

### Methodology
- **Source Domain**: US Accidents (All Severities 1-4)
- **Target Domain**: FARS (Fatal Crashes Only - Severity 4)
- **Mapping**: FARS columns are mapped to the US Accidents schema.
    - `HOUR` -> `Start_Hour`
    - `WEATHER` -> `Weather_Condition`
    - `LATITUDE`/`LONGITUD` -> `Start_Lat`/`Start_Lng`
    - `Severity` -> Fixed to 4 (Fatal)

### Usage

**Step 1: Preprocess FARS Data**
Requires a raw FARS CSV file (e.g., `accident.csv` from NHTSA).

```bash
python data/preprocess_fars.py \
  --fars_path data/raw/FARS_2020_accident.csv \
  --preprocessor_path models/csn_ce_weighted_official_preprocessors.joblib \
  --output_path data/processed/fars_features.npz
```

**Step 2: Run Evaluation**
Evaluates a trained US Accidents model on the preprocessed FARS data.

```bash
python experiments/cross_dataset_eval.py \
  --model_path models/csn_ce_weighted_official.pt \
  --data_path data/processed/fars_features.npz \
  --output_dir results/cross_dataset
```

### Metrics
- **Fatal Recall**: The primary metric of interest (since all FARS samples are fatal).
- **Accuracy**: Equivalent to Fatal Recall in this specific zero-shot setting (if dataset is pure fatal).
