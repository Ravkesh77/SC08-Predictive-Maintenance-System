# SC08 Data Quality Report

## Purpose
This report documents the inspection, cleaning, validation, and stratified splitting of the AI4I 2020 Predictive Maintenance dataset for the SC08 Capstone Project.

## Dataset Source
- **Dataset:** AI4I 2020 Predictive Maintenance Dataset
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
- **Licence:** CC BY 4.0
- **Original File:** `data/raw/ai4i2020.csv`
- **Original Records:** 10,000
- **Target Column:** `Machine failure` (0 = No failure, 1 = Failure)

## Raw Data Inspection
| Check | Result |
|---|---|
| Total Rows | 10,000 |
| Total Columns | 14 |
| Missing values | 0 |
| Duplicate rows | 0 |
| Normal operations (0) | 9,661 |
| Machine failures (1) | 339 |

## Cleaning Method
- The original raw dataset was preserved untouched in `data/raw/`.
- Median imputation pipeline was configured for numeric feature integrity.
- Verified absence of duplicate observation rows.
- Target labels (`Machine failure`) were preserved strictly without artificial alterations.

## Validation Results
All validation rules passed with 0 violations:
- Required feature and target columns present: Yes
- Remaining missing values: 0
- Physical temperature ranges valid [250 K to 350 K]: Yes
- Physical operational parameters non-negative (speed, torque, tool wear >= 0): Yes
- Target labels valid (binary {0, 1}): Yes
- Duplicate rows remaining: 0

## Reproducible Stratified Split
A fixed random seed of `42` was used. The dataset was stratified by `Machine failure` to preserve the 3.39% failure rate across splits.

| Dataset | Rows | Failures (1) | Non-Failures (0) | Purpose |
|---|---|---|---|---|
| `train.csv` | 7,000 | 237 (3.39%) | 6,763 (96.61%) | Model training |
| `validation.csv` | 1,500 | 51 (3.40%) | 1,449 (96.60%) | Hyperparameter tuning and model selection |
| `test.csv` | 1,500 | 51 (3.40%) | 1,449 (96.60%) | Final unbiased evaluation |
| **Total Cleaned** | 10,000 | 339 (3.39%) | 9,661 (96.61%) | Complete processed dataset |

## Output Files
The pipeline generated the following files in `data/processed/`:
- `predictive_maintenance_clean.csv`
- `train.csv`
- `validation.csv`
- `test.csv`

## Limitations and Safety Note
This dataset reflects synthetic industrial sensor distributions calibrated against real milling machines. Predictions made by this system serve as decision-support alerts and must not replace physical safety shutdowns, scheduled factory inspections, or certified maintenance procedures.
