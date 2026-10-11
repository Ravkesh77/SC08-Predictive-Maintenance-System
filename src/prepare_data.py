from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

FEATURE_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]
TARGET_COLUMN = "Machine failure"
REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]

SEED = 42
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

def clean_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    cleaned = dataframe.copy()
    # Median imputation for numeric features (if any missing values)
    cleaned[FEATURE_COLUMNS] = cleaned[FEATURE_COLUMNS].fillna(cleaned[FEATURE_COLUMNS].median())
    # Remove duplicates
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    return cleaned

def validate_data(dataframe: pd.DataFrame):
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in dataframe.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    if dataframe[REQUIRED_COLUMNS].isna().sum().sum() > 0:
        raise ValueError("Missing values remain after cleaning.")
    # Domain validation checks
    if (dataframe["Air temperature [K]"] < 250).any() or (dataframe["Air temperature [K]"] > 350).any():
        raise ValueError("Air temperature outside realistic physical bounds [250K, 350K].")
    if (dataframe["Process temperature [K]"] < 250).any() or (dataframe["Process temperature [K]"] > 350).any():
        raise ValueError("Process temperature outside realistic physical bounds [250K, 350K].")
    if (dataframe["Rotational speed [rpm]"] < 0).any():
        raise ValueError("Rotational speed cannot be negative.")
    if (dataframe["Torque [Nm]"] < 0).any():
        raise ValueError("Torque cannot be negative.")
    if (dataframe["Tool wear [min]"] < 0).any():
        raise ValueError("Tool wear cannot be negative.")
    if not dataframe[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError(f"Invalid target labels detected in {TARGET_COLUMN}.")
    if dataframe.duplicated().sum() > 0:
        raise ValueError("Duplicate rows remain.")

def create_stratified_splits(dataframe: pd.DataFrame):
    train_parts, val_parts, test_parts = [], [], []
    for _, group in dataframe.groupby(TARGET_COLUMN, sort=True):
        shuffled = group.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        total = len(shuffled)
        test_n = round(total * TEST_RATIO)
        val_n = round(total * VALIDATION_RATIO)
        test_parts.append(shuffled.iloc[:test_n])
        val_parts.append(shuffled.iloc[test_n : test_n + val_n])
        train_parts.append(shuffled.iloc[test_n + val_n :])
        
    train_data = pd.concat(train_parts).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    val_data = pd.concat(val_parts).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    test_data = pd.concat(test_parts).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return train_data, val_data, test_data

def run_pipeline():
    raw_data = pd.read_csv(RAW_FILE)
    cleaned = clean_data(raw_data)
    validate_data(cleaned)
    train_df, val_df, test_df = create_stratified_splits(cleaned)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(PROCESSED_DIR / "predictive_maintenance_clean.csv", index=False)
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "validation.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

    print("=== SC08 STEP 2 DATA PIPELINE COMPLETED ===")
    print(f"Raw rows: {len(raw_data)}")
    print(f"Cleaned rows: {len(cleaned)}")
    print(f"Training rows: {len(train_df)}")
    print(f"Validation rows: {len(val_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Random seed: {SEED}")
    print(f"Output location: {PROCESSED_DIR}")

if __name__ == "__main__":
    run_pipeline()
