from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_FILE = PROJECT_ROOT / "data" / "processed" / "predictive_maintenance_clean.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "generated"
OUTPUT_FILE = OUTPUT_DIR / "generated_predictive_maintenance_test_scenarios.csv"
REPORT_FILE = PROJECT_ROOT / "data" / "reports" / "generated_scenarios_report.md"

SEED = 42
FEATURE_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]
TARGET_COLUMN = "Machine failure"

def generate_scenarios(n_scenarios=256) -> pd.DataFrame:
    np.random.seed(SEED)
    clean_df = pd.read_csv(CLEAN_FILE)
    sampled = clean_df.sample(n=n_scenarios, replace=True, random_state=SEED).copy()
    
    # Apply small 1% IQR Gaussian jitter to continuous physical features
    for col in FEATURE_COLUMNS:
        iqr = clean_df[col].quantile(0.75) - clean_df[col].quantile(0.25)
        jitter = np.random.normal(0, 0.01 * max(iqr, 1e-4), size=n_scenarios)
        sampled[col] = (sampled[col] + jitter).clip(lower=0)
        
    sampled["Rotational speed [rpm]"] = sampled["Rotational speed [rpm]"].round().astype(int)
    sampled["Tool wear [min]"] = sampled["Tool wear [min]"].round().astype(int)
    
    sampled.insert(0, "Scenario_ID", [f"GEN-PM-{i+1:03d}" for i in range(n_scenarios)])
    sampled.rename(columns={TARGET_COLUMN: "Expected_Machine_Failure"}, inplace=True)
    return sampled

def validate_scenarios(scenarios_df: pd.DataFrame):
    if not scenarios_df["Scenario_ID"].is_unique:
        raise ValueError("Generated scenario IDs must be unique.")
    if scenarios_df[FEATURE_COLUMNS].isna().sum().sum() > 0:
        raise ValueError("Generated scenarios contain missing values.")
    if (scenarios_df[FEATURE_COLUMNS] < 0).any().any():
        raise ValueError("Generated scenarios contain negative physical measurements.")
    if not scenarios_df["Expected_Machine_Failure"].isin([0, 1]).all():
        raise ValueError("Generated scenarios contain invalid failure labels.")

def run_generator():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scenarios = generate_scenarios(256)
    validate_scenarios(scenarios)
    scenarios.to_csv(OUTPUT_FILE, index=False)
    
    print("=== SC08 GENERATED TESTING SCENARIOS COMPLETED ===")
    print(f"Real cleaned records: 10000")
    print(f"Generated testing scenarios: {len(scenarios)}")
    print(f"Generation seed: {SEED}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Scenario table shape: {scenarios.shape}")

if __name__ == "__main__":
    run_generator()
