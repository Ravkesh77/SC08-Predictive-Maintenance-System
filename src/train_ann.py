
import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "data" / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df = pd.read_csv(DATA_DIR / "validation.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")

target = "Machine failure"

features = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

X_train = train_df[features]
y_train = train_df[target]

X_val = val_df[features]
y_val = val_df[target]

X_test = test_df[features]
y_test = test_df[target]

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), features[1:]),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            ["Type"]
        )
    ]
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("ann", MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=300,
        early_stopping=True,
        random_state=42
    ))
])

model.fit(X_train, y_train)

val_predictions = model.predict(X_val)
test_predictions = model.predict(X_test)

metrics = {
    "validation_accuracy": float(
        accuracy_score(y_val, val_predictions)
    ),
    "accuracy": float(
        accuracy_score(y_test, test_predictions)
    ),
    "precision": float(
        precision_score(y_test, test_predictions, zero_division=0)
    ),
    "recall": float(
        recall_score(y_test, test_predictions, zero_division=0)
    ),
    "f1_score": float(
        f1_score(y_test, test_predictions, zero_division=0)
    )
}

joblib.dump(
    model,
    MODEL_DIR / "ann_machine_failure.joblib"
)

pd.DataFrame({
    "actual": y_test.to_numpy(),
    "predicted": test_predictions
}).to_csv(
    REPORT_DIR / "ann_test_predictions.csv",
    index=False
)

with open(REPORT_DIR / "ann_metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("ANN training completed successfully.")
print("\nEvaluation Metrics:")

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    test_predictions,
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_predictions))

print("\nSaved model:", MODEL_DIR / "ann_machine_failure.joblib")
print("Saved metrics:", REPORT_DIR / "ann_metrics.json")
print("Saved predictions:", REPORT_DIR / "ann_test_predictions.csv")
