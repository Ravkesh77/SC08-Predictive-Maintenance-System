# Step 3: Build and Train an ANN

## Objective
Build an Artificial Neural Network (ANN) to predict machine failures using the SC08 Predictive Maintenance System dataset.

## Input Features
- Machine type
- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear

## Target
`Machine failure`: 0 means no failure and 1 means failure.

## Model
A scikit-learn MLPClassifier with hidden layers of 64 and 32 neurons. Numerical features are standardized, and machine type is one-hot encoded.

## Evaluation
The model is evaluated using accuracy, precision, recall, F1-score, and a confusion matrix.

## Outputs
- `models/ann_machine_failure.joblib`
- `data/reports/ann_test_predictions.csv`
- `data/reports/ann_metrics.json`

## How to Run
Execute `python src/train_ann.py` from the project root.

## Limitation
Predictions require validation before use in real maintenance decisions.
