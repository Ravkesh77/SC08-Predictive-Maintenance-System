# Data Documentation

## Project Purpose
Predict water potability based on physical and chemical water quality measurements.

## Primary Labelled Dataset
- **Dataset:** Water Potability
- **Source:** <URL to your dataset or source paper>
- **Licence:** MIT (or your dataset's license)
- **Downloaded File:** water_potability_balanced.csv
- **Source Records:** 2,556
- **Target:** Potability (0 = Not potable, 1 = Potable)

## Input Fields
- `ph`: pH value (0–14)
- `Hardness`: Capacity of water to precipitate soap (mg/L)
- `Solids`: Total dissolved solids (ppm)
- `Chloramines`: Chloramines concentration (ppm)
- `Sulfate`: Dissolved sulfate (mg/L)
- `Conductivity`: Electrical conductivity (μS/cm)
- `Organic_carbon`: Organic carbon content (ppm)
- `Trihalomethanes`: Trihalomethanes amount (μg/L)
- `Turbidity`: Measure of water clarity (NTU)

## Step 2 Plan
Clean missing values via median imputation, drop duplicates, validate value ranges, and create reproducible stratified train/validation/test splits.
