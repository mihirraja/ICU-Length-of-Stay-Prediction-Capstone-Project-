# Saved Model Demos

This folder contains public inference demos for the ICU length-of-stay project.

The original offline models were trained from restricted MIMIC-IV-derived files.
Those source tables are not included in this repository. The committed artifacts
here are compact synthetic-data models with the same kind of feature interface,
so reviewers can verify the saved-model workflow without needing restricted
patient data.

## Included Public Artifacts

- `short_stay_random_forest/` - binary short-stay Random Forest classifier
- `short_stay_logistic_regression/` - binary short-stay Logistic Regression classifier
- `three_class_random_forest/` - three-class LOS Random Forest classifier
- `three_class_logistic_regression/` - three-class LOS Logistic Regression classifier
- `cox_survival/` - Cox proportional hazards survival demo

Each folder contains:

- `model.joblib` - saved model artifact
- `metadata.json` - model type, target, task, and expected feature columns

The shared input file is `../sample_data/fake_icu_los_sample.csv`.

## Run

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python saved_models/run_saved_model_demo.py
```

The script validates the synthetic sample schema, loads every public saved
artifact, and prints:

- longer-than-2-day probabilities for the binary classifiers
- short/medium/long probabilities for the three-class classifiers
- relative discharge hazard and median remaining LOS estimates for the Cox demo

These are public smoke-test artifacts, not clinical validation results.
