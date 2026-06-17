# Saved Model Demo

This folder contains a small public inference demo for the ICU length-of-stay project.

The original models were trained offline from restricted MIMIC-IV-derived files. Those source tables are not included in this repository. The committed artifact here is a compact Random Forest pipeline trained on synthetic ICU-style data with the same kind of feature interface, so reviewers can verify the saved-model workflow without needing restricted patient data.

## Contents

- `short_stay_random_forest/model.joblib` - saved scikit-learn pipeline
- `short_stay_random_forest/metadata.json` - model type, target, and expected feature columns
- `run_saved_model_demo.py` - runnable inference script
- `../sample_data/fake_icu_los_sample.csv` - synthetic input rows

## Run

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python saved_models/run_saved_model_demo.py
```

The script loads the saved model, maps the synthetic sample into the model schema, and prints predicted probabilities for whether each ICU stay is longer than 2 days.
