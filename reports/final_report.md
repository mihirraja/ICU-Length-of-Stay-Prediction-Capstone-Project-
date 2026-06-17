# Final Report: ICU Length-of-Stay Prediction

## Executive Summary

This project investigates whether information available during the first 24
hours of an ICU admission can be used to estimate length-of-stay risk. The work
uses MIMIC-IV-derived ICU data in the offline pipeline and combines admission
context, vital signs, labs, input/output events, prescriptions, procedure
events, and radiology-note features.

The public GitHub repository is designed for review without exposing restricted
patient data. It includes the preprocessing and modeling code, final report
figures, and a synthetic saved-model demo that verifies the inference workflow
from a fresh clone.

## Problem Definition

ICU length of stay is operationally important because long stays affect bed
availability, staffing, care coordination, and discharge planning. The main
question is:

> Can early clinical signals from the first 24 hours of an ICU stay help
> identify patients likely to have short versus longer ICU stays?

The project considers two related modeling setups:

- **Binary classification:** short ICU stay versus longer ICU stay.
- **Three-way classification:** short, medium, and long ICU stay classes.

Regression and survival-style modeling were also explored using remaining ICU
LOS after the first 24 hours as a target.

## Data

The offline project uses restricted MIMIC-IV-derived tables. These patient-level
tables are not committed to GitHub. The public repository includes only
synthetic sample rows for demonstrating model loading and inference.

The offline feature pipeline uses:

- ICU stay metadata and admission context
- patient demographics
- first-24-hour charted vital signs
- first-24-hour lab summaries
- first-24-hour input and output events
- prescription and medication-group features
- procedure and intervention event summaries
- first-24-hour radiology-note counts, keyword flags, modality flags, and text

## Feature Engineering

The project builds patient-stay-level feature tables from time-stamped ICU
events restricted to the first 24 hours after ICU admission. This keeps the
feature window aligned with an early decision-support use case.

Key preprocessing scripts:

- `preprocessing/build_chartevents_24h_features.py`
- `preprocessing/build_labevents_24h_features.py`
- `preprocessing/build_inputevents_24h_features.py`
- `preprocessing/build_outputevents_24h_features.py`
- `preprocessing/build_prescriptions_24h_features.py`
- `preprocessing/build_procedureevents_24h_features.py`
- `models/run_radiology_feature_models.py`

The modeling scripts merge these feature groups when the processed files are
available locally.

## Modeling Approach

The classification pipeline compares baseline and tree-based models, including
Logistic Regression and Random Forest classifiers. The project also includes
Cox-style survival modeling for a time-to-event framing.

Important modeling choices:

- Features are restricted to early ICU information.
- Offline experiments use patient-level grouped splitting by `subject_id` to
  reduce leakage across train/test splits.
- Missingness indicators are added for many clinical feature groups because
  the absence of a measurement can itself reflect care patterns.
- Radiology features are treated carefully because note availability can encode
  care intensity as well as clinical signal.

## Results Summary

The final figures in `report_plots/` summarize the offline experiments:

- `binary_model_comparison.png` compares model performance on the binary task.
- `binary_roc_curve.png` shows binary short-stay discrimination.
- `confusion_matrix_comparison.png` compares classification errors.
- `top_feature_importances.png` highlights influential Random Forest features.
- `binary_vs_three_way_target_setup.png` explains the target framing.
- `los_class_mix_by_icu_unit.png` shows how LOS class mix varies by ICU unit.

The main project takeaway is that richer first-24-hour clinical features provide
a stronger and more clinically meaningful modeling setup than static admission
features alone. ICU unit, admission pathway, early measurement burden, labs,
procedures, medication patterns, and radiology-note signals all contribute to
the prediction problem.

## Public Reproducibility Demo

The public repository includes a compact saved-model demo under `saved_models/`.
It uses synthetic ICU-style rows because real MIMIC-IV patient rows cannot be
redistributed.

Run from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python saved_models/run_saved_model_demo.py
```

The script validates the input schema, loads a saved scikit-learn pipeline, and
prints predicted probabilities for longer-than-2-day ICU stays.

The public demo is a smoke test for engineering reproducibility. It is not a
clinical validation result.

## Limitations

- The public repository cannot include raw MIMIC-IV data or generated
  patient-level processed tables.
- The public saved model is trained on synthetic ICU-style data so reviewers can
  test the inference workflow.
- Detailed ventilator settings are only partially represented through procedure
  and intervention flags.
- Diagnosis codes and formal severity scores are not emphasized in the current
  public pipeline.
- External validation would be required before making clinical deployment
  claims.

## Conclusion

This project demonstrates an end-to-end applied machine learning workflow for a
clinically relevant ICU operations problem. The work covers cohort definition,
privacy-aware data handling, first-24-hour feature engineering, leakage-aware
evaluation, model comparison, result visualization, and a public saved-model
demo that reviewers can run without restricted data access.
