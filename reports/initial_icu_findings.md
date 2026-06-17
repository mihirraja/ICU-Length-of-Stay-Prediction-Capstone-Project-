# Updated ICU LOS Findings

## Purpose

This note summarizes the ICU length-of-stay problem definition and the data
signals found during the project. It started as an early discovery memo, but it
has been updated to reflect the current repository: the project now includes
first-24-hour feature builders for vitals, labs, input/output events,
prescriptions, procedures, and radiology features.

The public GitHub repository still does **not** include real patient-level
MIMIC-IV data. Raw and processed clinical tables are restricted and intentionally
excluded from GitHub.

## Modeling Goal

Use information available during the first 24 hours of ICU admission to predict
ICU length-of-stay risk.

The project uses two main framings:

- **Regression / survival framing:** predict remaining ICU LOS after the first
  24 hours.
- **Classification framing:** classify ICU stays into practical LOS buckets,
  including a binary short-stay task and a three-way short/medium/long task.

## Current Data Status

The original early report said the workspace did not yet contain first-24-hour
physiologic measurements. That is no longer accurate for the completed project.

The current pipeline has code to build first-24-hour features from:

- `chartevents`: vital-sign summaries such as heart rate, blood pressure,
  respiratory rate, temperature, and oxygen saturation.
- `labevents`: early lab summaries such as creatinine, BUN, sodium, potassium,
  bicarbonate, glucose, WBC, hemoglobin, platelets, lactate, and anion gap.
- `inputevents`: fluid, medication, nutrition, blood product, drip, bolus, and
  other input summaries.
- `outputevents`: output volume and output-event summaries.
- `prescriptions`: medication order counts, route features, and flags for
  clinically relevant medication groups.
- `procedureevents`: procedure, ventilation, intubation/extubation, line, tube,
  dialysis, imaging, and event-count features.
- `radiology notes`: first-24-hour note counts, modality flags, keyword flags,
  and optional text features.

Relevant scripts:

- `preprocessing/build_chartevents_24h_features.py`
- `preprocessing/clean_chartevents_24h.py`
- `preprocessing/build_labevents_24h_features.py`
- `preprocessing/build_inputevents_24h_features.py`
- `preprocessing/build_outputevents_24h_features.py`
- `preprocessing/build_prescriptions_24h_features.py`
- `preprocessing/build_procedureevents_24h_features.py`
- `models/run_radiology_feature_models.py`

Downstream model scripts merge those feature tables when available:

- `models/run_clinical_features_models.py`
- `models/run_binary_short_stay_classification.py`
- `models/run_los_classification.py`
- `models/run_cox_survival_model.py`

## Cohort and Target Findings

Initial descriptive analysis found:

- Total ICU stays: `94,444`
- Mean ICU LOS: `3.63` days
- Median ICU LOS: `1.97` days
- 25th percentile: `1.10` days
- 75th percentile: `3.86` days
- 90th percentile: `7.92` days
- Maximum ICU LOS: `226.40` days

Interpretation:

- ICU LOS is strongly right-skewed.
- Raw LOS regression is sensitive to long-stay outliers.
- Log-transforming LOS or modeling remaining LOS after 24 hours is more stable
  than directly modeling raw total LOS.

## First-24-Hour Cohort Issue

Initial descriptive analysis also found:

- ICU stays shorter than 24 hours: `19,615`
- Share of all stays shorter than 24 hours: `20.8%`

Interpretation:

- About one in five ICU stays do not have a complete first-24-hour observation
  window.
- For models that require 24 hours of observed data, a clean main cohort is
  `los >= 1` day.
- Very short stays can also be treated as a separate operational task, such as
  early discharge / short-stay classification.

## Descriptive Findings

### LOS by Age Bin

| Age bin | N | Mean LOS | Median LOS |
|---|---:|---:|---:|
| `<40` | 9,573 | 3.52 | 1.73 |
| `40-59` | 26,214 | 3.75 | 1.93 |
| `60-74` | 33,020 | 3.75 | 2.02 |
| `75+` | 25,637 | 3.39 | 2.00 |

Takeaway:

- Age alone appears only modestly related to ICU LOS.
- Older groups have slightly higher median LOS than the youngest group, but the
  effect is not dramatic.

### LOS by Gender

| Gender | N | Mean LOS | Median LOS |
|---|---:|---:|---:|
| `F` | 41,577 | 3.51 | 1.94 |
| `M` | 52,867 | 3.72 | 1.98 |

Takeaway:

- Gender differences are small.
- Gender is unlikely to be a strong standalone predictor.

### LOS by First Care Unit

| Care unit | N | Mean LOS | Median LOS |
|---|---:|---:|---:|
| MICU | 20,699 | 3.76 | 1.91 |
| MICU/SICU | 15,447 | 3.09 | 1.79 |
| CVICU | 14,769 | 3.32 | 1.99 |
| SICU | 13,008 | 3.90 | 1.98 |
| CCU | 10,771 | 3.09 | 2.01 |
| TSICU | 10,474 | 3.64 | 1.88 |
| Neuro Intermediate | 5,776 | 5.02 | 3.00 |
| Neuro SICU | 1,750 | 4.48 | 2.24 |
| Neuro Stepdown | 1,421 | 4.07 | 2.20 |

Takeaway:

- First care unit is more informative than age or gender.
- Neuro-oriented units have noticeably longer typical stays.

### LOS by Admission Type

| Admission type | N | Mean LOS | Median LOS |
|---|---:|---:|---:|
| `EW EMER.` | 48,349 | 3.54 | 1.91 |
| `URGENT` | 15,374 | 4.41 | 2.27 |
| `OBSERVATION ADMIT` | 14,031 | 3.82 | 2.11 |
| `SURGICAL SAME DAY ADMISSION` | 9,544 | 2.87 | 1.68 |
| `DIRECT EMER.` | 3,316 | 3.77 | 2.08 |
| `ELECTIVE` | 3,027 | 3.09 | 1.96 |
| `EU OBSERVATION` | 541 | 1.09 | 0.72 |
| `DIRECT OBSERVATION` | 237 | 0.92 | 0.79 |

Takeaway:

- Admission pathway appears predictive.
- `URGENT` admissions stay longer on average than `EW EMER.` and same-day
  surgical admissions.
- Observation categories are much shorter and may need separate handling.

### LOS by Admission Location

| Admission location | N | Mean LOS | Median LOS |
|---|---:|---:|---:|
| `EMERGENCY ROOM` | 37,501 | 3.30 | 1.85 |
| `TRANSFER FROM HOSPITAL` | 24,304 | 4.48 | 2.37 |
| `PHYSICIAN REFERRAL` | 23,697 | 3.28 | 1.86 |
| `WALK-IN/SELF REFERRAL` | 4,477 | 3.84 | 1.94 |
| `TRANSFER FROM SKILLED NURSING FACILITY` | 1,516 | 3.66 | 2.13 |
| `CLINIC REFERRAL` | 1,186 | 3.59 | 2.00 |
| `PROCEDURE SITE` | 1,025 | 2.75 | 1.73 |

Takeaway:

- Transfer patients have meaningfully longer stays than ER or
  physician-referred patients.
- Admission source is useful in a baseline model.

## Radiology Note Signal

Initial analysis found:

- Distinct ICU stays with any radiology note linked during the ICU stay:
  `52,654`
- Distinct ICU stays with at least one radiology note in the first 24 hours:
  `52,653`
- Share of all ICU stays with a first-24-hour radiology note: `55.8%`
- Share among stays with `los >= 1` day: `60.9%`

### First-24-Hour Radiology Note Coverage by LOS Bucket

| LOS bucket | N | With first-24h radiology note | Percent |
|---|---:|---:|---:|
| `<1d` | 19,615 | 7,053 | 36.0% |
| `1-3d` | 43,651 | 25,206 | 57.7% |
| `3-7d` | 20,007 | 12,958 | 64.8% |
| `7d+` | 11,171 | 7,436 | 66.6% |

Takeaway:

- Radiology note availability increases with longer ICU stay.
- Note presence may reflect care intensity, so missingness and availability
  should be modeled carefully.
- The completed pipeline includes both structured radiology flags and optional
  TF-IDF text features.

## Updated Recommended Problem Definition

Recommended main task:

- Cohort: ICU stays with `los >= 1` day when using a full first-24-hour feature
  window.
- Regression/survival target: remaining ICU LOS after 24 hours, defined as
  `los - 1`.
- Classification target: binary short stay versus longer stay, with optional
  three-way LOS classes for a harder secondary task.

Reasoning:

- If the model uses the first 24 hours of data, the first ICU day has already
  elapsed.
- Predicting remaining LOS better matches the decision point at hour 24.
- The binary task is easier to explain and more operationally interpretable for
  a public demo or recruiter review.

## Current Project Status

The project is no longer only a static-feature baseline. The current offline
pipeline supports multimodal first-24-hour modeling with:

- demographics and admission context
- charted vitals
- labs
- input/output events
- prescriptions and medication groups
- procedure events
- first-day radiology-note counts, flags, keywords, and text

The public GitHub demo uses a compact synthetic-data model so reviewers can run
the repository without restricted MIMIC-IV access. The offline model scripts and
figures document the fuller clinical feature work.

## Remaining Limitations

- Raw MIMIC-IV tables and generated patient-level processed tables are not
  shareable in the public repo.
- The public saved models are synthetic-data smoke-test artifacts, not
  clinically validated models.
- The completed offline pipeline now includes first-24-hour vitals, labs,
  input/output events, prescriptions, procedure events, and radiology-note
  features. These should no longer be described as missing.
- Detailed ventilator settings or respiratory support measurements beyond
  procedure/intubation/ventilation flags are not fully represented in the
  current public pipeline.
- Diagnosis codes and formal severity scores are not emphasized in the current
  public pipeline. They would be useful additions if available at or before the
  first 24-hour decision point.
- External validation on another ICU dataset would be needed before making any
  clinical claims.
