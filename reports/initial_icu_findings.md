# Initial ICU LOS Findings

## Goal

Use only information available within the first 24 hours of ICU admission to predict how long a patient will stay in the ICU.

## Files In The Workspace

- `icu_stays.csv`
  - One row per ICU stay.
  - Includes `stay_id`, `subject_id`, `hadm_id`, `intime`, `outtime`, `los`, `first_careunit`, `admission_type`, and `admission_location`.
  - This is the main target table because `los` is the ICU length of stay in days.
- `icu_patients.csv`
  - Patient-level demographics available at baseline.
  - Includes `subject_id`, `gender`, and `anchor_age`.
- `icu_radiology_notes.csv`
  - Time-stamped radiology note text with linked `stay_id`.
  - Can contribute first-24-hour text features for the subset of stays with imaging reports early in the admission.
- `baseline_model_data.parquet`
  - A model-ready baseline table with 94,444 rows and 41 columns.
  - Contains only static baseline features: age, gender, first care unit, admission type, admission location, and the target `log_los`.
  - It does **not** contain first-24-hour vitals, labs, or procedures.
- `chartevents_reference.parquet`
  - A lookup table with 3,055 chart item definitions.
  - Contains `itemid`, `label`, `abbreviation`, `category`, `unitname`, and `param_type`.
  - This is metadata only, not actual charted patient values.

## What We Can Say Right Now

### 1. The target is available and the cohort is large enough

- Total ICU stays: `94,444`
- Mean ICU LOS: `3.63` days
- Median ICU LOS: `1.97` days
- 25th percentile: `1.10` days
- 75th percentile: `3.86` days
- 90th percentile: `7.92` days
- Maximum ICU LOS: `226.40` days

Interpretation:

- ICU LOS is strongly right-skewed.
- A regression target like raw LOS will be dominated by long-stay outliers.
- Using `log_los` is a sensible first baseline.

### 2. A first-24-hour model has an immediate cohort-definition issue

- ICU stays shorter than 24 hours: `19,615`
- Share of all stays shorter than 24 hours: `20.8%`

Interpretation:

- About one in five stays never has a full 24-hour observation window.
- If the model is defined as "use the first 24 hours," these stays should usually be excluded from training, or treated as a separate task.
- For the main modeling cohort, a clean first pass is to restrict to stays with `los >= 1` day.

### 3. The current workspace does not yet contain actual first-24-hour physiologic measurements

- We have baseline administrative features.
- We have radiology note text.
- We have chart item definitions.
- We do **not** yet have the underlying time-stamped chart events, vitals, labs, medications, or procedures needed to fully represent the first 24 hours.

Interpretation:

- The current files support baseline modeling and text augmentation.
- They do not yet support a true first-24-hour multimodal ICU LOS model.

## Initial Descriptive Findings

### LOS by age bin

| Age bin | N | Mean LOS | Median LOS |
|---|---:|---:|---:|
| `<40` | 9,573 | 3.52 | 1.73 |
| `40-59` | 26,214 | 3.75 | 1.93 |
| `60-74` | 33,020 | 3.75 | 2.02 |
| `75+` | 25,637 | 3.39 | 2.00 |

Takeaway:

- Age alone appears only modestly related to ICU LOS.
- Older groups have slightly higher median LOS than the youngest group, but the effect is not dramatic.

### LOS by gender

| Gender | N | Mean LOS | Median LOS |
|---|---:|---:|---:|
| `F` | 41,577 | 3.51 | 1.94 |
| `M` | 52,867 | 3.72 | 1.98 |

Takeaway:

- Gender differences are small.
- This is unlikely to be a strong standalone predictor.

### LOS by first care unit

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

- First care unit looks more informative than age or gender.
- Neuro-oriented units have noticeably longer typical stays.

### LOS by admission type

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
- `URGENT` admissions stay longer on average than `EW EMER.` and same-day surgical admissions.
- Observation categories are much shorter and could be handled differently or excluded depending on the clinical definition of the task.

### LOS by admission location

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

- Transfer patients are meaningfully longer stays than ER or physician-referred patients.
- Admission source is likely useful in a baseline model.

## Early Signal From Radiology Notes

- Distinct ICU stays with any radiology note linked during the ICU stay: `52,654`
- Distinct ICU stays with at least one radiology note in the first 24 hours: `52,653`
- Share of all ICU stays with a first-24-hour radiology note: `55.8%`
- Share among stays with `los >= 1` day: `60.9%`

### First-24-hour radiology note coverage by LOS bucket

| LOS bucket | N | With first-24h radiology note | Percent |
|---|---:|---:|---:|
| `<1d` | 19,615 | 7,053 | 36.0% |
| `1-3d` | 43,651 | 25,206 | 57.7% |
| `3-7d` | 20,007 | 12,958 | 64.8% |
| `7d+` | 11,171 | 7,436 | 66.6% |

Takeaway:

- Radiology note availability increases with longer ICU stay.
- This may be a real signal, but it also means note presence itself could encode care intensity.
- If text features are used, note availability should be treated carefully to avoid bias from missingness patterns.

## Recommended Problem Definition

For a first-pass model, use:

- Cohort: ICU stays with `los >= 1` day
- Target option A: total ICU LOS in days
- Target option B: remaining ICU LOS after 24 hours, defined as `los - 1`

Preferred option:

- `remaining ICU LOS after 24 hours`

Reason:

- If the model uses the first 24 hours of data, then the first day has already elapsed.
- Predicting remaining LOS is more aligned with the decision point at hour 24 and avoids baking observed time into the target.

## Best Baseline Model We Can Build With Current Files

Right now, the strongest reproducible baseline available from the workspace is:

- Inputs:
  - age
  - gender
  - first care unit
  - admission type
  - admission location
  - optional first-24-hour radiology-note presence or text-derived features
- Target:
  - `log_los` or `log(remaining_los + small_constant)`

This would be a useful starting benchmark, but it is not yet a full first-24-hour ICU severity model.

## What Data Is Still Needed

To answer the original problem well, the missing high-value tables are:

- time-stamped ICU chart events or vitals
- early labs
- ventilator settings or respiratory support data
- vasoactive medication use
- procedures and interventions in the first 24 hours
- diagnoses or severity proxies available at or before hour 24

## Suggested Next Step

1. Restrict the cohort to ICU stays with `los >= 1`.
2. Decide whether the target should be total LOS or remaining LOS after hour 24.
3. Build a baseline model from the existing static features.
4. Add radiology-note coverage and text features as an optional second baseline.
5. Bring in actual first-24-hour charted values so we can build the real clinical feature set.
