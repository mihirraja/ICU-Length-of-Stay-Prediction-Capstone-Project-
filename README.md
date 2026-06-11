# Prediction of ICU Length of Stay Using Early Clinical Data

This repository contains the code and lightweight reproducible demo for our STATS
170B capstone project. The project predicts whether an ICU stay will be short
or prolonged using patient information available during the first 24 hours of
ICU admission.

The full project used MIMIC-IV data from PhysioNet. MIMIC-IV is access
restricted, and the raw/processed data files are too large to upload to GitHub.
For that reason, this repository includes a small synthetic sample dataset that
has the same general structure as the first-24-hour modeling table. The sample
is only intended to demonstrate how the modeling pipeline runs.

## Files and Folders

- `project.ipynb`
  - Main runnable demonstration notebook. It loads the sample ICU-style data,
    trains fast short-vs-longer stay classifiers, evaluates them, and shows
    example predictions.
- `project.html`
  - HTML export of `project.ipynb` with all cell outputs already rendered.
- `data/sample/fake_icu_los_sample.csv`
  - Small hand-created synthetic dataset used by `project.ipynb`. It is not
    real patient data.
- `models/`
  - Python scripts used for the full project model experiments, including
    regression, classification, radiology-feature models, and Cox survival
    modeling.
- `preprocessing/`
  - Scripts for constructing first-24-hour features from MIMIC-IV tables such
    as chart events, lab events, input/output events, prescriptions, and
    procedure events.
- `visualizations/`
  - Scripts used to generate plots for model comparison, classification
    diagnostics, Cox model summaries, and poster/report figures.
- `output/`
  - Generated project outputs from the full local analysis. This folder is
    ignored for GitHub because some derived files are large. The final plots
    needed for the report are copied into `report_plots/`.
- `report_plots/`
  - Clean copies of selected final-report/poster plots.
- `reports/`
  - Notes and draft writeups.
- `.gitignore`
  - Excludes the local virtual environment, IDE files, caches, and restricted
    raw/processed MIMIC-IV data.

## How to Run the Demo Notebook

1. Create or activate a Python environment with the required packages.

   If you are using the local environment already present on the project
   machine:

   ```bash
   source capstone/bin/activate
   ```

   Otherwise install the dependencies listed in `requirements.txt`.

2. Run the notebook:

   ```bash
   jupyter notebook project.ipynb
   ```

   Then choose **Run All**.

   From the command line, you can also run:

   ```bash
   jupyter nbconvert --to notebook --execute project.ipynb --output project.ipynb --ExecutePreprocessor.timeout=60
   ```

3. Regenerate the HTML export:

   ```bash
   jupyter nbconvert --to html project.ipynb --output project.html
   ```

The notebook should run in under 1 minute. On the current project environment,
it runs in a few seconds.

## Full Pipeline Notes

The full MIMIC-IV pipeline requires access to restricted PhysioNet files. To
reproduce the full analysis, obtain MIMIC-IV access through PhysioNet, download
the required tables, and place them under `data/raw/` using the filenames
expected by the preprocessing scripts. The main full-pipeline scripts are:

```bash
python preprocessing/build_chartevents_24h_features.py
python preprocessing/build_labevents_24h_features.py
python preprocessing/build_inputevents_24h_features.py
python preprocessing/build_outputevents_24h_features.py
python preprocessing/build_prescriptions_24h_features.py
python preprocessing/build_procedureevents_24h_features.py
python models/run_los_classification.py
python models/run_binary_short_stay_classification.py
python models/run_cox_survival_model.py
python visualizations/make_model_comparison_visuals.py
```

Because the raw data is restricted and large, the GitHub-submission notebook
does not run the full pipeline. Instead, it demonstrates the same type of model
input, training, evaluation, and prediction workflow on a small synthetic
sample.
