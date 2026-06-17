from pathlib import Path
import json

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
SAMPLE_PATH = PROJECT_ROOT / "sample_data" / "fake_icu_los_sample.csv"

MODEL_DIRS = [
    "short_stay_random_forest",
    "short_stay_logistic_regression",
    "three_class_random_forest",
    "three_class_logistic_regression",
    "cox_survival",
]

COLUMN_MAP = {
    "anchor_age": "age",
    "heart_rate_mean": "heart_rate_mean_24h",
    "lab_lactate_max": "lactate_max_24h",
    "rad_note_count_24h": "radiology_note_count_24h",
    "proc_event_count_24h": "procedure_event_count_24h",
    "rx_order_count_24h": "medication_count_24h",
}

RADIOLOGY_FLAG_COLUMNS = [
    "kw_hemorrhage_24h",
    "kw_edema_24h",
    "kw_pneumonia_24h",
    "kw_effusion_24h",
    "kw_stroke_24h",
    "kw_fracture_24h",
    "kw_intubation_24h",
]


def load_metadata(model_dir: str) -> dict:
    with (SAVED_MODELS_DIR / model_dir / "metadata.json").open() as f:
        return json.load(f)


def prepare_features(sample: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    required_source_columns = set(COLUMN_MAP) | {
        "demo_stay_id",
        "first_careunit",
        "admission_type",
        "sbp_mean",
        "dbp_mean",
        *RADIOLOGY_FLAG_COLUMNS,
    }
    missing_source = sorted(required_source_columns - set(sample.columns))
    if missing_source:
        raise ValueError(f"Sample CSV is missing required source columns: {missing_source}")

    features = sample.rename(columns=COLUMN_MAP).copy()
    features["map_mean_24h"] = (features["sbp_mean"] + 2 * features["dbp_mean"]) / 3
    features["abnormal_radiology_flag_24h"] = (
        features[RADIOLOGY_FLAG_COLUMNS].fillna(0).max(axis=1)
    )

    missing_features = sorted(set(feature_columns) - set(features.columns))
    if missing_features:
        raise ValueError(f"Feature engineering did not create: {missing_features}")

    return features[feature_columns]


def make_cox_frame(X: pd.DataFrame, cox_feature_columns: list[str]) -> pd.DataFrame:
    cox_X = pd.get_dummies(X, columns=["first_careunit", "admission_type"], drop_first=False)
    return cox_X.reindex(columns=cox_feature_columns, fill_value=0)


def summarize_classifier(model, metadata: dict, X: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    classes = list(model.classes_)

    results = sample[["demo_stay_id", "anchor_age", "first_careunit", "admission_type"]].copy()
    results["prediction"] = predictions

    if metadata["task"] == "classification_binary":
        positive_class = 1 if 1 in classes else classes[-1]
        positive_idx = classes.index(positive_class)
        results["probability_longer_than_2d"] = probabilities[:, positive_idx].round(3)
    else:
        for idx, class_name in enumerate(classes):
            results[f"probability_{class_name}"] = probabilities[:, idx].round(3)

    return results


def summarize_survival(model, metadata: dict, X: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    cox_X = make_cox_frame(X, metadata["cox_feature_columns"])
    partial_hazard = model.predict_partial_hazard(cox_X)
    median_survival = model.predict_median(cox_X)

    results = sample[["demo_stay_id", "anchor_age", "first_careunit", "admission_type"]].copy()
    results["relative_discharge_hazard"] = partial_hazard.to_numpy().round(3)
    results["predicted_median_remaining_los_days"] = (
        pd.Series(median_survival).replace([float("inf")], pd.NA).round(2).to_numpy()
    )
    return results


def main() -> None:
    sample = pd.read_csv(SAMPLE_PATH)
    print("Saved model demo: ICU length-of-stay public artifacts")
    print(f"Rows scored: {len(sample)}")
    print()

    for model_dir in MODEL_DIRS:
        metadata = load_metadata(model_dir)
        model = joblib.load(PROJECT_ROOT / metadata["model_file"])
        X = prepare_features(sample, metadata["feature_columns"])

        print(f"== {model_dir.replace('_', ' ').title()} ==")
        print(metadata["target_definition"])
        print(f"Features validated: {len(metadata['feature_columns'])}")

        if metadata["task"] == "survival":
            results = summarize_survival(model, metadata, X, sample)
        else:
            results = summarize_classifier(model, metadata, X, sample)

        print(results.head(10).to_string(index=False))
        print()

    print("Smoke test passed: schema validation and saved-model inference completed.")
    print("Note: these public artifacts use synthetic examples, not clinical validation metrics.")


if __name__ == "__main__":
    main()
