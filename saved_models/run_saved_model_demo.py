from pathlib import Path
import json

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "saved_models" / "short_stay_random_forest"
SAMPLE_PATH = PROJECT_ROOT / "sample_data" / "fake_icu_los_sample.csv"


COLUMN_MAP = {
    "anchor_age": "age",
    "heart_rate_mean": "heart_rate_mean_24h",
    "lab_lactate_max": "lactate_max_24h",
    "rad_note_count_24h": "radiology_note_count_24h",
    "proc_event_count_24h": "procedure_event_count_24h",
    "rx_order_count_24h": "medication_count_24h",
}


def load_metadata() -> dict:
    with (MODEL_DIR / "metadata.json").open() as f:
        return json.load(f)


def prepare_features(sample: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    required_source_columns = set(COLUMN_MAP) | {
        "demo_stay_id",
        "first_careunit",
        "admission_type",
        "sbp_mean",
        "dbp_mean",
        "kw_hemorrhage_24h",
        "kw_edema_24h",
        "kw_pneumonia_24h",
        "kw_effusion_24h",
        "kw_stroke_24h",
        "kw_fracture_24h",
        "kw_intubation_24h",
    }
    missing_source = sorted(required_source_columns - set(sample.columns))
    if missing_source:
        raise ValueError(f"Sample CSV is missing required source columns: {missing_source}")

    features = sample.rename(columns=COLUMN_MAP).copy()
    features["map_mean_24h"] = (features["sbp_mean"] + 2 * features["dbp_mean"]) / 3
    features["abnormal_radiology_flag_24h"] = (
        features[
            [
                "kw_hemorrhage_24h",
                "kw_edema_24h",
                "kw_pneumonia_24h",
                "kw_effusion_24h",
                "kw_stroke_24h",
                "kw_fracture_24h",
                "kw_intubation_24h",
            ]
        ]
        .fillna(0)
        .max(axis=1)
    )
    missing_features = sorted(set(feature_columns) - set(features.columns))
    if missing_features:
        raise ValueError(f"Feature engineering did not create: {missing_features}")

    return features[feature_columns]


def main() -> None:
    metadata = load_metadata()
    model = joblib.load(PROJECT_ROOT / metadata["model_file"])
    sample = pd.read_csv(SAMPLE_PATH)
    X = prepare_features(sample, metadata["feature_columns"])

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    results = sample[["demo_stay_id", "anchor_age", "first_careunit", "admission_type"]].copy()
    results["predicted_longer_than_2d"] = predictions.astype(int)
    results["probability_longer_than_2d"] = probabilities[:, 1].round(3)

    print("Saved model demo: ICU short-stay Random Forest")
    print(f"Rows scored: {len(results)}")
    print(f"Features validated: {len(metadata['feature_columns'])}")
    print()
    print(results.head(10).to_string(index=False))
    print()
    print("Smoke test passed: schema validation and saved-model inference completed.")
    print("Note: this public demo uses synthetic examples, not clinical validation metrics.")


if __name__ == "__main__":
    main()
