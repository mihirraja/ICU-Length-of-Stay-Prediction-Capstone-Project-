from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MODELS_DIR = PROJECT_ROOT / "models"
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from models.run_binary_short_stay_classification import (
    build_preprocessor,
    prepare_modeling_data,
)


OUTPUT_DIR = PROJECT_ROOT / "output" / "classification_results"


def clean_feature_name(name: str) -> str:
    cleaned = name.replace("anchor_age", "age")
    cleaned = cleaned.replace("hours_from_hosp_admit_to_icu", "hours from hospital admit to ICU")
    cleaned = cleaned.replace("ed_length_hours", "ED length")
    cleaned = cleaned.replace("icu_admit_hour", "ICU admit hour")
    cleaned = cleaned.replace("icu_admit_weekend", "weekend ICU admit")
    cleaned = cleaned.replace("first_careunit", "first care unit")
    cleaned = cleaned.replace("admission_type", "admission type")
    cleaned = cleaned.replace("admission_location", "admission location")
    cleaned = cleaned.replace("rad_note_count_24h", "radiology note count")
    cleaned = cleaned.replace("has_rad_24h", "has radiology note")
    cleaned = cleaned.replace("lab_", "")
    cleaned = cleaned.replace("input_", "")
    cleaned = cleaned.replace("kw_", "")
    cleaned = cleaned.replace("_24h", " (24h)")
    cleaned = cleaned.replace("_", " ")
    cleaned = cleaned.replace("resp rate", "respiratory rate")
    cleaned = cleaned.replace("spo2", "SpO2")
    cleaned = cleaned.replace("sbp", "SBP")
    cleaned = cleaned.replace("dbp", "DBP")
    cleaned = cleaned.replace("iv", "IV")
    cleaned = cleaned.replace("icu", "ICU")
    return cleaned


def aggregate_importances(feature_names: list[str], importances: list[float]) -> pd.DataFrame:
    rows = []
    for name, importance in zip(feature_names, importances, strict=False):
        if name.startswith("num__"):
            base = name.replace("num__", "")
        elif name.startswith("cat__"):
            base = name.replace("cat__", "").split("_", 1)[0]
        elif name.startswith("text__"):
            base = "radiology_text"
        else:
            base = name
        rows.append({"feature_group": base, "importance": float(importance)})

    grouped = (
        pd.DataFrame(rows)
        .groupby("feature_group", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
    )
    grouped["feature_label"] = grouped["feature_group"].map(clean_feature_name)
    return grouped


def main() -> None:
    X, y, _groups = prepare_modeling_data()
    text_and_cat = {
        "rad_text_24h",
        "gender",
        "first_careunit",
        "admission_type",
        "admission_location",
        "insurance",
        "language",
        "marital_status",
        "race",
    }
    numeric_features = [c for c in X.columns if c not in text_and_cat]
    categorical_features = [
        c
        for c in [
            "gender",
            "first_careunit",
            "admission_type",
            "admission_location",
            "insurance",
            "language",
            "marital_status",
            "race",
        ]
        if c in X.columns
    ]

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=8,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )
    pipeline.fit(X, y)

    transformed_names = pipeline.named_steps["preprocess"].get_feature_names_out().tolist()
    importances = pipeline.named_steps["model"].feature_importances_.tolist()
    grouped = aggregate_importances(transformed_names, importances)

    top_n = 15
    plot_df = grouped.head(top_n).sort_values("importance", ascending=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "binary_random_forest_feature_importance.csv"
    grouped.to_csv(csv_path, index=False)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(
        data=plot_df,
        x="importance",
        y="feature_label",
        color="#2a9d8f",
        ax=ax,
    )
    ax.set_title("Binary Random Forest: Top Feature Importances", fontsize=15, pad=14)
    ax.set_xlabel("Aggregated Feature Importance", fontsize=12)
    ax.set_ylabel("")
    plt.tight_layout()

    fig_path = OUTPUT_DIR / "binary_random_forest_feature_importance.png"
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)

    print(f"Saved feature importance table to: {csv_path}")
    print(f"Saved feature importance figure to: {fig_path}")


if __name__ == "__main__":
    main()
