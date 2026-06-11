from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "output"
FEATURE_DIR = OUTPUT_DIR / "features"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stays = pd.read_csv(DATA_DIR / "icu_stays.csv")
    patients = pd.read_csv(DATA_DIR / "icu_patients.csv")
    admissions = pd.read_csv(DATA_DIR / "admissions.csv")
    notes = pd.read_csv(DATA_DIR / "icu_radiology_notes.csv")
    return stays, patients, admissions, notes


def build_first_24h_radiology_features(stays: pd.DataFrame, notes: pd.DataFrame) -> pd.DataFrame:
    note_cols = ["stay_id", "charttime", "text"]
    note_df = notes[note_cols].copy()
    note_df["stay_id"] = pd.to_numeric(note_df["stay_id"], errors="coerce")
    note_df["charttime"] = pd.to_datetime(note_df["charttime"], errors="coerce")
    note_df["text"] = note_df["text"].fillna("")
    note_df = note_df.dropna(subset=["stay_id", "charttime"])

    stay_times = stays[["stay_id", "intime"]].copy()
    stay_times["intime"] = pd.to_datetime(stay_times["intime"], errors="coerce")

    merged = note_df.merge(stay_times, on="stay_id", how="inner")
    merged = merged[
        (merged["charttime"] >= merged["intime"])
        & (merged["charttime"] < merged["intime"] + pd.Timedelta(hours=24))
    ].copy()

    merged["text_lower"] = merged["text"].str.lower()

    patterns = {
        "rad_ct_24h": r"\bct\b|computed tomography",
        "rad_mri_24h": r"\bmri\b|magnetic resonance",
        "rad_xray_24h": r"x-ray|xray|radiograph|portable chest|pa and lateral|ap portable",
        "rad_ultrasound_24h": r"ultrasound|\bu/s\b|\bus\b|sonograph",
        "kw_hemorrhage_24h": r"hemorrhage|haemorrhage|bleed|hematoma|haematoma",
        "kw_edema_24h": r"edema|oedema|swelling",
        "kw_pneumonia_24h": r"pneumonia|consolidation",
        "kw_effusion_24h": r"effusion",
        "kw_stroke_24h": r"stroke|infarct|ischemi",
        "kw_fracture_24h": r"fracture",
        "kw_postop_24h": r"postop|post-op|post operative|postoperative",
        "kw_intubation_24h": r"intubat|endotracheal tube|ett\b",
        "kw_line_24h": r"central line|picc|catheter|ng tube|enteric tube|chest tube",
    }

    for col, pattern in patterns.items():
        merged[col] = merged["text_lower"].str.contains(pattern, regex=True, na=False)

    agg_map = {col: "max" for col in patterns}
    agg_map["charttime"] = "count"
    agg_map["text"] = lambda s: "\n\n".join(t for t in s if t)

    feature_df = (
        merged.groupby("stay_id", as_index=False)
        .agg(agg_map)
        .rename(columns={"charttime": "rad_note_count_24h", "text": "rad_text_24h"})
    )

    out = stays[["stay_id"]].merge(feature_df, on="stay_id", how="left")
    out["has_rad_24h"] = out["rad_note_count_24h"].fillna(0).gt(0)
    out["rad_note_count_24h"] = out["rad_note_count_24h"].fillna(0).astype(int)
    out["rad_text_24h"] = out["rad_text_24h"].fillna("")

    bool_cols = ["has_rad_24h"] + list(patterns)
    for col in bool_cols:
        out[col] = out[col].fillna(False).astype(bool)

    return out


def prepare_modeling_data() -> tuple[pd.DataFrame, pd.Series]:
    stays, patients, admissions, notes = load_data()

    stays["los"] = pd.to_numeric(stays["los"], errors="coerce")
    stays = stays[stays["los"].notna()].copy()
    stays = stays[stays["los"] >= 1].copy()

    rad_features = build_first_24h_radiology_features(stays, notes)

    df = (
        stays.merge(patients, on="subject_id", how="left")
        .merge(admissions, on=["subject_id", "hadm_id"], how="left", suffixes=("", "_adm"))
        .merge(rad_features, on="stay_id", how="left")
    )

    df["intime"] = pd.to_datetime(df["intime"], errors="coerce")
    df["admittime"] = pd.to_datetime(df["admittime"], errors="coerce")
    df["edregtime"] = pd.to_datetime(df["edregtime"], errors="coerce")
    df["edouttime"] = pd.to_datetime(df["edouttime"], errors="coerce")

    df["hours_from_hosp_admit_to_icu"] = (
        (df["intime"] - df["admittime"]).dt.total_seconds() / 3600.0
    )
    df["ed_length_hours"] = (
        (df["edouttime"] - df["edregtime"]).dt.total_seconds() / 3600.0
    )
    df["icu_admit_hour"] = df["intime"].dt.hour
    df["icu_admit_weekend"] = df["intime"].dt.dayofweek.isin([5, 6])
    df["remaining_los_after_24h"] = df["los"] - 1.0
    df["log_remaining_los"] = np.log1p(df["remaining_los_after_24h"])

    for col in df.columns:
        if col.startswith(("rad_", "kw_")) or col in {"has_rad_24h", "icu_admit_weekend"}:
            if col == "rad_text_24h" or df[col].dtype == object:
                continue
            df[col] = df[col].fillna(False).astype(int)

    return df, df["log_remaining_los"]


def evaluate_predictions(y_true_days: pd.Series, y_pred_days: np.ndarray) -> dict[str, float]:
    y_pred_days = np.clip(y_pred_days, 0, None)
    return {
        "mae_days": mean_absolute_error(y_true_days, y_pred_days),
        "rmse_days": np.sqrt(mean_squared_error(y_true_days, y_pred_days)),
        "r2": r2_score(y_true_days, y_pred_days),
    }


def build_feature_sets(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    numeric_features = [
        "anchor_age",
        "hours_from_hosp_admit_to_icu",
        "ed_length_hours",
        "icu_admit_hour",
        "rad_note_count_24h",
        "icu_admit_weekend",
        "has_rad_24h",
        "rad_ct_24h",
        "rad_mri_24h",
        "rad_xray_24h",
        "rad_ultrasound_24h",
        "kw_hemorrhage_24h",
        "kw_edema_24h",
        "kw_pneumonia_24h",
        "kw_effusion_24h",
        "kw_stroke_24h",
        "kw_fracture_24h",
        "kw_postop_24h",
        "kw_intubation_24h",
        "kw_line_24h",
    ]
    categorical_features = [
        "gender",
        "first_careunit",
        "admission_type",
        "admission_location",
        "insurance",
        "language",
        "marital_status",
        "race",
    ]
    text_feature = ["rad_text_24h"]

    present_numeric = [c for c in numeric_features if c in df.columns]
    present_categorical = [c for c in categorical_features if c in df.columns]
    return present_numeric, present_categorical, text_feature


def build_structured_model(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", Ridge(alpha=1.0)),
        ]
    )


def build_text_model(numeric_features: list[str], categorical_features: list[str], text_feature: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
            (
                "text",
                Pipeline(
                    steps=[
                        (
                            "tfidf",
                            TfidfVectorizer(
                                lowercase=True,
                                stop_words="english",
                                max_features=600,
                                ngram_range=(1, 2),
                                min_df=10,
                            ),
                        ),
                        ("svd", TruncatedSVD(n_components=80, random_state=42)),
                    ]
                ),
                "rad_text_24h",
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", Ridge(alpha=2.0)),
        ]
    )


def save_feature_table(df: pd.DataFrame) -> Path:
    feature_cols = [
        "stay_id",
        "subject_id",
        "hadm_id",
        "remaining_los_after_24h",
        "has_rad_24h",
        "rad_note_count_24h",
        "rad_ct_24h",
        "rad_mri_24h",
        "rad_xray_24h",
        "rad_ultrasound_24h",
        "kw_hemorrhage_24h",
        "kw_edema_24h",
        "kw_pneumonia_24h",
        "kw_effusion_24h",
        "kw_stroke_24h",
        "kw_fracture_24h",
        "kw_postop_24h",
        "kw_intubation_24h",
        "kw_line_24h",
        "rad_text_24h",
    ]
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FEATURE_DIR / "radiology_features_first24h.csv"
    df[feature_cols].to_csv(out_path, index=False)
    return out_path


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    df, y_log = prepare_modeling_data()
    y_days = df["remaining_los_after_24h"]
    numeric_features, categorical_features, text_feature = build_feature_sets(df)

    X_train, X_test, y_train_log, y_test_log, y_train_days, y_test_days = train_test_split(
        df,
        y_log,
        y_days,
        test_size=0.2,
        random_state=42,
    )

    results = []

    structured_model = build_structured_model(numeric_features, categorical_features)
    structured_model.fit(X_train, y_train_log)
    structured_pred_days = np.expm1(structured_model.predict(X_test))
    results.append(
        {
            "model": "Structured + radiology flags",
            **evaluate_predictions(y_test_days, structured_pred_days),
        }
    )

    text_model = build_text_model(numeric_features, categorical_features, text_feature)
    text_model.fit(X_train, y_train_log)
    text_pred_days = np.expm1(text_model.predict(X_test))
    results.append(
        {
            "model": "Structured + flags + TF-IDF radiology text",
            **evaluate_predictions(y_test_days, text_pred_days),
        }
    )

    result_df = pd.DataFrame(results).sort_values("mae_days").reset_index(drop=True)
    feature_csv = save_feature_table(df)

    print("\nCohort")
    print(f"Rows used: {len(df):,}")
    print("Target: remaining ICU LOS after first 24 hours (days)")
    print(f"Train rows: {len(X_train):,}")
    print(f"Test rows: {len(X_test):,}")

    print("\nRadiology Feature Coverage")
    coverage_cols = [
        "has_rad_24h",
        "rad_ct_24h",
        "rad_mri_24h",
        "rad_xray_24h",
        "rad_ultrasound_24h",
        "kw_hemorrhage_24h",
        "kw_edema_24h",
        "kw_pneumonia_24h",
        "kw_effusion_24h",
        "kw_stroke_24h",
        "kw_fracture_24h",
        "kw_postop_24h",
        "kw_intubation_24h",
        "kw_line_24h",
    ]
    coverage = (
        df[coverage_cols]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
        .rename("pct_true")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    print(coverage.to_string(index=False, float_format=lambda x: f"{x:0.2f}"))

    print("\nModel Results")
    print(result_df.to_string(index=False, float_format=lambda x: f"{x:0.3f}"))

    print(f"\nSaved first-24h radiology feature table to: {feature_csv}")


if __name__ == "__main__":
    main()
