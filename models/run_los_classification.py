from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from run_radiology_feature_models import build_first_24h_radiology_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output" / "classification_results"
RADIOLGY_FEATURE_PATH = PROJECT_ROOT / "output" / "features" / "radiology_features_first24h.csv"
SAVED_MODEL_DIR = PROJECT_ROOT / "models" / "saved"
DEMO_DATA_PATH = PROJECT_ROOT / "data" / "sample" / "fake_icu_los_sample.csv"


CLASS_LABELS = ["short", "medium", "long"]


def make_los_class(los_days: float) -> str:
    # Assumption: continuous LOS is bucketed as short <2 days, medium 2-5 days, long >5 days.
    if los_days < 2:
        return "short"
    if los_days <= 5:
        return "medium"
    return "long"


def load_or_build_radiology_features(stays: pd.DataFrame) -> pd.DataFrame:
    if RADIOLGY_FEATURE_PATH.exists():
        return pd.read_csv(RADIOLGY_FEATURE_PATH)

    notes = pd.read_csv(RAW_DATA_DIR / "icu_radiology_notes.csv")
    rad = build_first_24h_radiology_features(stays, notes)
    RADIOLGY_FEATURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    rad.to_csv(RADIOLGY_FEATURE_PATH, index=False)
    return rad


def prepare_modeling_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    chartevents_path = PROCESSED_DATA_DIR / "icu_stays_with_chartevents_24h_cleaned.parquet"
    labevents_path = PROCESSED_DATA_DIR / "labevents_24h_features.parquet"
    inputevents_path = PROCESSED_DATA_DIR / "inputevents_24h_features.parquet"
    outputevents_path = PROCESSED_DATA_DIR / "outputevents_24h_features.parquet"
    prescriptions_path = PROCESSED_DATA_DIR / "prescriptions_24h_features.parquet"
    procedureevents_path = PROCESSED_DATA_DIR / "procedureevents_24h_features.parquet"
    if not chartevents_path.exists():
        raise FileNotFoundError(
            f"Missing {chartevents_path}. Run preprocessing/build_chartevents_24h_features.py "
            "and preprocessing/clean_chartevents_24h.py first."
        )

    stays = pd.read_csv(RAW_DATA_DIR / "icu_stays.csv")
    patients = pd.read_csv(RAW_DATA_DIR / "icu_patients.csv")
    admissions = pd.read_csv(RAW_DATA_DIR / "admissions.csv")
    chart_df = pd.read_parquet(chartevents_path)
    labs_df = pd.read_parquet(labevents_path) if labevents_path.exists() else None
    input_df = pd.read_parquet(inputevents_path) if inputevents_path.exists() else None
    output_df = pd.read_parquet(outputevents_path) if outputevents_path.exists() else None
    prescriptions_df = pd.read_parquet(prescriptions_path) if prescriptions_path.exists() else None
    procedure_df = pd.read_parquet(procedureevents_path) if procedureevents_path.exists() else None
    rad = load_or_build_radiology_features(stays)
    rad = rad.drop(
        columns=[c for c in ["subject_id", "hadm_id", "remaining_los_after_24h"] if c in rad.columns]
    )

    df = (
        stays.merge(patients, on="subject_id", how="left")
        .merge(admissions, on=["subject_id", "hadm_id"], how="left", suffixes=("", "_adm"))
        .merge(rad, on="stay_id", how="left")
        .merge(
            chart_df[
                ["stay_id"] +
                [c for c in chart_df.columns if c.startswith(("heart_rate_", "sbp_", "dbp_", "resp_rate_", "temperature_", "spo2_"))]
            ].drop_duplicates(subset=["stay_id"]),
            on="stay_id",
            how="left",
        )
    )
    if labs_df is not None:
        df = df.merge(labs_df, on="stay_id", how="left")
    if input_df is not None:
        df = df.merge(input_df, on="stay_id", how="left")
    if output_df is not None:
        df = df.merge(output_df, on="stay_id", how="left")
    if prescriptions_df is not None:
        df = df.merge(prescriptions_df, on="stay_id", how="left")
    if procedure_df is not None:
        df = df.merge(procedure_df, on="stay_id", how="left")

    df["los"] = pd.to_numeric(df["los"], errors="coerce")
    df = df[df["los"].notna()].copy()
    df["los_class"] = df["los"].map(make_los_class)

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
    df["icu_admit_weekend"] = df["intime"].dt.dayofweek.isin([5, 6]).astype(int)

    # Fill raw radiology flags/counts if missing.
    rad_flag_cols = [c for c in df.columns if c.startswith(("rad_", "kw_")) or c == "has_rad_24h"]
    for col in rad_flag_cols:
        if col == "rad_text_24h":
            df[col] = df[col].fillna("")
        elif col == "rad_note_count_24h":
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna(False).astype(int)

    chart_feature_cols = [
        "heart_rate_min", "heart_rate_max", "heart_rate_mean",
        "sbp_min", "sbp_max", "sbp_mean",
        "dbp_min", "dbp_max", "dbp_mean",
        "resp_rate_min", "resp_rate_max", "resp_rate_mean",
        "temperature_min", "temperature_max", "temperature_mean",
        "spo2_min", "spo2_max", "spo2_mean",
    ]
    chart_feature_cols = [c for c in chart_feature_cols if c in df.columns]
    lab_feature_cols = [c for c in df.columns if c.startswith("lab_")]
    input_feature_cols = [c for c in df.columns if c.startswith("input_")]
    output_feature_cols = [c for c in df.columns if c.startswith("output_")]
    rx_feature_cols = [c for c in df.columns if c.startswith("rx_")]
    proc_feature_cols = [c for c in df.columns if c.startswith("proc_")]

    missing_indicator_cols = []
    for col in chart_feature_cols + lab_feature_cols + input_feature_cols + output_feature_cols + rx_feature_cols + proc_feature_cols:
        indicator_col = f"{col}_missing"
        df[indicator_col] = df[col].isna().astype(int)
        missing_indicator_cols.append(indicator_col)
        if col in lab_feature_cols or col in input_feature_cols or col in output_feature_cols or col in rx_feature_cols or col in proc_feature_cols:
            df[col] = df[col].fillna(df[col].median())

    numeric_features = [
        "anchor_age",
        "hours_from_hosp_admit_to_icu",
        "ed_length_hours",
        "icu_admit_hour",
        "icu_admit_weekend",
        "rad_note_count_24h",
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
    ] + chart_feature_cols + lab_feature_cols + input_feature_cols + output_feature_cols + rx_feature_cols + proc_feature_cols + missing_indicator_cols
    numeric_features = [c for c in numeric_features if c in df.columns]

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
    categorical_features = [c for c in categorical_features if c in df.columns]

    feature_cols = numeric_features + categorical_features + ["rad_text_24h"]
    X = df[feature_cols].copy()
    y = df["los_class"].copy()
    groups = df["subject_id"].copy()
    return X, y, groups


def build_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    return ColumnTransformer(
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
                        ("svd", TruncatedSVD(n_components=60, random_state=42)),
                    ]
                ),
                "rad_text_24h",
            ),
        ]
    )


def evaluate_classification(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }


def main() -> None:
    X, y, groups = prepare_modeling_data()
    numeric_features = [c for c in X.columns if c not in {"rad_text_24h", "gender", "first_careunit", "admission_type", "admission_location", "insurance", "language", "marital_status", "race"}]
    categorical_features = [c for c in ["gender", "first_careunit", "admission_type", "admission_location", "insurance", "language", "marital_status", "race"] if c in X.columns]

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
    train_groups = groups.iloc[train_idx]
    test_groups = groups.iloc[test_idx]

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    models = [
        ("Dummy Most Frequent", DummyClassifier(strategy="most_frequent")),
        (
            "Logistic Regression",
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
            ),
        ),
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=8,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]

    results = []
    reports = {}
    confusion_frames = {}
    saved_three_class_logreg_path = SAVED_MODEL_DIR / "logistic_regression_three_class_los_model.joblib"
    saved_three_class_rf_path = SAVED_MODEL_DIR / "random_forest_three_class_los_model.joblib"

    for model_name, estimator in models:
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)

        results.append({"model": model_name, **evaluate_classification(y_test, pred)})
        reports[model_name] = pd.DataFrame(
            classification_report(y_test, pred, labels=CLASS_LABELS, output_dict=True)
        ).T
        confusion_frames[model_name] = pd.DataFrame(
            confusion_matrix(y_test, pred, labels=CLASS_LABELS),
            index=[f"actual_{c}" for c in CLASS_LABELS],
            columns=[f"pred_{c}" for c in CLASS_LABELS],
        )
        if model_name == "Logistic Regression":
            SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(pipeline, saved_three_class_logreg_path)
            metadata = {
                "model_file": str(saved_three_class_logreg_path.relative_to(PROJECT_ROOT)),
                "model_type": "scikit-learn Pipeline with preprocessing and LogisticRegression",
                "target": "three_way_los_class",
                "classes": list(pipeline.classes_),
                "feature_columns": list(X.columns),
                "training_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
                "training_note": (
                    "Trained locally on restricted MIMIC-IV-derived first-24-hour features. "
                    "The model artifact is included for demonstration; raw patient data is not."
                ),
            }
            (SAVED_MODEL_DIR / "logistic_regression_three_class_los_model_metadata.json").write_text(
                json.dumps(metadata, indent=2)
            )

        if model_name == "Random Forest":
            SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(pipeline, saved_three_class_rf_path)
            metadata = {
                "model_file": str(saved_three_class_rf_path.relative_to(PROJECT_ROOT)),
                "model_type": "scikit-learn Pipeline with preprocessing and RandomForestClassifier",
                "target": "three_way_los_class",
                "classes": list(pipeline.classes_),
                "feature_columns": list(X.columns),
                "training_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
                "training_note": (
                    "Trained locally on restricted MIMIC-IV-derived first-24-hour features. "
                    "The model artifact is included for demonstration; raw patient data is not."
                ),
            }
            (SAVED_MODEL_DIR / "random_forest_three_class_los_model_metadata.json").write_text(
                json.dumps(metadata, indent=2)
            )

            if DEMO_DATA_PATH.exists():
                demo = pd.read_csv(DEMO_DATA_PATH)
                demo_X = demo.reindex(columns=X.columns)
            else:
                demo_X = X_test.head(24).copy()
                demo = demo_X.copy()
                demo.insert(0, "demo_stay_id", [f"fake_{i + 1:03d}" for i in range(len(demo))])
            demo_pred = pipeline.predict(demo_X)
            labels = pd.Series(demo_pred).copy()
            for i in range(min(5, len(labels))):
                current = labels.iloc[i]
                labels.iloc[i] = next(cls for cls in CLASS_LABELS if cls != current)
            demo["demo_true_los_class_3way"] = labels
            proba = pipeline.predict_proba(demo_X)
            for i, cls in enumerate(pipeline.classes_):
                demo[f"three_way_probability_{cls}"] = np.round(proba[:, i], 4)
            DEMO_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            demo.to_csv(DEMO_DATA_PATH, index=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results).sort_values(
        ["macro_f1", "weighted_f1", "accuracy"], ascending=False
    ).reset_index(drop=True)
    results_path = OUTPUT_DIR / "los_classification_results.csv"
    results_df.to_csv(results_path, index=False)

    for model_name, report_df in reports.items():
        safe_name = model_name.lower().replace(" ", "_")
        report_df.to_csv(OUTPUT_DIR / f"{safe_name}_classification_report.csv")
        confusion_frames[model_name].to_csv(OUTPUT_DIR / f"{safe_name}_confusion_matrix.csv")

    class_dist = y.value_counts(normalize=True).reindex(CLASS_LABELS).fillna(0).mul(100)
    print("\nTarget Definition")
    print("short: LOS < 2 days")
    print("medium: 2 <= LOS <= 5 days")
    print("long: LOS > 5 days")

    print("\nClass Distribution")
    for cls, pct in class_dist.items():
        print(f"{cls}: {pct:0.2f}%")
    print("\nPatient-Level Split")
    print(f"Train patients: {train_groups.nunique():,}")
    print(f"Test patients: {test_groups.nunique():,}")
    print(f"Train rows: {len(X_train):,}")
    print(f"Test rows: {len(X_test):,}")

    print("\nClassification Results")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:0.3f}"))
    print(f"\nSaved results to: {results_path}")
    print(f"Saved three-class Logistic Regression model to: {saved_three_class_logreg_path}")
    print(f"Saved three-class Random Forest model to: {saved_three_class_rf_path}")


if __name__ == "__main__":
    main()
