from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output" / "model_results"


def prepare_modeling_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    cleaned_path = PROCESSED_DATA_DIR / "icu_stays_with_chartevents_24h_cleaned.parquet"
    patients_path = RAW_DATA_DIR / "icu_patients.csv"

    if not cleaned_path.exists():
        raise FileNotFoundError(
            f"Missing {cleaned_path}. Run preprocessing/build_chartevents_24h_features.py "
            "and preprocessing/clean_chartevents_24h.py first."
        )

    df = pd.read_parquet(cleaned_path)
    patients_df = pd.read_csv(patients_path)

    model_df = df.merge(
        patients_df[["subject_id", "gender", "anchor_age"]],
        on="subject_id",
        how="left",
    )
    model_df["los"] = pd.to_numeric(model_df["los"], errors="coerce")
    model_df = model_df[model_df["los"].notna()].copy()
    model_df = model_df[model_df["los"] >= 1].copy()

    chart_feature_cols = [
        "heart_rate_min", "heart_rate_max", "heart_rate_mean",
        "sbp_min", "sbp_max", "sbp_mean",
        "dbp_min", "dbp_max", "dbp_mean",
        "resp_rate_min", "resp_rate_max", "resp_rate_mean",
        "temperature_min", "temperature_max", "temperature_mean",
        "spo2_min", "spo2_max", "spo2_mean",
    ]
    categorical_cols = [
        "first_careunit",
        "admission_type",
        "admission_location",
        "gender",
    ]
    numeric_cols = ["anchor_age"] + chart_feature_cols

    missing_indicator_cols = []
    for col in chart_feature_cols:
        indicator_col = f"{col}_missing"
        model_df[indicator_col] = model_df[col].isna().astype(int)
        missing_indicator_cols.append(indicator_col)

    model_df["any_chart_feature_missing"] = model_df[chart_feature_cols].isna().any(axis=1).astype(int)
    model_df["all_chart_features_missing"] = model_df[chart_feature_cols].isna().all(axis=1).astype(int)
    missing_indicator_cols.extend(["any_chart_feature_missing", "all_chart_features_missing"])

    for col in categorical_cols:
        model_df[col] = model_df[col].fillna("missing")
    for col in numeric_cols:
        model_df[col] = model_df[col].fillna(model_df[col].median())

    model_df["remaining_los_after_24h"] = model_df["los"] - 1.0
    model_df["log_remaining_los"] = np.log1p(model_df["remaining_los_after_24h"])

    X = model_df[numeric_cols + missing_indicator_cols + categorical_cols].copy()
    y_days = model_df["remaining_los_after_24h"].copy()
    y_log = model_df["log_remaining_los"].copy()
    return X, y_days, y_log


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    y_pred = np.clip(y_pred, 0, None)
    return {
        "mae_days": mean_absolute_error(y_true, y_pred),
        "rmse_days": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
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
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )


def main() -> None:
    X, y_days, y_log = prepare_modeling_data()
    numeric_cols = [c for c in X.columns if c not in {"first_careunit", "admission_type", "admission_location", "gender"}]
    categorical_cols = ["first_careunit", "admission_type", "admission_location", "gender"]

    X_train, X_test, y_train_days, y_test_days, y_train_log, _ = train_test_split(
        X, y_days, y_log, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    models = [
        ("Median Baseline", None),
        ("Ridge + chart vitals", Ridge(alpha=1.0)),
        (
            "HistGradientBoosting + chart vitals",
            HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_depth=3,
                max_iter=200,
                min_samples_leaf=50,
                random_state=42,
            ),
        ),
    ]

    results = []
    for name, estimator in models:
        if estimator is None:
            pred = np.repeat(y_train_days.median(), len(y_test_days))
        else:
            pipeline = Pipeline(
                steps=[
                    ("preprocess", preprocessor),
                    ("model", estimator),
                ]
            )
            pipeline.fit(X_train, y_train_log)
            pred = np.expm1(pipeline.predict(X_test))
        results.append({"model": name, **evaluate_predictions(y_test_days, pred)})

    results_df = pd.DataFrame(results).sort_values("r2", ascending=False).reset_index(drop=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "chartevents_model_results.csv"
    results_df.to_csv(out_path, index=False)

    print(results_df.to_string(index=False, float_format=lambda x: f"{x:0.3f}"))
    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()
