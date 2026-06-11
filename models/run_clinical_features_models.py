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


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    y_pred = np.clip(y_pred, 0, None)
    return {
        "mae_days": mean_absolute_error(y_true, y_pred),
        "rmse_days": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


def prepare_modeling_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    chartevents_path = PROCESSED_DATA_DIR / "icu_stays_with_chartevents_24h_cleaned.parquet"
    labevents_path = PROCESSED_DATA_DIR / "labevents_24h_features.parquet"
    inputevents_path = PROCESSED_DATA_DIR / "inputevents_24h_features.parquet"
    outputevents_path = PROCESSED_DATA_DIR / "outputevents_24h_features.parquet"
    prescriptions_path = PROCESSED_DATA_DIR / "prescriptions_24h_features.parquet"
    procedureevents_path = PROCESSED_DATA_DIR / "procedureevents_24h_features.parquet"
    patients_path = RAW_DATA_DIR / "icu_patients.csv"

    missing = [p for p in [chartevents_path, labevents_path, patients_path] if not p.exists()]
    if missing:
        missing_str = "\n".join(str(p) for p in missing)
        raise FileNotFoundError(
            "Cannot run combined clinical-features model because these files are missing:\n"
            f"{missing_str}\n\n"
            "Build chart-events and lab features first."
        )

    df = pd.read_parquet(chartevents_path)
    labs = pd.read_parquet(labevents_path)
    inputs = pd.read_parquet(inputevents_path) if inputevents_path.exists() else None
    outputs = pd.read_parquet(outputevents_path) if outputevents_path.exists() else None
    prescriptions = pd.read_parquet(prescriptions_path) if prescriptions_path.exists() else None
    procedures = pd.read_parquet(procedureevents_path) if procedureevents_path.exists() else None
    patients = pd.read_csv(patients_path)

    model_df = (
        df.merge(labs, on="stay_id", how="left")
        .merge(patients[["subject_id", "gender", "anchor_age"]], on="subject_id", how="left")
    )
    if inputs is not None:
        model_df = model_df.merge(inputs, on="stay_id", how="left")
    if outputs is not None:
        model_df = model_df.merge(outputs, on="stay_id", how="left")
    if prescriptions is not None:
        model_df = model_df.merge(prescriptions, on="stay_id", how="left")
    if procedures is not None:
        model_df = model_df.merge(procedures, on="stay_id", how="left")
    model_df["los"] = pd.to_numeric(model_df["los"], errors="coerce")
    model_df = model_df[model_df["los"].notna()].copy()
    model_df = model_df[model_df["los"] >= 1].copy()

    chart_cols = [
        "heart_rate_min", "heart_rate_max", "heart_rate_mean",
        "sbp_min", "sbp_max", "sbp_mean",
        "dbp_min", "dbp_max", "dbp_mean",
        "resp_rate_min", "resp_rate_max", "resp_rate_mean",
        "temperature_min", "temperature_max", "temperature_mean",
        "spo2_min", "spo2_max", "spo2_mean",
    ]
    lab_cols = [c for c in model_df.columns if c.startswith("lab_")]
    input_cols = [c for c in model_df.columns if c.startswith("input_")]
    output_cols = [c for c in model_df.columns if c.startswith("output_")]
    rx_cols = [c for c in model_df.columns if c.startswith("rx_")]
    proc_cols = [c for c in model_df.columns if c.startswith("proc_")]
    categorical_cols = ["first_careunit", "admission_type", "admission_location", "gender"]
    numeric_cols = ["anchor_age"] + chart_cols + lab_cols + input_cols + output_cols + rx_cols + proc_cols

    missing_indicator_cols = []
    for col in chart_cols + lab_cols + input_cols + output_cols + rx_cols + proc_cols:
        indicator_col = f"{col}_missing"
        model_df[indicator_col] = model_df[col].isna().astype(int)
        missing_indicator_cols.append(indicator_col)
        if pd.api.types.is_numeric_dtype(model_df[col]):
            model_df[col] = model_df[col].fillna(model_df[col].median())

    for col in categorical_cols:
        model_df[col] = model_df[col].fillna("missing")

    model_df["remaining_los_after_24h"] = model_df["los"] - 1.0
    model_df["log_remaining_los"] = np.log1p(model_df["remaining_los_after_24h"])

    X = model_df[numeric_cols + missing_indicator_cols + categorical_cols].copy()
    y_days = model_df["remaining_los_after_24h"].copy()
    y_log = model_df["log_remaining_los"].copy()
    return X, y_days, y_log


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
    categorical_cols = ["first_careunit", "admission_type", "admission_location", "gender"]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    X_train, X_test, y_train_days, y_test_days, y_train_log, _ = train_test_split(
        X, y_days, y_log, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    models = [
        ("Median Baseline", None),
        ("Ridge + vitals + labs + interventions + outputs + prescriptions + procedures", Ridge(alpha=1.0)),
        (
            "HistGradientBoosting + vitals + labs + interventions + outputs + prescriptions + procedures",
            HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_depth=3,
                max_iter=220,
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
    out_path = OUTPUT_DIR / "clinical_features_model_results.csv"
    results_df.to_csv(out_path, index=False)
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:0.3f}"))
    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()
