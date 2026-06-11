from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stays = pd.read_csv(DATA_DIR / "icu_stays.csv")
    patients = pd.read_csv(DATA_DIR / "icu_patients.csv")
    admissions = pd.read_csv(DATA_DIR / "admissions.csv")
    notes = pd.read_csv(DATA_DIR / "icu_radiology_notes.csv", usecols=["stay_id", "charttime"])
    return stays, patients, admissions, notes


def add_first_24h_note_flag(stays: pd.DataFrame, notes: pd.DataFrame) -> pd.DataFrame:
    base = stays[["stay_id", "intime"]].copy()
    base["intime"] = pd.to_datetime(base["intime"], errors="coerce")

    note_df = notes.dropna(subset=["stay_id", "charttime"]).copy()
    note_df["stay_id"] = pd.to_numeric(note_df["stay_id"], errors="coerce")
    note_df["charttime"] = pd.to_datetime(note_df["charttime"], errors="coerce")
    note_df = note_df.dropna(subset=["stay_id", "charttime"])

    joined = note_df.merge(base, on="stay_id", how="inner")
    joined["has_rad_24h"] = (
        (joined["charttime"] >= joined["intime"])
        & (joined["charttime"] < joined["intime"] + pd.Timedelta(hours=24))
    )

    flags = joined.groupby("stay_id", as_index=False)["has_rad_24h"].max()
    out = stays.merge(flags, on="stay_id", how="left")
    out["has_rad_24h"] = out["has_rad_24h"].fillna(False)
    return out


def prepare_modeling_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    stays, patients, admissions, notes = load_data()
    stays = add_first_24h_note_flag(stays, notes)
    df = (
        stays.merge(patients, on="subject_id", how="left")
        .merge(admissions, on=["subject_id", "hadm_id"], how="left", suffixes=("", "_adm"))
    )

    df["los"] = pd.to_numeric(df["los"], errors="coerce")
    df = df[df["los"].notna()].copy()

    # Restrict to stays that have a full 24-hour observation window.
    df = df[df["los"] >= 1].copy()
    df["remaining_los_after_24h"] = df["los"] - 1.0
    df["log_remaining_los"] = np.log1p(df["remaining_los_after_24h"])
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

    features = df[
        [
            "anchor_age",
            "hours_from_hosp_admit_to_icu",
            "ed_length_hours",
            "icu_admit_hour",
            "gender",
            "first_careunit",
            "admission_type",
            "admission_location",
            "insurance",
            "language",
            "marital_status",
            "race",
            "icu_admit_weekend",
            "has_rad_24h",
        ]
    ].copy()

    target_days = df["remaining_los_after_24h"].copy()
    target_log = df["log_remaining_los"].copy()
    return features, target_days, target_log


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    y_pred = np.clip(y_pred, 0, None)
    return {
        "mae_days": mean_absolute_error(y_true, y_pred),
        "rmse_days": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


def build_preprocessor() -> ColumnTransformer:
    numeric_features = [
        "anchor_age",
        "hours_from_hosp_admit_to_icu",
        "ed_length_hours",
        "icu_admit_hour",
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
        "icu_admit_weekend",
        "has_rad_24h",
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def main() -> None:
    X, y_days, y_log = prepare_modeling_data()

    X_train, X_test, y_train_days, y_test_days, y_train_log, y_test_log = train_test_split(
        X,
        y_days,
        y_log,
        test_size=0.2,
        random_state=42,
    )

    results: list[dict[str, float | str]] = []

    median_pred = np.repeat(y_train_days.median(), len(y_test_days))
    results.append(
        {
            "model": "Median Baseline",
            **evaluate_predictions(y_test_days, median_pred),
        }
    )

    ridge_pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    ridge_pipeline.fit(X_train, y_train_log)
    ridge_pred_days = np.expm1(ridge_pipeline.predict(X_test))
    results.append(
        {
            "model": "Ridge on log(remaining_los)",
            **evaluate_predictions(y_test_days, ridge_pred_days),
        }
    )

    rf_pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=10,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    rf_pipeline.fit(X_train, y_train_log)
    rf_pred_days = np.expm1(rf_pipeline.predict(X_test))
    results.append(
        {
            "model": "Random Forest on log(remaining_los)",
            **evaluate_predictions(y_test_days, rf_pred_days),
        }
    )

    result_df = pd.DataFrame(results).sort_values("mae_days").reset_index(drop=True)
    print("\nCohort")
    print(f"Rows used: {len(X):,}")
    print(f"Target: remaining ICU LOS after first 24 hours (days)")
    print(f"Train rows: {len(X_train):,}")
    print(f"Test rows: {len(X_test):,}")

    print("\nModel Results")
    print(result_df.to_string(index=False, float_format=lambda x: f"{x:0.3f}"))

    rf_model = rf_pipeline.named_steps["model"]
    preprocessor = rf_pipeline.named_steps["preprocess"]
    feature_names = preprocessor.get_feature_names_out()
    importances = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": rf_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    # Group one-hot encoded columns back to their source feature for a cleaner summary.
    grouped_rows = []
    for raw_feature in [
        "anchor_age",
        "hours_from_hosp_admit_to_icu",
        "ed_length_hours",
        "icu_admit_hour",
        "gender",
        "first_careunit",
        "admission_type",
        "admission_location",
        "insurance",
        "language",
        "marital_status",
        "race",
        "icu_admit_weekend",
        "has_rad_24h",
    ]:
        grouped_rows.append(
            {
                "feature_group": raw_feature,
                "importance": importances.loc[
                    importances["feature"].str.contains(raw_feature, regex=False), "importance"
                ].sum(),
            }
        )
    grouped_df = pd.DataFrame(grouped_rows).sort_values("importance", ascending=False)

    print("\nTop Encoded Feature Importances (Random Forest)")
    print(
        importances.head(20).to_string(
            index=False, float_format=lambda x: f"{x:0.4f}"
        )
    )

    print("\nGrouped Feature Importances (Random Forest)")
    print(
        grouped_df.to_string(
            index=False, float_format=lambda x: f"{x:0.4f}"
        )
    )


if __name__ == "__main__":
    main()
