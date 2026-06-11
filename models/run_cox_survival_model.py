from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output" / "survival_results"


def prepare_modeling_data() -> pd.DataFrame:
    chartevents_path = PROCESSED_DATA_DIR / "icu_stays_with_chartevents_24h_cleaned.parquet"
    labevents_path = PROCESSED_DATA_DIR / "labevents_24h_features.parquet"
    inputevents_path = PROCESSED_DATA_DIR / "inputevents_24h_features.parquet"
    outputevents_path = PROCESSED_DATA_DIR / "outputevents_24h_features.parquet"
    prescriptions_path = PROCESSED_DATA_DIR / "prescriptions_24h_features.parquet"
    procedureevents_path = PROCESSED_DATA_DIR / "procedureevents_24h_features.parquet"

    if not chartevents_path.exists():
        raise FileNotFoundError(
            f"Missing {chartevents_path}. Run the chart-events preprocessing first."
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
    rad = pd.read_csv(PROJECT_ROOT / "output" / "features" / "radiology_features_first24h.csv")
    rad = rad.drop(
        columns=[c for c in ["subject_id", "hadm_id", "remaining_los_after_24h"] if c in rad.columns]
    )

    df = (
        stays.merge(patients, on="subject_id", how="left")
        .merge(admissions, on=["subject_id", "hadm_id"], how="left", suffixes=("", "_adm"))
        .merge(rad, on="stay_id", how="left")
        .merge(
            chart_df[
                ["stay_id"]
                + [
                    c
                    for c in chart_df.columns
                    if c.startswith(
                        ("heart_rate_", "sbp_", "dbp_", "resp_rate_", "temperature_", "spo2_")
                    )
                ]
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
    df = df[df["los"] >= 1].copy()

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

    rad_flag_cols = [c for c in df.columns if c.startswith(("rad_", "kw_")) or c == "has_rad_24h"]
    for col in rad_flag_cols:
        if col == "rad_text_24h":
            continue
        if col == "rad_note_count_24h":
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

    indicator_frames = []
    for col in chart_feature_cols + lab_feature_cols + input_feature_cols + output_feature_cols + rx_feature_cols + proc_feature_cols:
        indicator_frames.append(
            pd.Series(df[col].isna().astype(int), name=f"{col}_missing", index=df.index)
        )
        df[col] = df[col].fillna(df[col].median())
    if indicator_frames:
        df = pd.concat([df] + indicator_frames, axis=1)

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
    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].fillna("missing")

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
    ] + chart_feature_cols + lab_feature_cols + input_feature_cols + output_feature_cols + rx_feature_cols + proc_feature_cols
    numeric_features += [c for c in df.columns if c.endswith("_missing")]
    numeric_features = [c for c in numeric_features if c in df.columns]
    categorical_features = [c for c in categorical_features if c in df.columns]

    model_df = df[numeric_features + categorical_features].copy()
    model_df["subject_id"] = df["subject_id"].values
    model_df["duration_days"] = df["los"] - 1.0
    model_df["event_observed"] = 1
    return model_df


def build_design_matrices(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, int, int]:
    feature_cols = [c for c in df.columns if c not in {"duration_days", "event_observed", "subject_id"}]
    categorical_cols = [c for c in df.columns if df[c].dtype == "object"]
    categorical_cols = [c for c in categorical_cols if c in feature_cols]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(df, groups=df["subject_id"]))
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

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
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )

    X_train = preprocessor.fit_transform(train_df[feature_cols])
    X_test = preprocessor.transform(test_df[feature_cols])
    feature_names = preprocessor.get_feature_names_out()

    X_train_df = pd.DataFrame(X_train, columns=feature_names, index=train_df.index)
    X_test_df = pd.DataFrame(X_test, columns=feature_names, index=test_df.index)

    train_ready = pd.concat(
        [X_train_df, train_df[["duration_days", "event_observed"]]], axis=1
    )
    test_ready = pd.concat(
        [X_test_df, test_df[["duration_days", "event_observed"]]], axis=1
    )
    return (
        train_ready,
        train_df["duration_days"],
        test_ready,
        test_df["duration_days"],
        train_df["subject_id"].nunique(),
        test_df["subject_id"].nunique(),
    )


def make_risk_group_plot(
    durations: pd.Series,
    events: pd.Series,
    risk_scores: pd.Series,
    out_path: Path,
) -> None:
    risk_group = pd.qcut(risk_scores.rank(method="first"), 3, labels=["low", "medium", "high"])
    plot_df = pd.DataFrame(
        {
            "duration_days": durations,
            "event_observed": events,
            "risk_group": risk_group,
        }
    )

    plt.figure(figsize=(8, 5))
    kmf = KaplanMeierFitter()
    color_map = {"low": "#2a9d8f", "medium": "#e9c46a", "high": "#e76f51"}
    for group in ["low", "medium", "high"]:
        group_df = plot_df[plot_df["risk_group"] == group]
        kmf.fit(
            durations=group_df["duration_days"],
            event_observed=group_df["event_observed"],
            label=f"{group.title()} predicted risk",
        )
        kmf.plot_survival_function(ci_show=False, color=color_map[group])

    plt.title("Remaining ICU Stay Survival Curves by Predicted Risk")
    plt.xlabel("Days Remaining in ICU After First 24 Hours")
    plt.ylabel("Probability of Still Being in ICU")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_df = prepare_modeling_data()
    (
        train_ready,
        train_durations,
        test_ready,
        test_durations,
        train_patients,
        test_patients,
    ) = build_design_matrices(model_df)

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(train_ready, duration_col="duration_days", event_col="event_observed")

    train_risk = cph.predict_partial_hazard(train_ready.drop(columns=["duration_days", "event_observed"]))
    test_risk = cph.predict_partial_hazard(test_ready.drop(columns=["duration_days", "event_observed"]))

    train_cindex = concordance_index(
        train_durations, -train_risk.to_numpy().ravel(), train_ready["event_observed"]
    )
    test_cindex = concordance_index(
        test_durations, -test_risk.to_numpy().ravel(), test_ready["event_observed"]
    )

    summary = cph.summary.reset_index().rename(columns={"index": "feature"})
    summary_out = OUTPUT_DIR / "cox_feature_summary.csv"
    summary.to_csv(summary_out, index=False)

    top_features = (
        summary.loc[:, ["covariate", "coef", "exp(coef)", "se(coef)", "p"]]
        .sort_values("p", ascending=True)
        .head(25)
        .rename(columns={"covariate": "feature", "exp(coef)": "hazard_ratio"})
    )
    top_out = OUTPUT_DIR / "cox_top_features.csv"
    top_features.to_csv(top_out, index=False)

    metrics_df = pd.DataFrame(
        [
            {
                "model": "Cox proportional hazards",
                "train_concordance_index": train_cindex,
                "test_concordance_index": test_cindex,
                "rows_used": len(model_df),
                "num_features_after_encoding": train_ready.shape[1] - 2,
                "train_patients": train_patients,
                "test_patients": test_patients,
            }
        ]
    )
    metrics_out = OUTPUT_DIR / "cox_model_metrics.csv"
    metrics_df.to_csv(metrics_out, index=False)

    plot_out = OUTPUT_DIR / "cox_risk_group_survival_curves.png"
    make_risk_group_plot(
        durations=test_ready["duration_days"],
        events=test_ready["event_observed"],
        risk_scores=pd.Series(test_risk.to_numpy().ravel(), index=test_ready.index),
        out_path=plot_out,
    )

    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:0.3f}"))
    print("\nTop hazard-ratio features")
    print(top_features.head(15).to_string(index=False, float_format=lambda x: f"{x:0.3f}"))
    print(f"\nSaved metrics to: {metrics_out}")
    print(f"Saved coefficient summary to: {summary_out}")
    print(f"Saved top features to: {top_out}")
    print(f"Saved risk-group survival plot to: {plot_out}")


if __name__ == "__main__":
    main()
