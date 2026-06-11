from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MODELS_DIR = PROJECT_ROOT / "models"
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from models.run_binary_short_stay_classification import (  # noqa: E402
    build_preprocessor,
    prepare_modeling_data,
)


OUTPUT_DIR = PROJECT_ROOT / "output" / "model_comparison_figures"


def fit_binary_random_forest() -> tuple[Pipeline, pd.DataFrame, pd.Series, np.ndarray]:
    X, y, groups = prepare_modeling_data()
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

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()

    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(numeric_features, categorical_features)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=8,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)
    short_idx = list(pipeline.classes_).index("short")
    short_prob = proba[:, short_idx]
    return pipeline, X_test, y_test, short_prob


def make_calibration_plot(y_test: pd.Series, short_prob: np.ndarray) -> Path:
    y_true = (y_test == "short").astype(int)
    frac_pos, mean_pred = calibration_curve(y_true, short_prob, n_bins=10, strategy="quantile")

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#7f8c8d", linewidth=1.5, label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, marker="o", linewidth=2.5, color="#1d3557", label="Random Forest")
    ax.set_title("Binary Model Calibration", fontsize=15, pad=12)
    ax.set_xlabel("Predicted Probability of Short Stay", fontsize=12)
    ax.set_ylabel("Observed Short-Stay Rate", fontsize=12)
    ax.legend(frameon=True, loc="upper left")
    plt.tight_layout()

    output_path = OUTPUT_DIR / "binary_calibration_curve.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def make_unit_performance_plot(X_test: pd.DataFrame, y_test: pd.Series, short_prob: np.ndarray) -> Path:
    pred = np.where(short_prob >= 0.5, "short", "longer")
    eval_df = pd.DataFrame(
        {
            "first_careunit": X_test["first_careunit"].astype(str).values,
            "y_true": y_test.values,
            "pred": pred,
            "short_prob": short_prob,
        }
    )
    top_units = eval_df["first_careunit"].value_counts().head(6).index
    eval_df = eval_df[eval_df["first_careunit"].isin(top_units)].copy()

    rows = []
    for unit, group in eval_df.groupby("first_careunit"):
        y_bin = (group["y_true"] == "short").astype(int)
        rows.append(
            {
                "first_careunit": unit,
                "accuracy": accuracy_score(group["y_true"], group["pred"]),
                "roc_auc": roc_auc_score(y_bin, group["short_prob"]) if y_bin.nunique() > 1 else np.nan,
                "n": len(group),
            }
        )
    perf = pd.DataFrame(rows).sort_values("roc_auc", ascending=True)
    perf["unit_label"] = perf["first_careunit"] + " (n=" + perf["n"].astype(str) + ")"

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.5), sharey=True)

    sns.barplot(data=perf, x="accuracy", y="unit_label", color="#457b9d", ax=axes[0])
    axes[0].set_title("Accuracy by ICU Unit", fontsize=14, pad=10)
    axes[0].set_xlabel("Accuracy", fontsize=12)
    axes[0].set_ylabel("")
    axes[0].set_xlim(0, 1)

    sns.barplot(data=perf, x="roc_auc", y="unit_label", color="#1d3557", ax=axes[1])
    axes[1].set_title("ROC AUC by ICU Unit", fontsize=14, pad=10)
    axes[1].set_xlabel("ROC AUC", fontsize=12)
    axes[1].set_ylabel("")
    axes[1].set_xlim(0, 1)

    fig.suptitle("Binary Model Performance Across Major ICU Units", fontsize=16, y=1.02)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "binary_performance_by_careunit.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def make_risk_stratification_plot(y_test: pd.Series, short_prob: np.ndarray) -> Path:
    df = pd.DataFrame(
        {
            "short_prob": short_prob,
            "observed_short": (y_test == "short").astype(int).values,
        }
    )
    df["risk_decile"] = pd.qcut(df["short_prob"], 10, labels=False, duplicates="drop") + 1
    summary = (
        df.groupby("risk_decile", as_index=False)
        .agg(
            mean_predicted_short=("short_prob", "mean"),
            observed_short_rate=("observed_short", "mean"),
            count=("observed_short", "size"),
        )
    )
    summary["observed_short_pct"] = summary["observed_short_rate"] * 100
    summary["mean_predicted_short_pct"] = summary["mean_predicted_short"] * 100

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.plot(
        summary["risk_decile"],
        summary["mean_predicted_short_pct"],
        marker="o",
        linewidth=2.5,
        color="#1d3557",
        label="Predicted short-stay probability",
    )
    ax.plot(
        summary["risk_decile"],
        summary["observed_short_pct"],
        marker="s",
        linewidth=2.5,
        color="#e76f51",
        label="Observed short-stay rate",
    )
    ax.set_title("Observed Outcomes Across Predicted Risk Groups", fontsize=15, pad=12)
    ax.set_xlabel("Predicted Short-Stay Risk Decile", fontsize=12)
    ax.set_ylabel("Percent Short Stay", fontsize=12)
    ax.set_ylim(0, 100)
    ax.legend(frameon=True, loc="upper left")
    plt.tight_layout()

    output_path = OUTPUT_DIR / "binary_risk_stratification_curve.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _pipeline, X_test, y_test, short_prob = fit_binary_random_forest()

    calibration_path = make_calibration_plot(y_test, short_prob)
    unit_perf_path = make_unit_performance_plot(X_test, y_test, short_prob)
    risk_path = make_risk_stratification_plot(y_test, short_prob)

    print(f"Saved calibration plot to: {calibration_path}")
    print(f"Saved unit performance plot to: {unit_perf_path}")
    print(f"Saved risk stratification plot to: {risk_path}")


if __name__ == "__main__":
    main()
