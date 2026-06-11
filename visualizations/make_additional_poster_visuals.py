from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.errors import PerformanceWarning
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MODELS_DIR = PROJECT_ROOT / "models"
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from models.run_binary_short_stay_classification import (  # noqa: E402
    build_preprocessor as build_binary_preprocessor,
    prepare_modeling_data as prepare_binary_data,
)
from models.run_los_classification import prepare_modeling_data as prepare_multiclass_data  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "output" / "model_comparison_figures"

warnings.filterwarnings("ignore", category=PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def make_los_mix_by_careunit() -> Path:
    X_multi, y_multi, _groups = prepare_multiclass_data()
    plot_df = pd.DataFrame(
        {
            "first_careunit": X_multi["first_careunit"].astype(str),
            "los_class": y_multi.astype(str),
        }
    )
    top_units = plot_df["first_careunit"].value_counts().head(6).index
    plot_df = plot_df[plot_df["first_careunit"].isin(top_units)].copy()

    unit_labels = {
        "Medical Intensive Care Unit (MICU)": "MICU",
        "Medical/Surgical Intensive Care Unit (MICU/SICU)": "MICU/SICU",
        "Cardiac Vascular Intensive Care Unit (CVICU)": "CVICU",
        "Surgical Intensive Care Unit (SICU)": "SICU",
        "Coronary Care Unit (CCU)": "CCU",
        "Trauma SICU (TSICU)": "TSICU",
    }

    counts = (
        plot_df.groupby(["first_careunit", "los_class"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["short", "medium", "long"])
    )
    proportions = counts.div(counts.sum(axis=1), axis=0)
    ordered_units = counts.sum(axis=1).sort_values(ascending=False).index
    proportions = proportions.loc[ordered_units]

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6.8))
    colors = ["#457b9d", "#e9c46a", "#e76f51"]
    bottom = np.zeros(len(proportions))
    for idx, cls in enumerate(["short", "medium", "long"]):
        vals = proportions[cls].values
        ax.bar(
            [unit_labels.get(idx_name, idx_name) for idx_name in proportions.index],
            vals,
            bottom=bottom,
            color=colors[idx],
            edgecolor="white",
            linewidth=1,
            label=cls.title(),
        )
        bottom += vals

    ax.set_title("LOS Class Mix Across Major ICU Units", fontsize=16, pad=14)
    ax.set_ylabel("Share of ICU Stays", fontsize=12)
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels([f"{int(v * 100)}%" for v in np.linspace(0, 1, 6)])
    ax.tick_params(axis="x", rotation=0, labelsize=11)
    ax.legend(
        frameon=True,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
    )
    plt.tight_layout()

    out = OUTPUT_DIR / "los_mix_by_careunit.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def make_early_burden_heatmap() -> Path:
    X_multi, y_multi, _groups = prepare_multiclass_data()
    selected = {
        "heart_rate_mean": "Mean heart rate",
        "resp_rate_mean": "Mean respiratory rate",
        "spo2_min": "Minimum SpO2",
        "lab_creat_max": "Max creatinine",
        "lab_wbc_max": "Max WBC",
        "input_total_ml_24h": "Input volume",
        "output_total_ml_24h": "Output volume",
        "rx_order_count_24h": "Medication orders",
        "proc_event_count_24h": "Procedure count",
        "rad_note_count_24h": "Radiology note count",
    }
    cols = [c for c in selected if c in X_multi.columns]
    plot_df = X_multi[cols].copy()
    plot_df["los_class"] = pd.Categorical(y_multi, categories=["short", "medium", "long"], ordered=True)

    summary = plot_df.groupby("los_class")[cols].median().T
    # Standardize within feature so the heatmap highlights relative burden across classes.
    z = summary.apply(
        lambda row: np.zeros(len(row)) if row.std(ddof=0) == 0 else (row - row.mean()) / row.std(ddof=0),
        axis=1,
        result_type="broadcast",
    )
    z.index = [selected[c] for c in cols]
    z.columns = ["Short", "Medium", "Long"]

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(8.6, 7.4))
    sns.heatmap(
        z,
        cmap="RdYlBu_r",
        center=0,
        linewidths=0.75,
        linecolor="white",
        cbar_kws={"label": "Relative level within feature"},
        ax=ax,
    )
    ax.set_title("First-24-Hour Clinical Burden Shifts Across LOS Groups", fontsize=16, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    plt.tight_layout()

    out = OUTPUT_DIR / "early_burden_by_los_class_heatmap.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def _fit_binary_random_forest() -> tuple[pd.Series, np.ndarray]:
    X_bin, y_bin, groups = prepare_binary_data()

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
    numeric_features = [c for c in X_bin.columns if c not in text_and_cat]
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
        if c in X_bin.columns
    ]

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(splitter.split(X_bin, y_bin, groups))
    X_train, X_test = X_bin.iloc[train_idx].copy(), X_bin.iloc[test_idx].copy()
    y_train, y_test = y_bin.iloc[train_idx].copy(), y_bin.iloc[test_idx].copy()

    pipeline = Pipeline(
        steps=[
            ("preprocess", build_binary_preprocessor(numeric_features, categorical_features)),
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
    short_idx = list(pipeline.classes_).index("short")
    short_prob = pipeline.predict_proba(X_test)[:, short_idx]
    return y_test, short_prob


def make_binary_quintile_lift_plot() -> Path:
    y_test, short_prob = _fit_binary_random_forest()
    risk_df = pd.DataFrame(
        {
            "short_prob": short_prob,
            "observed_short": (y_test == "short").astype(int).values,
        }
    )
    risk_df["quintile"] = pd.qcut(risk_df["short_prob"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    summary = (
        risk_df.groupby("quintile", observed=False)
        .agg(
            predicted_short_prob=("short_prob", "mean"),
            observed_short_rate=("observed_short", "mean"),
            n=("observed_short", "size"),
        )
        .reset_index()
    )
    summary["predicted_short_pct"] = summary["predicted_short_prob"] * 100
    summary["observed_short_pct"] = summary["observed_short_rate"] * 100

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    ax.plot(
        summary["quintile"],
        summary["predicted_short_pct"],
        marker="o",
        linewidth=2.8,
        color="#1d3557",
        label="Predicted short-stay probability",
    )
    ax.plot(
        summary["quintile"],
        summary["observed_short_pct"],
        marker="s",
        linewidth=2.8,
        color="#e76f51",
        label="Observed short-stay rate",
    )
    for _, row in summary.iterrows():
        ax.text(
            row["quintile"],
            row["observed_short_pct"] + 2.3,
            f"n={int(row['n'])}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#555555",
        )

    ax.set_title("Observed Short-Stay Rates Rise Sharply Across Predicted Risk Groups", fontsize=15, pad=12)
    ax.set_xlabel("Predicted Short-Stay Probability Quintile", fontsize=12)
    ax.set_ylabel("Percent Short Stay", fontsize=12)
    ax.set_ylim(0, 100)
    ax.legend(frameon=True, loc="upper left")
    plt.tight_layout()

    out = OUTPUT_DIR / "binary_short_stay_quintile_lift.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        make_los_mix_by_careunit(),
        make_early_burden_heatmap(),
        make_binary_quintile_lift_plot(),
    ]
    for path in outputs:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
