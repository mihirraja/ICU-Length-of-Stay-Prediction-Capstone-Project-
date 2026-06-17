import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "output" / "user_insight_figures"
MPL_DIR = PROJECT_ROOT / ".mplconfig"
MPL_DIR.mkdir(exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPL_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_data() -> pd.DataFrame:
    stays = pd.read_csv(
        DATA_DIR / "icu_stays.csv",
        usecols=[
            "stay_id",
            "subject_id",
            "los",
            "first_careunit",
            "admission_type",
            "admission_location",
        ],
    )
    patients = pd.read_csv(DATA_DIR / "icu_patients.csv", usecols=["subject_id", "anchor_age", "gender"])
    rad = pd.read_csv(PROJECT_ROOT / "output" / "features" / "radiology_features_first24h.csv")

    df = stays.merge(patients, on="subject_id", how="left").merge(rad, on="stay_id", how="left")
    df["los"] = pd.to_numeric(df["los"], errors="coerce")
    df["remaining_los_after_24h"] = pd.to_numeric(df["remaining_los_after_24h"], errors="coerce")
    df["eligible_24h_model"] = df["los"] >= 1
    df["los_bucket"] = pd.cut(
        df["los"],
        bins=[0, 1, 3, 7, float("inf")],
        labels=["<1 day", "1-3 days", "3-7 days", "7+ days"],
        right=False,
    )
    df["age_bin"] = pd.cut(
        df["anchor_age"],
        bins=[0, 40, 60, 75, 200],
        labels=["<40", "40-59", "60-74", "75+"],
        right=False,
    )
    return df


def save_24h_eligibility(df: pd.DataFrame) -> None:
    summary = (
        df["eligible_24h_model"]
        .value_counts(dropna=False)
        .rename(index={True: ">=24h stay", False: "<24h stay"})
        .reset_index()
    )
    summary.columns = ["group", "count"]
    summary["pct"] = 100 * summary["count"] / summary["count"].sum()

    plt.figure(figsize=(7, 5))
    ax = sns.barplot(data=summary, x="group", y="count", palette=["#4c956c", "#f28482"])
    plt.title("Who Is Eligible for a First-24-Hour ICU LOS Model?")
    plt.xlabel("")
    plt.ylabel("Number of ICU stays")
    plt.xticks(rotation=0)
    for i, row in summary.iterrows():
        ax.text(i, row["count"], f'{row["pct"]:.1f}%', ha="center", va="bottom", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eligibility_for_24h_model.png", dpi=220)
    plt.close()


def save_remaining_los_distribution(df: pd.DataFrame) -> None:
    plot_df = df[df["eligible_24h_model"]].copy()

    plt.figure(figsize=(9, 5))
    sns.histplot(plot_df.loc[plot_df["remaining_los_after_24h"] <= 15, "remaining_los_after_24h"], bins=40, color="#2a6f97")
    plt.title("Remaining ICU LOS After the First 24 Hours")
    plt.xlabel("Remaining ICU LOS (days, trimmed to 15)")
    plt.ylabel("Number of ICU stays")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "remaining_los_distribution.png", dpi=220)
    plt.close()


def save_radiology_signal_by_los_bucket(df: pd.DataFrame) -> None:
    plot_df = df[df["eligible_24h_model"]].copy()
    features = {
        "has_rad_24h": "Any radiology note",
        "rad_ct_24h": "CT",
        "rad_xray_24h": "X-ray",
        "kw_intubation_24h": "Intubation",
        "kw_effusion_24h": "Effusion",
        "kw_hemorrhage_24h": "Hemorrhage",
    }
    prevalence = (
        plot_df.groupby("los_bucket", observed=False)[list(features)]
        .mean()
        .mul(100)
        .T.rename(index=features)
    )

    plt.figure(figsize=(8.5, 6))
    sns.heatmap(prevalence, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=0.5)
    plt.title("First-Day Radiology Signals Become More Common in Longer ICU Stays")
    plt.xlabel("ICU LOS bucket")
    plt.ylabel("")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "radiology_signal_by_los_bucket.png", dpi=220)
    plt.close()


def save_remaining_los_by_key_findings(df: pd.DataFrame) -> None:
    plot_df = df[df["eligible_24h_model"]].copy()
    flags = {
        "kw_intubation_24h": "Intubation",
        "kw_effusion_24h": "Effusion",
        "kw_hemorrhage_24h": "Hemorrhage",
        "rad_ct_24h": "CT",
    }
    rows = []
    for col, label in flags.items():
        temp = plot_df[[col, "remaining_los_after_24h"]].copy()
        temp["finding"] = label
        temp["status"] = temp[col].map({0: "Absent", 1: "Present"})
        rows.append(temp[["finding", "status", "remaining_los_after_24h"]])
    long_df = pd.concat(rows, ignore_index=True)

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=long_df,
        x="remaining_los_after_24h",
        y="finding",
        hue="status",
        showfliers=False,
        palette=["#adb5bd", "#e76f51"],
    )
    plt.title("Key First-Day Radiology Findings and Remaining ICU LOS")
    plt.xlabel("Remaining ICU LOS after hour 24 (days, trimmed to 15)")
    plt.ylabel("")
    plt.xlim(0, 15)
    plt.legend(title="", loc="lower right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "remaining_los_by_key_findings.png", dpi=220)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    df = load_data()
    save_24h_eligibility(df)
    save_remaining_los_distribution(df)
    save_radiology_signal_by_los_bucket(df)
    save_remaining_los_by_key_findings(df)

    print("Saved user insight figures to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
