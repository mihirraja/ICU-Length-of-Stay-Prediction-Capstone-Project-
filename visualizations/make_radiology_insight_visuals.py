import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROJECT_DIR = PROJECT_ROOT
MPL_DIR = PROJECT_DIR / ".mplconfig"
MPL_DIR.mkdir(exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPL_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


OUTPUT_DIR = PROJECT_DIR / "output" / "radiology_insight_figures"


FLAG_LABELS = {
    "kw_effusion_24h": "Effusion",
    "kw_line_24h": "Line/Tube",
    "kw_edema_24h": "Edema",
    "kw_pneumonia_24h": "Pneumonia",
    "rad_ct_24h": "CT",
    "kw_intubation_24h": "Intubation",
    "kw_hemorrhage_24h": "Hemorrhage",
    "kw_stroke_24h": "Stroke/Infarct",
    "kw_fracture_24h": "Fracture",
    "kw_postop_24h": "Post-op",
    "rad_xray_24h": "X-ray",
    "rad_ultrasound_24h": "Ultrasound",
    "rad_mri_24h": "MRI",
}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(PROJECT_ROOT / "output" / "features" / "radiology_features_first24h.csv")
    stays = pd.read_csv(
        DATA_DIR / "icu_stays.csv",
        usecols=["stay_id", "first_careunit", "admission_type", "admission_location", "los"],
    )
    df = df.merge(stays, on="stay_id", how="left")
    df["remaining_los_after_24h"] = pd.to_numeric(df["remaining_los_after_24h"], errors="coerce")
    df["los"] = pd.to_numeric(df["los"], errors="coerce")
    df["los_bucket"] = pd.cut(
        df["los"],
        bins=[0, 1, 3, 7, float("inf")],
        labels=["<1 day", "1-3 days", "3-7 days", "7+ days"],
        right=False,
    )
    return df


def save_flag_prevalence_by_los(df: pd.DataFrame) -> None:
    flags = list(FLAG_LABELS.keys())
    prevalence = (
        df.groupby("los_bucket", observed=False)[flags]
        .mean()
        .mul(100)
        .T
        .rename(index=FLAG_LABELS)
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(prevalence, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=0.5)
    plt.title("First-24h Radiology Flag Prevalence by ICU LOS Bucket")
    plt.xlabel("ICU LOS bucket")
    plt.ylabel("Radiology feature")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "flag_prevalence_by_los_bucket.png", dpi=220)
    plt.close()


def save_remaining_los_by_major_flags(df: pd.DataFrame) -> None:
    flags = [
        "kw_intubation_24h",
        "kw_hemorrhage_24h",
        "kw_effusion_24h",
        "kw_pneumonia_24h",
        "rad_ct_24h",
        "rad_xray_24h",
    ]

    rows = []
    for flag in flags:
        temp = df[[flag, "remaining_los_after_24h"]].copy()
        temp["feature"] = FLAG_LABELS[flag]
        temp["present"] = temp[flag].map({0: "Absent", 1: "Present"})
        rows.append(temp[["feature", "present", "remaining_los_after_24h"]])

    plot_df = pd.concat(rows, ignore_index=True)

    plt.figure(figsize=(12, 7))
    sns.boxplot(
        data=plot_df,
        x="remaining_los_after_24h",
        y="feature",
        hue="present",
        showfliers=False,
    )
    plt.xlim(0, 15)
    plt.title("Remaining ICU LOS After 24h by Major Radiology Flags")
    plt.xlabel("Remaining ICU LOS after hour 24 (days, trimmed to 15 on axis)")
    plt.ylabel("Radiology feature")
    plt.legend(title="", loc="lower right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "remaining_los_by_major_flags.png", dpi=220)
    plt.close()


def save_note_count_vs_remaining_los(df: pd.DataFrame) -> None:
    note_bins = pd.cut(
        df["rad_note_count_24h"],
        bins=[-1, 0, 1, 2, 100],
        labels=["0", "1", "2", "3+"],
    )
    summary = (
        df.assign(note_count_bin=note_bins)
        .groupby("note_count_bin", observed=False)["remaining_los_after_24h"]
        .agg(["median", "mean", "count"])
        .reset_index()
    )

    plt.figure(figsize=(8, 5))
    sns.barplot(data=summary, x="note_count_bin", y="median", color="#4c956c")
    plt.title("Median Remaining ICU LOS by Number of Radiology Notes in First 24h")
    plt.xlabel("Radiology notes in first 24h")
    plt.ylabel("Median remaining ICU LOS (days)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "median_remaining_los_by_note_count.png", dpi=220)
    plt.close()


def save_flag_prevalence_by_careunit(df: pd.DataFrame) -> None:
    top_units = df["first_careunit"].value_counts().head(6).index
    flags = [
        "kw_effusion_24h",
        "kw_pneumonia_24h",
        "kw_intubation_24h",
        "kw_hemorrhage_24h",
        "rad_ct_24h",
        "rad_xray_24h",
    ]
    prevalence = (
        df[df["first_careunit"].isin(top_units)]
        .groupby("first_careunit")[flags]
        .mean()
        .mul(100)
        .rename(columns=FLAG_LABELS)
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(prevalence, annot=True, fmt=".1f", cmap="mako", linewidths=0.5)
    plt.title("Radiology Feature Prevalence by First Care Unit")
    plt.xlabel("Radiology feature")
    plt.ylabel("First care unit")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "flag_prevalence_by_careunit.png", dpi=220)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    df = load_data()
    save_flag_prevalence_by_los(df)
    save_remaining_los_by_major_flags(df)
    save_note_count_vs_remaining_los(df)
    save_flag_prevalence_by_careunit(df)

    print("Saved radiology insight figures to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
