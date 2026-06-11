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
import numpy as np
import pandas as pd
import seaborn as sns


OUTPUT_DIR = PROJECT_DIR / "output" / "advanced_figures"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stays = pd.read_csv(DATA_DIR / "icu_stays.csv")
    patients = pd.read_csv(DATA_DIR / "icu_patients.csv")
    notes = pd.read_csv(
        DATA_DIR / "icu_radiology_notes.csv",
        usecols=["stay_id", "charttime"],
    )
    return stays, patients, notes


def prepare_base(stays: pd.DataFrame, patients: pd.DataFrame) -> pd.DataFrame:
    df = stays.merge(patients, on="subject_id", how="left")
    df["los"] = pd.to_numeric(df["los"], errors="coerce")
    df["intime"] = pd.to_datetime(df["intime"], errors="coerce")
    df["outtime"] = pd.to_datetime(df["outtime"], errors="coerce")
    df["short_stay"] = df["los"] < 1
    df["los_bucket"] = pd.cut(
        df["los"],
        bins=[0, 1, 3, 7, np.inf],
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


def add_first_day_note_flag(df: pd.DataFrame, notes: pd.DataFrame) -> pd.DataFrame:
    note_df = notes.dropna(subset=["stay_id", "charttime"]).copy()
    note_df["stay_id"] = pd.to_numeric(note_df["stay_id"], errors="coerce")
    note_df["charttime"] = pd.to_datetime(note_df["charttime"], errors="coerce")
    note_df = note_df.dropna(subset=["stay_id", "charttime"])

    joined = note_df.merge(df[["stay_id", "intime"]], on="stay_id", how="inner")
    joined["within_24h"] = (
        (joined["charttime"] >= joined["intime"])
        & (joined["charttime"] < joined["intime"] + pd.Timedelta(hours=24))
    )

    flags = (
        joined.groupby("stay_id", as_index=False)["within_24h"]
        .max()
        .rename(columns={"within_24h": "has_rad_24h"})
    )

    out = df.merge(flags, on="stay_id", how="left")
    out["has_rad_24h"] = out["has_rad_24h"].fillna(False)
    return out


def save_heatmap_median_los(df: pd.DataFrame) -> None:
    top_units = df["first_careunit"].value_counts().head(6).index
    top_types = df["admission_type"].value_counts().head(6).index
    sub = df[df["first_careunit"].isin(top_units) & df["admission_type"].isin(top_types)].copy()

    counts = sub.pivot_table(
        index="first_careunit",
        columns="admission_type",
        values="los",
        aggfunc="size",
    )
    medians = sub.pivot_table(
        index="first_careunit",
        columns="admission_type",
        values="los",
        aggfunc="median",
    )
    medians = medians.where(counts >= 75)

    plt.figure(figsize=(12, 7))
    sns.heatmap(medians, annot=True, fmt=".2f", cmap="YlOrRd", linewidths=0.5)
    plt.title("Median ICU LOS by First Care Unit and Admission Type")
    plt.xlabel("Admission type")
    plt.ylabel("First care unit")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "heatmap_median_los_unit_vs_admission_type.png", dpi=220)
    plt.close()


def save_heatmap_short_stay_rate(df: pd.DataFrame) -> None:
    top_units = df["first_careunit"].value_counts().head(6).index
    top_locations = df["admission_location"].value_counts().head(6).index
    sub = df[
        df["first_careunit"].isin(top_units) & df["admission_location"].isin(top_locations)
    ].copy()

    counts = sub.pivot_table(
        index="first_careunit",
        columns="admission_location",
        values="short_stay",
        aggfunc="size",
    )
    short_rate = (
        sub.pivot_table(
            index="first_careunit",
            columns="admission_location",
            values="short_stay",
            aggfunc="mean",
        )
        * 100
    )
    short_rate = short_rate.where(counts >= 75)

    plt.figure(figsize=(13, 7))
    sns.heatmap(short_rate, annot=True, fmt=".1f", cmap="Blues", linewidths=0.5)
    plt.title("Percent of ICU Stays Under 24 Hours by Unit and Admission Location")
    plt.xlabel("Admission location")
    plt.ylabel("First care unit")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "heatmap_short_stay_rate_unit_vs_location.png", dpi=220)
    plt.close()


def save_survival_style_curves(df: pd.DataFrame) -> None:
    top_units = df["first_careunit"].value_counts().head(5).index

    plt.figure(figsize=(11, 7))
    for unit in top_units:
        vals = np.sort(df.loc[df["first_careunit"] == unit, "los"].dropna().to_numpy())
        vals = vals[vals <= 15]
        if len(vals) == 0:
            continue
        survival = 1 - (np.arange(len(vals)) + 1) / len(vals)
        plt.step(vals, survival, where="post", label=unit, linewidth=2)

    plt.title("Probability Patient Is Still in ICU by Day")
    plt.xlabel("ICU day")
    plt.ylabel("Estimated probability still in ICU")
    plt.ylim(0, 1)
    plt.xlim(0, 15)
    plt.xticks(rotation=0)
    plt.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "survival_style_los_by_careunit.png", dpi=220)
    plt.close()


def save_radiology_coverage(df: pd.DataFrame) -> None:
    top_units = df["first_careunit"].value_counts().head(6).index
    sub = df[df["first_careunit"].isin(top_units)].copy()

    summary = (
        sub.groupby(["first_careunit", "los_bucket"], observed=False)["has_rad_24h"]
        .mean()
        .mul(100)
        .reset_index()
    )

    plt.figure(figsize=(12, 7))
    sns.barplot(
        data=summary,
        x="los_bucket",
        y="has_rad_24h",
        hue="first_careunit",
        palette="Set2",
    )
    plt.title("First-24-Hour Radiology Note Coverage by LOS Bucket")
    plt.xlabel("LOS bucket")
    plt.ylabel("Percent with radiology note in first 24h")
    plt.xticks(rotation=0)
    plt.legend(title="First care unit", fontsize=8, title_fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "radiology_coverage_by_los_bucket_and_unit.png", dpi=220)
    plt.close()


def save_age_vs_remaining_los(df: pd.DataFrame) -> None:
    sub = df[df["los"] >= 1].copy()
    sub["remaining_los_after_24h"] = sub["los"] - 1

    plt.figure(figsize=(10, 6))
    sns.violinplot(
        data=sub,
        x="age_bin",
        y="remaining_los_after_24h",
        order=["<40", "40-59", "60-74", "75+"],
        cut=0,
        inner="quartile",
        color="#cdb4db",
    )
    plt.ylim(0, 15)
    plt.title("Remaining ICU LOS After First 24 Hours by Age Group")
    plt.xlabel("Age group")
    plt.ylabel("Remaining ICU LOS after hour 24 (days, trimmed to 15 on axis)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "remaining_los_after_24h_by_age_group.png", dpi=220)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    stays, patients, notes = load_data()
    df = prepare_base(stays, patients)
    df = add_first_day_note_flag(df, notes)

    save_heatmap_median_los(df)
    save_heatmap_short_stay_rate(df)
    save_survival_style_curves(df)
    save_radiology_coverage(df)
    save_age_vs_remaining_los(df)

    print("Saved advanced figures to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
