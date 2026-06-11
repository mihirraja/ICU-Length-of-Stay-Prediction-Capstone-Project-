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


OUTPUT_DIR = PROJECT_DIR / "output" / "figures"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    stays = pd.read_csv(DATA_DIR / "icu_stays.csv")
    patients = pd.read_csv(DATA_DIR / "icu_patients.csv")
    return stays, patients


def prepare_data(stays: pd.DataFrame, patients: pd.DataFrame) -> pd.DataFrame:
    df = stays.merge(patients, on="subject_id", how="left")
    df["los"] = pd.to_numeric(df["los"], errors="coerce")

    df["age_bin"] = pd.cut(
        df["anchor_age"],
        bins=[0, 40, 60, 75, 200],
        labels=["<40", "40-59", "60-74", "75+"],
        right=False,
    )

    df["los_bucket"] = pd.cut(
        df["los"],
        bins=[0, 1, 3, 7, float("inf")],
        labels=["<1 day", "1-3 days", "3-7 days", "7+ days"],
        right=False,
    )
    return df


def save_los_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    sns.histplot(df.loc[df["los"] <= 15, "los"], bins=50, color="#2a6f97")
    plt.title("ICU Length of Stay Distribution (trimmed to 15 days)")
    plt.xlabel("ICU LOS (days)")
    plt.ylabel("Number of stays")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "los_distribution.png", dpi=200)
    plt.close()


def save_los_by_careunit(df: pd.DataFrame) -> None:
    top_units = df["first_careunit"].value_counts().head(8).index
    plot_df = df[df["first_careunit"].isin(top_units)].copy()

    order = (
        plot_df.groupby("first_careunit")["los"]
        .median()
        .sort_values(ascending=False)
        .index
    )

    plt.figure(figsize=(12, 7))
    sns.boxplot(
        data=plot_df,
        x="los",
        y="first_careunit",
        order=order,
        showfliers=False,
        color="#90be6d",
    )
    plt.xlim(0, 15)
    plt.title("ICU LOS by First Care Unit")
    plt.xlabel("ICU LOS (days, trimmed to 15 on axis)")
    plt.ylabel("First care unit")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "los_by_careunit.png", dpi=200)
    plt.close()


def save_los_by_admission_type(df: pd.DataFrame) -> None:
    summary = (
        df.groupby("admission_type", dropna=False)["los"]
        .median()
        .sort_values(ascending=False)
        .reset_index()
    )

    plt.figure(figsize=(11, 6))
    sns.barplot(data=summary, x="los", y="admission_type", color="#f4a261")
    plt.title("Median ICU LOS by Admission Type")
    plt.xlabel("Median ICU LOS (days)")
    plt.ylabel("Admission type")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "median_los_by_admission_type.png", dpi=200)
    plt.close()


def save_los_by_age_bin(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="age_bin",
        y="los",
        showfliers=False,
        color="#e76f51",
        order=["<40", "40-59", "60-74", "75+"],
    )
    plt.ylim(0, 15)
    plt.title("ICU LOS by Age Group")
    plt.xlabel("Age group")
    plt.ylabel("ICU LOS (days, trimmed to 15 on axis)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "los_by_age_group.png", dpi=200)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    stays, patients = load_data()
    df = prepare_data(stays, patients)

    save_los_distribution(df)
    save_los_by_careunit(df)
    save_los_by_admission_type(df)
    save_los_by_age_bin(df)

    print("Saved figures to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
