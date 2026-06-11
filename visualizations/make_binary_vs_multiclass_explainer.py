from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
CLASSIFICATION_DIR = PROJECT_ROOT / "output" / "classification_results"
OUTPUT_DIR = PROJECT_ROOT / "output" / "model_comparison_figures"


def main() -> None:
    stays = pd.read_csv(RAW_DATA_DIR / "icu_stays.csv")
    stays["los"] = pd.to_numeric(stays["los"], errors="coerce")
    stays = stays[stays["los"].notna()].copy()

    stays["binary_class"] = stays["los"].apply(lambda x: "Short (<=2d)" if x <= 2 else "Longer (>2d)")

    def make_three_way(x: float) -> str:
        if x < 2:
            return "Short (<2d)"
        if x <= 5:
            return "Medium (2-5d)"
        return "Long (>5d)"

    stays["three_way_class"] = stays["los"].apply(make_three_way)

    binary_dist = (
        stays["binary_class"].value_counts(normalize=True)
        .reindex(["Short (<=2d)", "Longer (>2d)"])
        .fillna(0)
        .mul(100)
        .reset_index()
    )
    binary_dist.columns = ["class", "percent"]

    three_dist = (
        stays["three_way_class"].value_counts(normalize=True)
        .reindex(["Short (<2d)", "Medium (2-5d)", "Long (>5d)"])
        .fillna(0)
        .mul(100)
        .reset_index()
    )
    three_dist.columns = ["class", "percent"]

    binary_results = pd.read_csv(CLASSIFICATION_DIR / "binary_short_vs_longer_results.csv")
    multiclass_results = pd.read_csv(CLASSIFICATION_DIR / "los_classification_results.csv")

    binary_best = binary_results.loc[binary_results["roc_auc_short"].idxmax()]
    multiclass_best = multiclass_results.loc[multiclass_results["macro_f1"].idxmax()]

    performance_df = pd.DataFrame(
        [
            {
                "task": "Binary short vs longer",
                "metric": "Accuracy",
                "score": binary_best["accuracy"],
            },
            {
                "task": "Binary short vs longer",
                "metric": "Macro F1",
                "score": binary_best["macro_f1"],
            },
            {
                "task": "Three-way LOS",
                "metric": "Accuracy",
                "score": multiclass_best["accuracy"],
            },
            {
                "task": "Three-way LOS",
                "metric": "Macro F1",
                "score": multiclass_best["macro_f1"],
            },
        ]
    )

    sns.set_theme(style="whitegrid")

    fig, left = plt.subplots(figsize=(8, 5.5))
    left.barh(["Binary task"], [binary_dist.loc[0, "percent"]], color="#2a9d8f", label="Short (<=2d)")
    left.barh(
        ["Binary task"],
        [binary_dist.loc[1, "percent"]],
        left=[binary_dist.loc[0, "percent"]],
        color="#457b9d",
        label="Longer (>2d)",
    )
    left.barh(["Three-way task"], [three_dist.loc[0, "percent"]], color="#2a9d8f", label="Short (<2d)")
    left.barh(
        ["Three-way task"],
        [three_dist.loc[1, "percent"]],
        left=[three_dist.loc[0, "percent"]],
        color="#e9c46a",
        label="Medium (2-5d)",
    )
    left.barh(
        ["Three-way task"],
        [three_dist.loc[2, "percent"]],
        left=[three_dist.loc[0, "percent"] + three_dist.loc[1, "percent"]],
        color="#e76f51",
        label="Long (>5d)",
    )
    left.set_xlim(0, 100)
    left.set_xlabel("Share of ICU stays (%)")
    left.set_title("Binary vs Three-Way ICU LOS Target Setup", fontsize=15, pad=14)
    left.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path_target = OUTPUT_DIR / "binary_vs_multiclass_target_setup.png"
    fig.savefig(out_path_target, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, right = plt.subplots(figsize=(8, 5.5))
    sns.barplot(
        data=performance_df,
        x="metric",
        y="score",
        hue="task",
        palette=["#264653", "#8ab17d"],
        ax=right,
    )
    right.set_ylim(0, 1.0)
    right.set_ylabel("Score")
    right.set_xlabel("")
    right.set_title("Best Model Performance: Binary vs Three-Way LOS", fontsize=15, pad=14)
    right.legend(title="")
    plt.tight_layout()
    out_path_perf = OUTPUT_DIR / "binary_vs_multiclass_performance.png"
    fig.savefig(out_path_perf, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved target setup figure to: {out_path_target}")
    print(f"Saved performance figure to: {out_path_perf}")


if __name__ == "__main__":
    main()
