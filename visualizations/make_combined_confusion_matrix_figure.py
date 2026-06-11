from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASSIFICATION_OUTPUT_DIR = PROJECT_ROOT / "output" / "classification_results"
OUTPUT_DIR = PROJECT_ROOT / "output" / "model_comparison_figures"


def load_confusion(path: Path, label_map: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = pd.read_csv(path, index_col=0)
    counts.index = [label_map.get(idx, idx) for idx in counts.index]
    counts.columns = [label_map.get(col, col) for col in counts.columns]

    row_sums = counts.sum(axis=1).replace(0, 1)
    proportions = counts.div(row_sums, axis=0)
    return counts, proportions


def main() -> None:
    binary_counts, binary_props = load_confusion(
        CLASSIFICATION_OUTPUT_DIR / "random_forest_binary_short_vs_longer_confusion_matrix.csv",
        {
            "actual_short": "True Short",
            "actual_longer": "True Longer",
            "pred_short": "Pred Short",
            "pred_longer": "Pred Longer",
        },
    )
    multiclass_counts, multiclass_props = load_confusion(
        CLASSIFICATION_OUTPUT_DIR / "random_forest_confusion_matrix.csv",
        {
            "actual_short": "True Short",
            "actual_medium": "True Medium",
            "actual_long": "True Long",
            "pred_short": "Pred Short",
            "pred_medium": "Pred Medium",
            "pred_long": "Pred Long",
        },
    )

    binary_ann = binary_counts.astype(int).astype(str) + "\n" + binary_props.mul(100).round(1).astype(str) + "%"
    multiclass_ann = multiclass_counts.astype(int).astype(str) + "\n" + multiclass_props.mul(100).round(1).astype(str) + "%"

    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2))

    sns.heatmap(
        binary_props,
        annot=binary_ann,
        fmt="",
        cmap="Blues",
        linewidths=0.75,
        linecolor="white",
        cbar=False,
        vmin=0,
        vmax=1,
        ax=axes[0],
    )
    axes[0].set_title("Binary Random Forest\nShort (<=2d) vs Longer (>2d)", fontsize=14, pad=12)
    axes[0].set_xlabel("Predicted Class", fontsize=11)
    axes[0].set_ylabel("True Class", fontsize=11)

    hm = sns.heatmap(
        multiclass_props,
        annot=multiclass_ann,
        fmt="",
        cmap="Blues",
        linewidths=0.75,
        linecolor="white",
        cbar=True,
        cbar_kws={"label": "Row-Normalized Proportion"},
        vmin=0,
        vmax=1,
        ax=axes[1],
    )
    axes[1].set_title("Three-Class Random Forest\nShort / Medium / Long", fontsize=14, pad=12)
    axes[1].set_xlabel("Predicted Class", fontsize=11)
    axes[1].set_ylabel("")

    fig.suptitle("Confusion Matrix Comparison Across LOS Classification Tasks", fontsize=16, y=1.02)
    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "combined_confusion_matrix_comparison.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved combined confusion matrix figure to: {output_path}")


if __name__ == "__main__":
    main()
