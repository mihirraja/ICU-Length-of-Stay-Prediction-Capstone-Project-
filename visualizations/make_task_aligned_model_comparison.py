from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASSIFICATION_DIR = PROJECT_ROOT / "output" / "classification_results"
SURVIVAL_DIR = PROJECT_ROOT / "output" / "survival_results"
OUTPUT_DIR = PROJECT_ROOT / "output" / "model_comparison_figures"


def main() -> None:
    binary = pd.read_csv(CLASSIFICATION_DIR / "binary_short_vs_longer_results.csv")
    multiclass = pd.read_csv(CLASSIFICATION_DIR / "los_classification_results.csv")
    cox = pd.read_csv(SURVIVAL_DIR / "cox_model_metrics.csv")

    binary_best = binary.loc[binary["roc_auc_short"].idxmax()]
    multiclass_best = multiclass.loc[multiclass["macro_f1"].idxmax()]
    cox_best = cox.iloc[0]

    comparison = pd.DataFrame(
        [
            {
                "task": "Binary short stay\n(<=2d vs >2d)",
                "model": binary_best["model"],
                "metric_name": "ROC AUC",
                "score": binary_best["roc_auc_short"],
                "detail": f'Acc {binary_best["accuracy"]:.3f} | F1 {binary_best["f1_short"]:.3f}',
            },
            {
                "task": "Three-class LOS\n(short/medium/long)",
                "model": multiclass_best["model"],
                "metric_name": "Macro F1",
                "score": multiclass_best["macro_f1"],
                "detail": f'Acc {multiclass_best["accuracy"]:.3f} | Weighted F1 {multiclass_best["weighted_f1"]:.3f}',
            },
            {
                "task": "Survival timing\n(time to ICU discharge)",
                "model": "Cox proportional hazards",
                "metric_name": "C-index",
                "score": cox_best["test_concordance_index"],
                "detail": f'Train c-index {cox_best["train_concordance_index"]:.3f}',
            },
        ]
    )

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = ["#2a9d8f", "#457b9d", "#e76f51"]
    sns.barplot(data=comparison, x="score", y="task", palette=palette, ax=ax)

    ax.set_title("Best Model by Prediction Task", fontsize=16, pad=14)
    ax.set_xlabel("Best Task-Aligned Score", fontsize=12)
    ax.set_ylabel("")
    ax.set_xlim(0, 1.0)

    for i, row in comparison.iterrows():
        ax.text(
            row["score"] + 0.015,
            i,
            f'{row["model"]}\n{row["metric_name"]}: {row["score"]:.3f}\n{row["detail"]}',
            va="center",
            fontsize=10,
        )

    note = (
        "Metrics differ by task: ROC AUC for binary classification, Macro F1 for 3-class\n"
        "classification, and concordance index for survival ranking."
    )
    fig.text(0.01, 0.01, note, fontsize=9)
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "task_aligned_model_comparison.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"Saved figure to: {out_path}")


if __name__ == "__main__":
    main()
