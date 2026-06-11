from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASSIFICATION_DIR = PROJECT_ROOT / "output" / "classification_results"
SURVIVAL_DIR = PROJECT_ROOT / "output" / "survival_results"
OUTPUT_DIR = PROJECT_ROOT / "output" / "model_comparison_figures"


def save_binary_comparison() -> None:
    df = pd.read_csv(CLASSIFICATION_DIR / "binary_short_vs_longer_results.csv")
    metrics = ["accuracy", "f1_short", "roc_auc_short"]
    plot_df = df[["model"] + metrics].melt(
        id_vars="model", var_name="metric", value_name="value"
    )
    plot_df["metric"] = plot_df["metric"].map(
        {
            "accuracy": "Accuracy",
            "f1_short": "Short-Stay F1",
            "roc_auc_short": "ROC AUC",
        }
    )

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=plot_df, x="metric", y="value", hue="model", ax=ax)
    ax.set_title("Binary Short-vs-Longer Stay: Model Comparison", fontsize=15, pad=14)
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.legend(title="")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "binary_model_comparison.png", dpi=220)
    plt.close(fig)


def save_multiclass_comparison() -> None:
    df = pd.read_csv(CLASSIFICATION_DIR / "los_classification_results.csv")
    metrics = ["accuracy", "macro_f1", "weighted_f1"]
    plot_df = df[["model"] + metrics].melt(
        id_vars="model", var_name="metric", value_name="value"
    )
    plot_df["metric"] = plot_df["metric"].map(
        {
            "accuracy": "Accuracy",
            "macro_f1": "Macro F1",
            "weighted_f1": "Weighted F1",
        }
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=plot_df, x="metric", y="value", hue="model", ax=ax)
    ax.set_title("Three-Class LOS Prediction: Model Comparison", fontsize=15, pad=14)
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.legend(title="")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "multiclass_model_comparison.png", dpi=220)
    plt.close(fig)


def save_binary_precision_recall() -> None:
    df = pd.read_csv(CLASSIFICATION_DIR / "binary_short_vs_longer_results.csv")
    df = df[df["model"] != "Dummy Most Frequent"].copy()

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(
        data=df,
        x="recall_short",
        y="precision_short",
        hue="model",
        s=160,
        ax=ax,
    )
    for _, row in df.iterrows():
        ax.text(
            row["recall_short"] + 0.002,
            row["precision_short"] + 0.002,
            row["model"],
            fontsize=10,
        )
    ax.set_xlim(0.68, 0.78)
    ax.set_ylim(0.70, 0.78)
    ax.set_xlabel("Recall for Short Stays")
    ax.set_ylabel("Precision for Short Stays")
    ax.set_title("Binary Model Tradeoff for Short-Stay Detection", fontsize=15, pad=14)
    ax.legend().remove()
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "binary_precision_recall_tradeoff.png", dpi=220)
    plt.close(fig)


def save_cox_panel() -> None:
    df = pd.read_csv(SURVIVAL_DIR / "cox_model_metrics.csv")
    plot_df = pd.DataFrame(
        {
            "split": ["Train", "Test"],
            "c_index": [
                df.loc[0, "train_concordance_index"],
                df.loc[0, "test_concordance_index"],
            ],
        }
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(data=plot_df, x="split", y="c_index", color="#457b9d", ax=ax)
    ax.set_ylim(0.5, 0.8)
    ax.set_ylabel("Concordance Index")
    ax.set_xlabel("")
    ax.set_title("Cox Survival Model Performance", fontsize=15, pad=14)
    for idx, row in plot_df.iterrows():
        ax.text(idx, row["c_index"] + 0.01, f'{row["c_index"]:.3f}', ha="center")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "cox_model_performance.png", dpi=220)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_binary_comparison()
    save_multiclass_comparison()
    save_binary_precision_recall()
    save_cox_panel()
    print(f"Saved model comparison figures to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
