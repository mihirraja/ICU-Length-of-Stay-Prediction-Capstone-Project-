from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SURVIVAL_DIR = PROJECT_ROOT / "output" / "survival_results"


def clean_feature_name(name: str) -> str:
    cleaned = name.replace("num__", "").replace("cat__", "")
    cleaned = cleaned.replace("first_careunit_", "care unit: ")
    cleaned = cleaned.replace("admission_type_", "admission: ")
    cleaned = cleaned.replace("admission_location_", "location: ")
    cleaned = cleaned.replace("input_", "")
    cleaned = cleaned.replace("lab_", "")
    cleaned = cleaned.replace("kw_", "")
    cleaned = cleaned.replace("_24h", " (24h)")
    cleaned = cleaned.replace("_", " ")
    cleaned = cleaned.replace("rad note count", "radiology note count")
    cleaned = cleaned.replace("resp rate", "respiratory rate")
    cleaned = cleaned.replace("iv", "IV")
    return cleaned


def main() -> None:
    summary_path = SURVIVAL_DIR / "cox_feature_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing {summary_path}. Run models/run_cox_survival_model.py first."
        )

    df = pd.read_csv(summary_path)
    df = df[df["p"] < 0.001].copy()

    longer = df.nsmallest(8, "exp(coef)").copy()
    shorter = df.nlargest(8, "exp(coef)").copy()
    plot_df = pd.concat([longer, shorter], ignore_index=True).drop_duplicates("covariate")
    plot_df["feature_label"] = plot_df["covariate"].map(clean_feature_name)
    plot_df["direction"] = plot_df["exp(coef)"].apply(
        lambda x: "Associated with longer ICU stay" if x < 1 else "Associated with shorter ICU stay"
    )
    plot_df = plot_df.sort_values("exp(coef)")

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(11, 7))
    palette = {
        "Associated with longer ICU stay": "#d1495b",
        "Associated with shorter ICU stay": "#2a9d8f",
    }

    ax.hlines(
        y=plot_df["feature_label"],
        xmin=plot_df["exp(coef) lower 95%"],
        xmax=plot_df["exp(coef) upper 95%"],
        color="#9aa0a6",
        linewidth=2,
        zorder=1,
    )
    ax.scatter(
        plot_df["exp(coef)"],
        plot_df["feature_label"],
        c=plot_df["direction"].map(palette),
        s=80,
        zorder=2,
    )
    ax.axvline(1.0, color="#444444", linestyle="--", linewidth=1.5)

    ax.set_title("Top Cox Survival Features for ICU Discharge Timing", fontsize=15, pad=14)
    ax.set_xlabel("Hazard Ratio for ICU Discharge", fontsize=12)
    ax.set_ylabel("")

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=label, markerfacecolor=color, markersize=9)
        for label, color in palette.items()
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True)

    plt.tight_layout()
    out_path = SURVIVAL_DIR / "cox_top_feature_hazard_ratios.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"Saved figure to: {out_path}")


if __name__ == "__main__":
    main()
