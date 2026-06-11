from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SURVIVAL_DIR = PROJECT_ROOT / "output" / "survival_results"
OUTPUT_DIR = PROJECT_ROOT / "output" / "model_comparison_figures"


def clean_label(name: str) -> str:
    cleaned = name
    cleaned = cleaned.replace("cat__", "")
    cleaned = cleaned.replace("num__", "")
    cleaned = cleaned.replace("first_careunit_", "")
    cleaned = cleaned.replace("admission_type_", "Admission: ")
    cleaned = cleaned.replace("output_item_226588_count_24h", "Urine output count item 226588")
    cleaned = cleaned.replace("output_item_226559_ml_24h", "Urine output volume item 226559")
    cleaned = cleaned.replace("input_oral_gastric_ml_24h", "Oral / gastric intake")
    cleaned = cleaned.replace("rad_note_count_24h", "Radiology note count")
    cleaned = cleaned.replace("lab_lactate_count", "Lactate count")
    cleaned = cleaned.replace("resp_rate_mean", "Respiratory rate mean")
    cleaned = cleaned.replace("input_any_enteral_nutrition_24h", "Any enteral nutrition")
    cleaned = cleaned.replace("rx_mean_doses_per_24h", "Mean prescription doses")
    cleaned = cleaned.replace("input_crystalloid_ml_24h", "Crystalloid volume")
    cleaned = cleaned.replace("input_any_med_bolus_24h", "Any medication bolus")
    cleaned = cleaned.replace("input_any_drip_24h", "Any drip")
    cleaned = cleaned.replace("input_unique_category_count_24h", "Distinct input categories")
    cleaned = cleaned.replace("input_colloid_ml_24h", "Colloid volume")
    cleaned = cleaned.replace("input_any_blood_product_24h", "Any blood product")
    cleaned = cleaned.replace("input_insulin_noniv_count_24h", "Non-IV insulin count")
    cleaned = cleaned.replace("rx_route_iv_count_24h", "IV prescription route count")
    cleaned = cleaned.replace("rx_unique_route_count_24h", "Distinct prescription routes")
    cleaned = cleaned.replace("_24h", " (24h)")
    cleaned = cleaned.replace("_", " ")
    return cleaned


def main() -> None:
    top_features = pd.read_csv(SURVIVAL_DIR / "cox_top_features.csv")
    metrics = pd.read_csv(SURVIVAL_DIR / "cox_model_metrics.csv").iloc[0]

    low_hr = top_features[top_features["hazard_ratio"] < 1].nsmallest(6, "hazard_ratio").copy()
    high_hr = top_features[top_features["hazard_ratio"] > 1].nlargest(6, "hazard_ratio").copy()
    plot_df = pd.concat([low_hr, high_hr], ignore_index=True)
    plot_df["feature_label"] = plot_df["feature"].map(clean_label)
    plot_df["direction"] = plot_df["hazard_ratio"].apply(
        lambda x: "Longer ICU stay" if x < 1 else "Faster ICU discharge"
    )
    plot_df = plot_df.sort_values("hazard_ratio", ascending=True)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    palette = {"Longer ICU stay": "#d1495b", "Faster ICU discharge": "#2a9d8f"}
    sns.scatterplot(
        data=plot_df,
        x="hazard_ratio",
        y="feature_label",
        hue="direction",
        palette=palette,
        s=120,
        ax=ax,
    )
    for _, row in plot_df.iterrows():
        ax.hlines(
            y=row["feature_label"],
            xmin=1.0,
            xmax=row["hazard_ratio"],
            color=palette[row["direction"]],
            linewidth=2,
            alpha=0.8,
        )

    ax.axvline(1.0, linestyle="--", color="#6c757d", linewidth=1.5)
    ax.set_title(
        f"Cox Survival Model: Most Informative Early Predictors\n(Test c-index = {metrics['test_concordance_index']:.3f})",
        fontsize=15,
        pad=12,
    )
    ax.set_xlabel("Hazard Ratio for ICU Discharge", fontsize=12)
    ax.set_ylabel("")
    ax.legend(title="", frameon=True, loc="lower right")
    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "cox_informative_feature_summary.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved Cox informative feature summary to: {output_path}")


if __name__ == "__main__":
    main()
