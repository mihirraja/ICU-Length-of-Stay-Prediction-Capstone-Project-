from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MODELS_DIR = PROJECT_ROOT / "models"
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from models.run_binary_short_stay_classification import (  # noqa: E402
    build_preprocessor as build_binary_preprocessor,
    prepare_modeling_data as prepare_binary_modeling_data,
)


OUTPUT_DIR = PROJECT_ROOT / "output" / "model_comparison_figures"
CLASSIFICATION_OUTPUT_DIR = PROJECT_ROOT / "output" / "classification_results"


def make_binary_roc_curve() -> Path:
    X, y, groups = prepare_binary_modeling_data()
    text_and_cat = {
        "rad_text_24h",
        "gender",
        "first_careunit",
        "admission_type",
        "admission_location",
        "insurance",
        "language",
        "marital_status",
        "race",
    }
    numeric_features = [c for c in X.columns if c not in text_and_cat]
    categorical_features = [
        c
        for c in [
            "gender",
            "first_careunit",
            "admission_type",
            "admission_location",
            "insurance",
            "language",
            "marital_status",
            "race",
        ]
        if c in X.columns
    ]

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
    y_test_bin = (y_test == "short").astype(int)

    preprocessor = build_binary_preprocessor(numeric_features, categorical_features)
    models = [
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=8,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
            "#1d3557",
        ),
        (
            "Logistic Regression",
            LogisticRegression(max_iter=3000, class_weight="balanced"),
            "#457b9d",
        ),
        (
            "Dummy Most Frequent",
            DummyClassifier(strategy="most_frequent"),
            "#b0b7c3",
        ),
    ]

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    for model_name, estimator, color in models:
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)

        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(X_test)
            short_idx = list(pipeline.classes_).index("short")
            scores = proba[:, short_idx]
        else:
            scores = (pipeline.predict(X_test) == "short").astype(int)

        fpr, tpr, _ = roc_curve(y_test_bin, scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2.5, color=color, label=f"{model_name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, color="#7f8c8d")
    ax.set_title("Binary Short-Stay ROC Curve", fontsize=15, pad=12)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.legend(frameon=True, loc="lower right")
    plt.tight_layout()

    output_path = OUTPUT_DIR / "binary_roc_curve.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def _load_confusion_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    return df


def make_confusion_heatmap(input_path: Path, output_path: Path, title: str, label_map: dict[str, str]) -> Path:
    df = _load_confusion_csv(input_path)
    df.index = [label_map.get(idx, idx) for idx in df.index]
    df.columns = [label_map.get(col, col) for col in df.columns]

    row_sums = df.sum(axis=1).replace(0, 1)
    norm = df.div(row_sums, axis=0)
    annotations = df.astype(int).astype(str) + "\n" + norm.mul(100).round(1).astype(str) + "%"

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    sns.heatmap(
        norm,
        annot=annotations,
        fmt="",
        cmap="Blues",
        cbar_kws={"label": "Row-Normalized Proportion"},
        linewidths=0.75,
        linecolor="white",
        ax=ax,
        vmin=0,
        vmax=1,
    )
    ax.set_title(title, fontsize=15, pad=12)
    ax.set_xlabel("Predicted Class", fontsize=12)
    ax.set_ylabel("True Class", fontsize=12)
    plt.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    roc_path = make_binary_roc_curve()
    binary_cm_path = make_confusion_heatmap(
        CLASSIFICATION_OUTPUT_DIR / "random_forest_binary_short_vs_longer_confusion_matrix.csv",
        OUTPUT_DIR / "binary_confusion_matrix_heatmap.png",
        "Binary Random Forest Confusion Matrix",
        {
            "actual_short": "True Short",
            "actual_longer": "True Longer",
            "pred_short": "Pred Short",
            "pred_longer": "Pred Longer",
        },
    )
    multiclass_cm_path = make_confusion_heatmap(
        CLASSIFICATION_OUTPUT_DIR / "random_forest_confusion_matrix.csv",
        OUTPUT_DIR / "three_class_confusion_matrix_heatmap.png",
        "Three-Class Random Forest Confusion Matrix",
        {
            "actual_short": "True Short",
            "actual_medium": "True Medium",
            "actual_long": "True Long",
            "pred_short": "Pred Short",
            "pred_medium": "Pred Medium",
            "pred_long": "Pred Long",
        },
    )

    print(f"Saved binary ROC curve to: {roc_path}")
    print(f"Saved binary confusion matrix heatmap to: {binary_cm_path}")
    print(f"Saved three-class confusion matrix heatmap to: {multiclass_cm_path}")


if __name__ == "__main__":
    main()
