from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from run_radiology_feature_models import build_feature_sets, prepare_modeling_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output" / "model_results"


def evaluate_predictions(y_true_days: pd.Series, y_pred_days: np.ndarray) -> dict[str, float]:
    y_pred_days = np.clip(y_pred_days, 0, None)
    return {
        "mae_days": mean_absolute_error(y_true_days, y_pred_days),
        "rmse_days": np.sqrt(mean_squared_error(y_true_days, y_pred_days)),
        "r2": r2_score(y_true_days, y_pred_days),
    }


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    include_text: bool,
) -> ColumnTransformer:
    transformers = [
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric_features,
        ),
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            categorical_features,
        ),
    ]

    if include_text:
        transformers.append(
            (
                "text",
                Pipeline(
                    steps=[
                        (
                            "tfidf",
                            TfidfVectorizer(
                                lowercase=True,
                                stop_words="english",
                                max_features=600,
                                ngram_range=(1, 2),
                                min_df=10,
                            ),
                        ),
                        ("svd", TruncatedSVD(n_components=40, random_state=42)),
                    ]
                ),
                "rad_text_24h",
            )
        )

    return ColumnTransformer(transformers=transformers)


def run_pipeline_model(
    model_name: str,
    estimator,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_log: pd.Series,
    y_test_days: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    include_text: bool = True,
) -> tuple[dict[str, float | str], np.ndarray]:
    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(numeric_features, categorical_features, include_text)),
            ("model", estimator),
        ]
    )
    pipeline.fit(X_train, y_train_log)
    pred_days = np.expm1(pipeline.predict(X_test))
    result = {"model": model_name, **evaluate_predictions(y_test_days, pred_days)}
    return result, pred_days


def run_dense_model(
    model_name: str,
    estimator,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_log: pd.Series,
    y_test_days: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[dict[str, float | str], np.ndarray]:
    preprocessor = build_preprocessor(numeric_features, categorical_features, include_text=True)
    X_train_transformed = preprocessor.fit_transform(X_train, y_train_log)
    X_test_transformed = preprocessor.transform(X_test)

    estimator.fit(X_train_transformed, y_train_log)
    pred_days = np.expm1(estimator.predict(X_test_transformed))
    result = {"model": model_name, **evaluate_predictions(y_test_days, pred_days)}
    return result, pred_days


def save_outputs(results_df: pd.DataFrame, predictions_df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_DIR / "enhanced_model_results.csv", index=False)
    predictions_df.to_csv(OUTPUT_DIR / "enhanced_model_predictions.csv", index=False)


def main() -> None:
    df, y_log = prepare_modeling_data()
    y_days = df["remaining_los_after_24h"]
    numeric_features, categorical_features, _ = build_feature_sets(df)

    X_train, X_test, y_train_log, _, y_train_days, y_test_days = train_test_split(
        df,
        y_log,
        y_days,
        test_size=0.2,
        random_state=42,
    )

    results: list[dict[str, float | str]] = []
    predictions = pd.DataFrame(
        {
            "stay_id": X_test["stay_id"].to_numpy(),
            "actual_remaining_los_days": y_test_days.to_numpy(),
        }
    )

    median_pred = np.repeat(y_train_days.median(), len(y_test_days))
    results.append({"model": "Median Baseline", **evaluate_predictions(y_test_days, median_pred)})
    predictions["pred_median_baseline"] = median_pred

    experiments = [
        (
            "Ridge + text",
            Ridge(alpha=2.0),
            "pipeline",
        ),
        (
            "ElasticNet + text",
            ElasticNet(alpha=0.0008, l1_ratio=0.2, max_iter=5000, random_state=42),
            "pipeline",
        ),
        (
            "HistGradientBoosting + text",
            HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.05,
                max_depth=3,
                max_iter=120,
                min_samples_leaf=60,
                l2_regularization=0.05,
                random_state=42,
            ),
            "dense",
        ),
    ]

    for model_name, estimator, kind in experiments:
        if kind == "pipeline":
            result, pred = run_pipeline_model(
                model_name,
                estimator,
                X_train,
                X_test,
                y_train_log,
                y_test_days,
                numeric_features,
                categorical_features,
            )
        else:
            result, pred = run_dense_model(
                model_name,
                estimator,
                X_train,
                X_test,
                y_train_log,
                y_test_days,
                numeric_features,
                categorical_features,
            )
        results.append(result)
        predictions[f"pred_{model_name.lower().replace(' ', '_').replace('+', 'plus')}"] = pred

    results_df = pd.DataFrame(results).sort_values(["r2", "mae_days"], ascending=[False, True]).reset_index(drop=True)
    save_outputs(results_df, predictions)

    print("\nCohort")
    print(f"Rows used: {len(df):,}")
    print("Target: remaining ICU LOS after first 24 hours (days)")
    print(f"Train rows: {len(X_train):,}")
    print(f"Test rows: {len(X_test):,}")

    print("\nEnhanced Model Results")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:0.3f}"))
    print(f"\nSaved results to: {OUTPUT_DIR / 'enhanced_model_results.csv'}")
    print(f"Saved predictions to: {OUTPUT_DIR / 'enhanced_model_predictions.csv'}")


if __name__ == "__main__":
    main()
