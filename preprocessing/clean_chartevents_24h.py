from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def main() -> None:
    input_path = PROCESSED_DATA_DIR / "icu_stays_with_chartevents_24h.parquet"
    output_path = PROCESSED_DATA_DIR / "icu_stays_with_chartevents_24h_cleaned.parquet"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing {input_path}. Run preprocessing/build_chartevents_24h_features.py first."
        )

    df = pd.read_parquet(input_path)

    vital_bounds = {
        "heart_rate": (20, 250),
        "sbp": (30, 300),
        "dbp": (20, 200),
        "resp_rate": (5, 80),
        "temperature": (30, 43),
        "spo2": (40, 100),
    }
    summary_suffixes = ["min", "max", "mean"]
    vital_columns = [f"{vital}_{suffix}" for vital in vital_bounds for suffix in summary_suffixes]

    missing_columns = [col for col in vital_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")

    clean_df = df.copy()
    for vital, (lower, upper) in vital_bounds.items():
        for suffix in summary_suffixes:
            col = f"{vital}_{suffix}"
            clean_df[col] = clean_df[col].mask(
                (clean_df[col] < lower) | (clean_df[col] > upper)
            )

    consistency_rows = []
    for vital in vital_bounds:
        min_col = f"{vital}_min"
        mean_col = f"{vital}_mean"
        max_col = f"{vital}_max"

        invalid_order = (
            clean_df[min_col].notna()
            & clean_df[mean_col].notna()
            & clean_df[max_col].notna()
            & (
                (clean_df[min_col] > clean_df[mean_col])
                | (clean_df[mean_col] > clean_df[max_col])
            )
        )

        consistency_rows.append(
            {
                "vital": vital,
                "rows_with_order_issue": int(invalid_order.sum()),
            }
        )
        clean_df.loc[invalid_order, [min_col, mean_col, max_col]] = np.nan

    clean_df.to_parquet(output_path, index=False)
    print(f"Saved cleaned chart-events file to: {output_path}")
    print(pd.DataFrame(consistency_rows).to_string(index=False))


if __name__ == "__main__":
    main()
