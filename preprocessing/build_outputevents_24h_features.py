from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


TOP_OUTPUT_ITEMIDS = [226559, 226560, 226588, 226606, 227510]


def resolve_raw_path(stem: str) -> Path:
    csv_gz = RAW_DATA_DIR / f"{stem}.csv.gz"
    csv = RAW_DATA_DIR / f"{stem}.csv"
    if csv_gz.exists():
        return csv_gz
    if csv.exists():
        return csv
    raise FileNotFoundError(
        f"Missing raw file for {stem}. Expected either {csv_gz} or {csv}."
    )


def main() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    icu_stays_path = RAW_DATA_DIR / "icu_stays.csv"
    outputevents_path = resolve_raw_path("outputevents")

    con = duckdb.connect()
    con.execute("PRAGMA threads=4;")
    con.execute("PRAGMA preserve_insertion_order=false;")
    con.execute("PRAGMA memory_limit='6GB';")
    con.execute(f"PRAGMA temp_directory='{(PROJECT_ROOT / '.duckdb_tmp').as_posix()}';")
    con.execute("PRAGMA max_temp_directory_size='20GiB';")

    con.execute(
        f"CREATE OR REPLACE TABLE icu_stays AS SELECT * FROM read_csv_auto('{icu_stays_path.as_posix()}');"
    )
    con.execute(
        f"CREATE OR REPLACE TABLE outputevents AS SELECT * FROM read_csv_auto('{outputevents_path.as_posix()}');"
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE outputevents_24h AS
        SELECT
            i.subject_id,
            i.hadm_id,
            i.stay_id,
            oe.charttime,
            oe.itemid,
            oe.value,
            oe.valueuom
        FROM outputevents oe
        JOIN icu_stays i
          ON oe.stay_id = i.stay_id
        WHERE oe.charttime >= i.intime
          AND oe.charttime < i.intime + INTERVAL '24 hours'
          AND oe.charttime <= i.outtime
        """
    )

    dynamic_cols = []
    for itemid in TOP_OUTPUT_ITEMIDS:
        dynamic_cols.extend(
            [
                f"SUM(CASE WHEN itemid = {itemid} AND lower(coalesce(valueuom, '')) = 'ml' THEN value ELSE 0 END) AS output_item_{itemid}_ml_24h",
                f"COUNT(*) FILTER (WHERE itemid = {itemid}) AS output_item_{itemid}_count_24h",
            ]
        )
    dynamic_sql = ",\n                ".join(dynamic_cols)

    con.execute(
        f"""
        CREATE OR REPLACE TABLE outputevents_24h_features AS
        SELECT
            stay_id,
            COUNT(*) AS output_event_count_24h,
            COUNT(DISTINCT itemid) AS output_unique_item_count_24h,
            SUM(CASE WHEN lower(coalesce(valueuom, '')) = 'ml' THEN value ELSE 0 END) AS output_total_ml_24h,
            AVG(CASE WHEN lower(coalesce(valueuom, '')) = 'ml' THEN value ELSE NULL END) AS output_mean_ml_24h,
            MAX(CASE WHEN lower(coalesce(valueuom, '')) = 'ml' THEN value ELSE NULL END) AS output_max_ml_24h,
            {dynamic_sql}
        FROM outputevents_24h
        GROUP BY stay_id
        """
    )

    features_path = PROCESSED_DATA_DIR / "outputevents_24h_features.parquet"
    con.execute(
        f"COPY outputevents_24h_features TO '{features_path.as_posix()}' (FORMAT PARQUET);"
    )
    print(f"Saved outputevents 24-hour features to: {features_path}")


if __name__ == "__main__":
    main()
