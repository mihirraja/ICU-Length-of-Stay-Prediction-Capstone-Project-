from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
TEMP_DIR = PROJECT_ROOT / ".duckdb_tmp"


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


def resolve_reference_path() -> Path:
    parquet_path = RAW_DATA_DIR / "chartevents_reference.parquet"
    csv_gz_path = RAW_DATA_DIR / "d_items.csv.gz"
    csv_path = RAW_DATA_DIR / "d_items.csv"
    if parquet_path.exists():
        return parquet_path
    if csv_gz_path.exists():
        return csv_gz_path
    if csv_path.exists():
        return csv_path
    raise FileNotFoundError(
        "Missing chart item reference. Expected either "
        f"{parquet_path}, {csv_gz_path}, or {csv_path}."
    )


def main() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    icu_stays_path = RAW_DATA_DIR / "icu_stays.csv"
    chartevents_path = resolve_raw_path("chartevents")
    reference_path = resolve_reference_path()

    missing = [p for p in [icu_stays_path] if not p.exists()]
    if missing:
        missing_str = "\n".join(str(p) for p in missing)
        raise FileNotFoundError(
            "Cannot build 24-hour chart-event features because these raw files are missing:\n"
            f"{missing_str}\n\n"
            "Add the raw MIMIC chart events file to data/raw first."
        )

    con = duckdb.connect()
    con.execute("SET threads=4;")
    con.execute("SET preserve_insertion_order=false;")
    con.execute("SET memory_limit='6GB';")
    con.execute(f"SET temp_directory='{TEMP_DIR.as_posix()}';")
    con.execute("SET max_temp_directory_size='20GiB';")

    con.execute(
        f"""
        CREATE OR REPLACE TABLE icu_stays AS
        SELECT *
        FROM read_csv_auto('{icu_stays_path.as_posix()}');
        """
    )

    if reference_path.suffix == ".parquet":
        con.execute(
            f"""
            CREATE OR REPLACE TABLE d_items AS
            SELECT itemid, label
            FROM read_parquet('{reference_path.as_posix()}');
            """
        )
    else:
        con.execute(
            f"""
            CREATE OR REPLACE TABLE d_items AS
            SELECT itemid, label
            FROM read_csv_auto('{reference_path.as_posix()}');
            """
        )

    con.execute(
        f"""
        CREATE OR REPLACE TABLE vitals_24h_features AS
        WITH relevant_d_items AS (
            SELECT
                itemid,
                label,
                CASE
                    WHEN lower(label) = 'heart rate' THEN 'heart_rate'
                    WHEN lower(label) IN (
                        'non invasive blood pressure systolic',
                        'arterial blood pressure systolic'
                    ) THEN 'sbp'
                    WHEN lower(label) IN (
                        'non invasive blood pressure diastolic',
                        'arterial blood pressure diastolic'
                    ) THEN 'dbp'
                    WHEN lower(label) = 'respiratory rate' THEN 'resp_rate'
                    WHEN lower(label) IN ('temperature celsius', 'temperature fahrenheit') THEN 'temperature'
                    WHEN lower(label) IN (
                        'o2 saturation pulseoxymetry',
                        'o2 saturation pulseoxymetry alarm - high',
                        'o2 saturation pulseoxymetry alarm - low',
                        'spo2'
                    ) THEN 'spo2'
                    ELSE NULL
                END AS vital_name
            FROM d_items
        ),
        vitals_24h_long AS (
            SELECT
                c.subject_id,
                c.hadm_id,
                c.stay_id,
                c.charttime,
                r.vital_name,
                CASE
                    WHEN r.label = 'Temperature Fahrenheit' THEN (c.valuenum - 32) * 5.0 / 9.0
                    ELSE c.valuenum
                END AS valuenum
            FROM read_csv_auto('{chartevents_path.as_posix()}') c
            JOIN relevant_d_items r
              ON c.itemid = r.itemid
            JOIN icu_stays i
              ON c.subject_id = i.subject_id
             AND c.hadm_id = i.hadm_id
             AND c.stay_id = i.stay_id
            WHERE r.vital_name IS NOT NULL
              AND c.valuenum IS NOT NULL
              AND c.charttime >= i.intime
              AND c.charttime < i.intime + INTERVAL '24 hours'
              AND c.charttime <= i.outtime
        ),
        vitals_agg AS (
            SELECT
                subject_id,
                hadm_id,
                stay_id,
                vital_name,
                MIN(valuenum) AS min_val,
                MAX(valuenum) AS max_val,
                AVG(valuenum) AS mean_val
            FROM vitals_24h_long
            GROUP BY subject_id, hadm_id, stay_id, vital_name
        )
        SELECT
            subject_id,
            hadm_id,
            stay_id,
            MAX(min_val) FILTER (WHERE vital_name = 'heart_rate') AS heart_rate_min,
            MAX(max_val) FILTER (WHERE vital_name = 'heart_rate') AS heart_rate_max,
            MAX(mean_val) FILTER (WHERE vital_name = 'heart_rate') AS heart_rate_mean,
            MAX(min_val) FILTER (WHERE vital_name = 'sbp') AS sbp_min,
            MAX(max_val) FILTER (WHERE vital_name = 'sbp') AS sbp_max,
            MAX(mean_val) FILTER (WHERE vital_name = 'sbp') AS sbp_mean,
            MAX(min_val) FILTER (WHERE vital_name = 'dbp') AS dbp_min,
            MAX(max_val) FILTER (WHERE vital_name = 'dbp') AS dbp_max,
            MAX(mean_val) FILTER (WHERE vital_name = 'dbp') AS dbp_mean,
            MAX(min_val) FILTER (WHERE vital_name = 'resp_rate') AS resp_rate_min,
            MAX(max_val) FILTER (WHERE vital_name = 'resp_rate') AS resp_rate_max,
            MAX(mean_val) FILTER (WHERE vital_name = 'resp_rate') AS resp_rate_mean,
            MAX(min_val) FILTER (WHERE vital_name = 'temperature') AS temperature_min,
            MAX(max_val) FILTER (WHERE vital_name = 'temperature') AS temperature_max,
            MAX(mean_val) FILTER (WHERE vital_name = 'temperature') AS temperature_mean,
            MAX(min_val) FILTER (WHERE vital_name = 'spo2') AS spo2_min,
            MAX(max_val) FILTER (WHERE vital_name = 'spo2') AS spo2_max,
            MAX(mean_val) FILTER (WHERE vital_name = 'spo2') AS spo2_mean
        FROM vitals_agg
        GROUP BY subject_id, hadm_id, stay_id
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE icu_stays_with_chartevents_24h AS
        SELECT
            i.*,
            f.heart_rate_min,
            f.heart_rate_max,
            f.heart_rate_mean,
            f.sbp_min,
            f.sbp_max,
            f.sbp_mean,
            f.dbp_min,
            f.dbp_max,
            f.dbp_mean,
            f.resp_rate_min,
            f.resp_rate_max,
            f.resp_rate_mean,
            f.temperature_min,
            f.temperature_max,
            f.temperature_mean,
            f.spo2_min,
            f.spo2_max,
            f.spo2_mean
        FROM icu_stays i
        LEFT JOIN vitals_24h_features f
          ON i.subject_id = f.subject_id
         AND i.hadm_id = f.hadm_id
         AND i.stay_id = f.stay_id
        """
    )

    vitals_features_path = PROCESSED_DATA_DIR / "chartevents_24h_features.parquet"
    merged_path = PROCESSED_DATA_DIR / "icu_stays_with_chartevents_24h.parquet"

    con.execute(
        f"COPY vitals_24h_features TO '{vitals_features_path.as_posix()}' (FORMAT PARQUET);"
    )
    con.execute(
        f"COPY icu_stays_with_chartevents_24h TO '{merged_path.as_posix()}' (FORMAT PARQUET);"
    )

    summary = con.execute(
        """
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT stay_id) AS unique_stays
        FROM icu_stays_with_chartevents_24h
        """
    ).fetchone()

    print("Saved:")
    print(vitals_features_path)
    print(merged_path)
    print(f"Rows: {summary[0]:,}")
    print(f"Unique stays: {summary[1]:,}")


if __name__ == "__main__":
    main()
