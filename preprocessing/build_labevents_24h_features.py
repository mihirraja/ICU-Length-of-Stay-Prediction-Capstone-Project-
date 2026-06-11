from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


LAB_LABELS = {
    "creatinine": "creat",
    "urea nitrogen": "bun",
    "sodium": "na",
    "potassium": "k",
    "chloride": "cl",
    "bicarbonate": "hco3",
    "glucose": "glucose",
    "white blood cells": "wbc",
    "hemoglobin": "hgb",
    "hematocrit": "hct",
    "platelet count": "plt",
    "lactate": "lactate",
    "anion gap": "anion_gap",
}


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
    missing = [p for p in [icu_stays_path] if not p.exists()]
    if missing:
        missing_str = "\n".join(str(p) for p in missing)
        raise FileNotFoundError(
            "Cannot build 24-hour lab features because these raw files are missing:\n"
            f"{missing_str}\n\n"
            "Add the raw MIMIC lab files to data/raw first."
        )

    labevents_path = resolve_raw_path("labevents")
    d_labitems_path = resolve_raw_path("d_labitems")

    con = duckdb.connect()
    con.execute(f"CREATE OR REPLACE TABLE icu_stays AS SELECT * FROM read_csv_auto('{icu_stays_path.as_posix()}');")
    con.execute(f"CREATE OR REPLACE TABLE labevents AS SELECT * FROM read_csv_auto('{labevents_path.as_posix()}');")
    con.execute(f"CREATE OR REPLACE TABLE d_labitems AS SELECT * FROM read_csv_auto('{d_labitems_path.as_posix()}');")

    label_match_sql = []
    for human, short in LAB_LABELS.items():
        escaped = human.replace("'", "''")
        label_match_sql.append(
            f"""
            WHEN lower(d.label) = '{escaped}' AND lower(coalesce(d.fluid, '')) = 'blood' THEN '{short}'
            WHEN lower(d.label) LIKE '%{escaped}%' AND lower(coalesce(d.fluid, '')) = 'blood' THEN '{short}'
            """
        )
    case_sql = "\n".join(label_match_sql)

    con.execute(
        f"""
        CREATE OR REPLACE TABLE labs_24h_long AS
        WITH labeled_events AS (
            SELECT
                i.subject_id,
                i.hadm_id,
                i.stay_id,
                i.intime,
                i.outtime,
                l.charttime,
                l.itemid,
                l.valuenum,
                CASE
                    {case_sql}
                    ELSE NULL
                END AS lab_name
            FROM labevents l
            JOIN icu_stays i
              ON l.hadm_id = i.hadm_id
            JOIN d_labitems d
              ON l.itemid = d.itemid
        )
        SELECT
            subject_id,
            hadm_id,
            stay_id,
            charttime,
            itemid,
            lab_name,
            valuenum
        FROM labeled_events
        WHERE lab_name IS NOT NULL
          AND valuenum IS NOT NULL
          AND charttime >= intime
          AND charttime < intime + INTERVAL '24 hours'
          AND charttime <= outtime
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE labs_24h_features AS
        WITH lab_agg AS (
            SELECT
                stay_id,
                lab_name,
                MIN(valuenum) AS min_val,
                MAX(valuenum) AS max_val,
                AVG(valuenum) AS mean_val,
                STDDEV_SAMP(valuenum) AS std_val,
                ARG_MIN(valuenum, charttime) AS first_val,
                ARG_MAX(valuenum, charttime) AS last_val,
                COUNT(*) AS count_val
            FROM labs_24h_long
            GROUP BY stay_id, lab_name
        )
        SELECT
            stay_id,
            MAX(min_val) FILTER (WHERE lab_name = 'creat') AS lab_creat_min,
            MAX(max_val) FILTER (WHERE lab_name = 'creat') AS lab_creat_max,
            MAX(mean_val) FILTER (WHERE lab_name = 'creat') AS lab_creat_mean,
            MAX(std_val) FILTER (WHERE lab_name = 'creat') AS lab_creat_std,
            MAX(first_val) FILTER (WHERE lab_name = 'creat') AS lab_creat_first,
            MAX(last_val) FILTER (WHERE lab_name = 'creat') AS lab_creat_last,
            MAX(count_val) FILTER (WHERE lab_name = 'creat') AS lab_creat_count,

            MAX(min_val) FILTER (WHERE lab_name = 'bun') AS lab_bun_min,
            MAX(max_val) FILTER (WHERE lab_name = 'bun') AS lab_bun_max,
            MAX(mean_val) FILTER (WHERE lab_name = 'bun') AS lab_bun_mean,
            MAX(std_val) FILTER (WHERE lab_name = 'bun') AS lab_bun_std,
            MAX(first_val) FILTER (WHERE lab_name = 'bun') AS lab_bun_first,
            MAX(last_val) FILTER (WHERE lab_name = 'bun') AS lab_bun_last,
            MAX(count_val) FILTER (WHERE lab_name = 'bun') AS lab_bun_count,

            MAX(min_val) FILTER (WHERE lab_name = 'na') AS lab_na_min,
            MAX(max_val) FILTER (WHERE lab_name = 'na') AS lab_na_max,
            MAX(mean_val) FILTER (WHERE lab_name = 'na') AS lab_na_mean,
            MAX(std_val) FILTER (WHERE lab_name = 'na') AS lab_na_std,
            MAX(first_val) FILTER (WHERE lab_name = 'na') AS lab_na_first,
            MAX(last_val) FILTER (WHERE lab_name = 'na') AS lab_na_last,
            MAX(count_val) FILTER (WHERE lab_name = 'na') AS lab_na_count,

            MAX(min_val) FILTER (WHERE lab_name = 'k') AS lab_k_min,
            MAX(max_val) FILTER (WHERE lab_name = 'k') AS lab_k_max,
            MAX(mean_val) FILTER (WHERE lab_name = 'k') AS lab_k_mean,
            MAX(std_val) FILTER (WHERE lab_name = 'k') AS lab_k_std,
            MAX(first_val) FILTER (WHERE lab_name = 'k') AS lab_k_first,
            MAX(last_val) FILTER (WHERE lab_name = 'k') AS lab_k_last,
            MAX(count_val) FILTER (WHERE lab_name = 'k') AS lab_k_count,

            MAX(min_val) FILTER (WHERE lab_name = 'cl') AS lab_cl_min,
            MAX(max_val) FILTER (WHERE lab_name = 'cl') AS lab_cl_max,
            MAX(mean_val) FILTER (WHERE lab_name = 'cl') AS lab_cl_mean,
            MAX(std_val) FILTER (WHERE lab_name = 'cl') AS lab_cl_std,
            MAX(first_val) FILTER (WHERE lab_name = 'cl') AS lab_cl_first,
            MAX(last_val) FILTER (WHERE lab_name = 'cl') AS lab_cl_last,
            MAX(count_val) FILTER (WHERE lab_name = 'cl') AS lab_cl_count,

            MAX(min_val) FILTER (WHERE lab_name = 'hco3') AS lab_hco3_min,
            MAX(max_val) FILTER (WHERE lab_name = 'hco3') AS lab_hco3_max,
            MAX(mean_val) FILTER (WHERE lab_name = 'hco3') AS lab_hco3_mean,
            MAX(std_val) FILTER (WHERE lab_name = 'hco3') AS lab_hco3_std,
            MAX(first_val) FILTER (WHERE lab_name = 'hco3') AS lab_hco3_first,
            MAX(last_val) FILTER (WHERE lab_name = 'hco3') AS lab_hco3_last,
            MAX(count_val) FILTER (WHERE lab_name = 'hco3') AS lab_hco3_count,

            MAX(min_val) FILTER (WHERE lab_name = 'glucose') AS lab_glucose_min,
            MAX(max_val) FILTER (WHERE lab_name = 'glucose') AS lab_glucose_max,
            MAX(mean_val) FILTER (WHERE lab_name = 'glucose') AS lab_glucose_mean,
            MAX(std_val) FILTER (WHERE lab_name = 'glucose') AS lab_glucose_std,
            MAX(first_val) FILTER (WHERE lab_name = 'glucose') AS lab_glucose_first,
            MAX(last_val) FILTER (WHERE lab_name = 'glucose') AS lab_glucose_last,
            MAX(count_val) FILTER (WHERE lab_name = 'glucose') AS lab_glucose_count,

            MAX(min_val) FILTER (WHERE lab_name = 'wbc') AS lab_wbc_min,
            MAX(max_val) FILTER (WHERE lab_name = 'wbc') AS lab_wbc_max,
            MAX(mean_val) FILTER (WHERE lab_name = 'wbc') AS lab_wbc_mean,
            MAX(std_val) FILTER (WHERE lab_name = 'wbc') AS lab_wbc_std,
            MAX(first_val) FILTER (WHERE lab_name = 'wbc') AS lab_wbc_first,
            MAX(last_val) FILTER (WHERE lab_name = 'wbc') AS lab_wbc_last,
            MAX(count_val) FILTER (WHERE lab_name = 'wbc') AS lab_wbc_count,

            MAX(min_val) FILTER (WHERE lab_name = 'hgb') AS lab_hgb_min,
            MAX(max_val) FILTER (WHERE lab_name = 'hgb') AS lab_hgb_max,
            MAX(mean_val) FILTER (WHERE lab_name = 'hgb') AS lab_hgb_mean,
            MAX(std_val) FILTER (WHERE lab_name = 'hgb') AS lab_hgb_std,
            MAX(first_val) FILTER (WHERE lab_name = 'hgb') AS lab_hgb_first,
            MAX(last_val) FILTER (WHERE lab_name = 'hgb') AS lab_hgb_last,
            MAX(count_val) FILTER (WHERE lab_name = 'hgb') AS lab_hgb_count,

            MAX(min_val) FILTER (WHERE lab_name = 'hct') AS lab_hct_min,
            MAX(max_val) FILTER (WHERE lab_name = 'hct') AS lab_hct_max,
            MAX(mean_val) FILTER (WHERE lab_name = 'hct') AS lab_hct_mean,
            MAX(std_val) FILTER (WHERE lab_name = 'hct') AS lab_hct_std,
            MAX(first_val) FILTER (WHERE lab_name = 'hct') AS lab_hct_first,
            MAX(last_val) FILTER (WHERE lab_name = 'hct') AS lab_hct_last,
            MAX(count_val) FILTER (WHERE lab_name = 'hct') AS lab_hct_count,

            MAX(min_val) FILTER (WHERE lab_name = 'plt') AS lab_plt_min,
            MAX(max_val) FILTER (WHERE lab_name = 'plt') AS lab_plt_max,
            MAX(mean_val) FILTER (WHERE lab_name = 'plt') AS lab_plt_mean,
            MAX(std_val) FILTER (WHERE lab_name = 'plt') AS lab_plt_std,
            MAX(first_val) FILTER (WHERE lab_name = 'plt') AS lab_plt_first,
            MAX(last_val) FILTER (WHERE lab_name = 'plt') AS lab_plt_last,
            MAX(count_val) FILTER (WHERE lab_name = 'plt') AS lab_plt_count,

            MAX(min_val) FILTER (WHERE lab_name = 'lactate') AS lab_lactate_min,
            MAX(max_val) FILTER (WHERE lab_name = 'lactate') AS lab_lactate_max,
            MAX(mean_val) FILTER (WHERE lab_name = 'lactate') AS lab_lactate_mean,
            MAX(std_val) FILTER (WHERE lab_name = 'lactate') AS lab_lactate_std,
            MAX(first_val) FILTER (WHERE lab_name = 'lactate') AS lab_lactate_first,
            MAX(last_val) FILTER (WHERE lab_name = 'lactate') AS lab_lactate_last,
            MAX(count_val) FILTER (WHERE lab_name = 'lactate') AS lab_lactate_count,

            MAX(min_val) FILTER (WHERE lab_name = 'anion_gap') AS lab_anion_gap_min,
            MAX(max_val) FILTER (WHERE lab_name = 'anion_gap') AS lab_anion_gap_max,
            MAX(mean_val) FILTER (WHERE lab_name = 'anion_gap') AS lab_anion_gap_mean,
            MAX(std_val) FILTER (WHERE lab_name = 'anion_gap') AS lab_anion_gap_std,
            MAX(first_val) FILTER (WHERE lab_name = 'anion_gap') AS lab_anion_gap_first,
            MAX(last_val) FILTER (WHERE lab_name = 'anion_gap') AS lab_anion_gap_last,
            MAX(count_val) FILTER (WHERE lab_name = 'anion_gap') AS lab_anion_gap_count
        FROM lab_agg
        GROUP BY stay_id
        """
    )

    long_path = PROCESSED_DATA_DIR / "labevents_24h_long.parquet"
    features_path = PROCESSED_DATA_DIR / "labevents_24h_features.parquet"
    con.execute(f"COPY labs_24h_long TO '{long_path.as_posix()}' (FORMAT PARQUET);")
    con.execute(f"COPY labs_24h_features TO '{features_path.as_posix()}' (FORMAT PARQUET);")

    summary = con.execute(
        """
        SELECT COUNT(*) AS rows, COUNT(DISTINCT stay_id) AS unique_stays
        FROM labs_24h_features
        """
    ).fetchone()

    print("Saved:")
    print(long_path)
    print(features_path)
    print(f"Rows: {summary[0]:,}")
    print(f"Unique stays: {summary[1]:,}")


if __name__ == "__main__":
    main()
