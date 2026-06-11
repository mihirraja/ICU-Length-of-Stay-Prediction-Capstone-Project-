from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


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
    prescriptions_path = resolve_raw_path("prescriptions")

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
        f"CREATE OR REPLACE TABLE prescriptions AS SELECT * FROM read_csv_auto('{prescriptions_path.as_posix()}');"
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE prescriptions_24h AS
        SELECT
            i.subject_id,
            i.hadm_id,
            i.stay_id,
            p.starttime,
            p.stoptime,
            p.drug_type,
            p.drug,
            p.dose_val_rx,
            p.dose_unit_rx,
            p.doses_per_24_hrs,
            p.route
        FROM prescriptions p
        JOIN icu_stays i
          ON p.hadm_id = i.hadm_id
        WHERE p.starttime < i.intime + INTERVAL '24 hours'
          AND COALESCE(p.stoptime, p.starttime) >= i.intime
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE prescriptions_24h_features AS
        WITH labeled AS (
            SELECT
                stay_id,
                lower(coalesce(drug, '')) AS drug_lower,
                upper(coalesce(route, '')) AS route_upper,
                CAST(coalesce(doses_per_24_hrs, 0) AS DOUBLE) AS doses_24h
            FROM prescriptions_24h
        )
        SELECT
            stay_id,
            COUNT(*) AS rx_order_count_24h,
            COUNT(DISTINCT drug_lower) AS rx_unique_drug_count_24h,
            COUNT(DISTINCT route_upper) AS rx_unique_route_count_24h,
            AVG(NULLIF(doses_24h, 0)) AS rx_mean_doses_per_24h,

            COUNT(*) FILTER (WHERE route_upper = 'IV') AS rx_route_iv_count_24h,
            COUNT(*) FILTER (WHERE route_upper = 'PO/NG') AS rx_route_pong_count_24h,
            COUNT(*) FILTER (WHERE route_upper = 'PO') AS rx_route_po_count_24h,
            COUNT(*) FILTER (WHERE route_upper = 'SC') AS rx_route_sc_count_24h,
            COUNT(*) FILTER (WHERE route_upper = 'IV DRIP') AS rx_route_iv_drip_count_24h,

            MAX(CASE WHEN drug_lower LIKE '%norepinephrine%' THEN 1 ELSE 0 END) AS rx_any_norepinephrine_24h,
            MAX(CASE WHEN drug_lower LIKE '%vasopressin%' THEN 1 ELSE 0 END) AS rx_any_vasopressin_24h,
            MAX(CASE WHEN drug_lower LIKE '%phenylephrine%' THEN 1 ELSE 0 END) AS rx_any_phenylephrine_24h,
            MAX(CASE WHEN drug_lower LIKE '%epinephrine%' THEN 1 ELSE 0 END) AS rx_any_epinephrine_24h,
            MAX(CASE WHEN drug_lower LIKE '%propofol%' THEN 1 ELSE 0 END) AS rx_any_propofol_24h,
            MAX(CASE WHEN drug_lower LIKE '%fentanyl%' THEN 1 ELSE 0 END) AS rx_any_fentanyl_24h,
            MAX(CASE WHEN drug_lower LIKE '%dexmedetomidine%' THEN 1 ELSE 0 END) AS rx_any_dexmedetomidine_24h,
            MAX(CASE WHEN drug_lower LIKE '%midazolam%' THEN 1 ELSE 0 END) AS rx_any_midazolam_24h,
            MAX(CASE WHEN drug_lower LIKE '%insulin%' THEN 1 ELSE 0 END) AS rx_any_insulin_24h,
            MAX(CASE WHEN drug_lower LIKE '%heparin%' THEN 1 ELSE 0 END) AS rx_any_heparin_24h,
            MAX(CASE WHEN drug_lower LIKE '%vancomycin%' THEN 1 ELSE 0 END) AS rx_any_vancomycin_24h,
            MAX(CASE WHEN drug_lower LIKE '%furosemide%' THEN 1 ELSE 0 END) AS rx_any_furosemide_24h,
            MAX(CASE WHEN drug_lower LIKE '%potassium chloride%' THEN 1 ELSE 0 END) AS rx_any_potassium_chloride_24h,
            MAX(CASE WHEN drug_lower LIKE '%magnesium sulfate%' THEN 1 ELSE 0 END) AS rx_any_magnesium_sulfate_24h,

            MAX(
                CASE
                    WHEN drug_lower LIKE '%norepinephrine%'
                      OR drug_lower LIKE '%vasopressin%'
                      OR drug_lower LIKE '%phenylephrine%'
                      OR drug_lower LIKE '%epinephrine%'
                    THEN 1 ELSE 0
                END
            ) AS rx_any_vasopressor_24h,
            MAX(
                CASE
                    WHEN drug_lower LIKE '%propofol%'
                      OR drug_lower LIKE '%fentanyl%'
                      OR drug_lower LIKE '%dexmedetomidine%'
                      OR drug_lower LIKE '%midazolam%'
                    THEN 1 ELSE 0
                END
            ) AS rx_any_sedation_24h
        FROM labeled
        GROUP BY stay_id
        """
    )

    features_path = PROCESSED_DATA_DIR / "prescriptions_24h_features.parquet"
    con.execute(
        f"COPY prescriptions_24h_features TO '{features_path.as_posix()}' (FORMAT PARQUET);"
    )
    print(f"Saved prescriptions 24-hour features to: {features_path}")


if __name__ == "__main__":
    main()
