from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


TOP_PROCEDURE_ITEMIDS = [224275, 225459, 224277, 225402, 225752, 225792]


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
    procedureevents_path = resolve_raw_path("procedureevents")

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
        f"CREATE OR REPLACE TABLE procedureevents AS SELECT * FROM read_csv_auto('{procedureevents_path.as_posix()}');"
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE procedureevents_24h AS
        SELECT
            i.subject_id,
            i.hadm_id,
            i.stay_id,
            pe.starttime,
            pe.endtime,
            pe.itemid,
            pe.value,
            pe.valueuom,
            pe.ordercategoryname,
            pe.ordercategorydescription,
            pe.statusdescription
        FROM procedureevents pe
        JOIN icu_stays i
          ON pe.stay_id = i.stay_id
        WHERE pe.starttime < i.intime + INTERVAL '24 hours'
          AND COALESCE(pe.endtime, pe.starttime) >= i.intime
        """
    )

    dynamic_cols = []
    for itemid in TOP_PROCEDURE_ITEMIDS:
        dynamic_cols.extend(
            [
                f"COUNT(*) FILTER (WHERE itemid = {itemid}) AS proc_item_{itemid}_count_24h",
                f"AVG(CASE WHEN itemid = {itemid} THEN value ELSE NULL END) AS proc_item_{itemid}_mean_value_24h",
            ]
        )
    dynamic_sql = ",\n            ".join(dynamic_cols)

    con.execute(
        f"""
        CREATE OR REPLACE TABLE procedureevents_24h_features AS
        SELECT
            stay_id,
            COUNT(*) AS proc_event_count_24h,
            COUNT(DISTINCT itemid) AS proc_unique_item_count_24h,
            COUNT(DISTINCT ordercategoryname) AS proc_unique_category_count_24h,

            COUNT(*) FILTER (WHERE lower(coalesce(ordercategoryname, '')) = 'procedures') AS proc_procedures_count_24h,
            COUNT(*) FILTER (WHERE lower(coalesce(ordercategoryname, '')) = 'imaging') AS proc_imaging_count_24h,
            COUNT(*) FILTER (WHERE lower(coalesce(ordercategoryname, '')) = 'ventilation') AS proc_ventilation_count_24h,
            COUNT(*) FILTER (WHERE lower(coalesce(ordercategoryname, '')) = 'intubation/extubation') AS proc_intubation_extubation_count_24h,
            COUNT(*) FILTER (WHERE lower(coalesce(ordercategoryname, '')) = 'invasive lines') AS proc_invasive_lines_count_24h,
            COUNT(*) FILTER (WHERE lower(coalesce(ordercategoryname, '')) = 'peripheral lines') AS proc_peripheral_lines_count_24h,
            COUNT(*) FILTER (WHERE lower(coalesce(ordercategoryname, '')) = 'tubes') AS proc_tubes_count_24h,
            COUNT(*) FILTER (WHERE lower(coalesce(ordercategoryname, '')) = 'dialysis') AS proc_dialysis_count_24h,
            COUNT(*) FILTER (WHERE lower(coalesce(ordercategoryname, '')) = 'communication') AS proc_communication_count_24h,
            COUNT(*) FILTER (WHERE lower(coalesce(ordercategoryname, '')) = 'significant events') AS proc_significant_events_count_24h,

            MAX(CASE WHEN lower(coalesce(ordercategoryname, '')) = 'ventilation' THEN 1 ELSE 0 END) AS proc_any_ventilation_24h,
            MAX(CASE WHEN lower(coalesce(ordercategoryname, '')) = 'intubation/extubation' THEN 1 ELSE 0 END) AS proc_any_intubation_extubation_24h,
            MAX(CASE WHEN lower(coalesce(ordercategoryname, '')) = 'invasive lines' THEN 1 ELSE 0 END) AS proc_any_invasive_line_24h,
            MAX(CASE WHEN lower(coalesce(ordercategoryname, '')) = 'peripheral lines' THEN 1 ELSE 0 END) AS proc_any_peripheral_line_24h,
            MAX(CASE WHEN lower(coalesce(ordercategoryname, '')) = 'tubes' THEN 1 ELSE 0 END) AS proc_any_tube_24h,
            MAX(CASE WHEN lower(coalesce(ordercategoryname, '')) = 'dialysis' THEN 1 ELSE 0 END) AS proc_any_dialysis_24h,
            MAX(CASE WHEN lower(coalesce(ordercategoryname, '')) = 'imaging' THEN 1 ELSE 0 END) AS proc_any_imaging_24h,

            {dynamic_sql}
        FROM procedureevents_24h
        GROUP BY stay_id
        """
    )

    features_path = PROCESSED_DATA_DIR / "procedureevents_24h_features.parquet"
    con.execute(
        f"COPY procedureevents_24h_features TO '{features_path.as_posix()}' (FORMAT PARQUET);"
    )
    print(f"Saved procedureevents 24-hour features to: {features_path}")


if __name__ == "__main__":
    main()
