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
    inputevents_path = resolve_raw_path("inputevents")

    missing = [p for p in [icu_stays_path, inputevents_path] if not p.exists()]
    if missing:
        missing_str = "\n".join(str(p) for p in missing)
        raise FileNotFoundError(
            "Cannot build 24-hour inputevents features because these raw files are missing:\n"
            f"{missing_str}"
        )

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
        f"CREATE OR REPLACE TABLE inputevents AS SELECT * FROM read_csv_auto('{inputevents_path.as_posix()}');"
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE inputevents_24h AS
        SELECT
            i.subject_id,
            i.hadm_id,
            i.stay_id,
            ie.starttime,
            ie.endtime,
            ie.itemid,
            ie.amount,
            ie.amountuom,
            ie.rate,
            ie.rateuom,
            ie.ordercategoryname,
            ie.ordercategorydescription,
            ie.statusdescription,
            ie.patientweight,
            ie.totalamount,
            ie.totalamountuom
        FROM inputevents ie
        JOIN icu_stays i
          ON ie.stay_id = i.stay_id
        WHERE ie.starttime < i.intime + INTERVAL '24 hours'
          AND COALESCE(ie.endtime, ie.starttime) >= i.intime
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE inputevents_24h_features AS
        WITH base AS (
            SELECT
                stay_id,
                COUNT(*) AS input_event_count_24h,
                COUNT(DISTINCT itemid) AS input_unique_item_count_24h,
                COUNT(DISTINCT ordercategoryname) AS input_unique_category_count_24h,
                COUNT(*) FILTER (
                    WHERE lower(coalesce(ordercategorydescription, '')) = 'continuous med'
                ) AS input_continuous_med_count_24h,
                COUNT(*) FILTER (
                    WHERE lower(coalesce(ordercategorydescription, '')) = 'continuous iv'
                ) AS input_continuous_iv_count_24h,
                COUNT(*) FILTER (
                    WHERE lower(coalesce(ordercategorydescription, '')) = 'bolus'
                ) AS input_bolus_count_24h,
                COUNT(*) FILTER (
                    WHERE lower(coalesce(ordercategorydescription, '')) = 'drug push'
                ) AS input_drug_push_count_24h,
                COUNT(*) FILTER (
                    WHERE lower(coalesce(ordercategorydescription, '')) = 'non iv meds'
                ) AS input_non_iv_med_count_24h,
                SUM(CASE WHEN lower(coalesce(amountuom, '')) = 'ml' THEN amount ELSE 0 END) AS input_total_ml_24h,
                SUM(
                    CASE
                        WHEN lower(coalesce(totalamountuom, '')) = 'ml' THEN totalamount
                        ELSE 0
                    END
                ) AS input_totalamount_ml_24h,
                AVG(NULLIF(patientweight, 0)) AS input_avg_patientweight_24h,

                COUNT(*) FILTER (
                    WHERE lower(coalesce(ordercategoryname, '')) = '01-drips'
                ) AS input_drips_count_24h,
                COUNT(*) FILTER (
                    WHERE lower(coalesce(ordercategoryname, '')) = '02-fluids (crystalloids)'
                ) AS input_crystalloid_count_24h,
                COUNT(*) FILTER (
                    WHERE lower(coalesce(ordercategoryname, '')) = '03-iv fluid bolus'
                ) AS input_iv_fluid_bolus_count_24h,
                COUNT(*) FILTER (
                    WHERE lower(coalesce(ordercategoryname, '')) = '04-fluids (colloids)'
                ) AS input_colloid_count_24h,
                COUNT(*) FILTER (
                    WHERE lower(coalesce(ordercategoryname, '')) = '05-med bolus'
                ) AS input_med_bolus_count_24h,
                COUNT(*) FILTER (
                    WHERE lower(coalesce(ordercategoryname, '')) = '06-insulin (non iv)'
                ) AS input_insulin_noniv_count_24h,
                COUNT(*) FILTER (
                    WHERE lower(coalesce(ordercategoryname, '')) = '07-blood products'
                ) AS input_blood_product_count_24h,
                COUNT(*) FILTER (
                    WHERE lower(coalesce(ordercategoryname, '')) = '08-antibiotics (iv)'
                ) AS input_antibiotic_iv_count_24h,
                COUNT(*) FILTER (
                    WHERE lower(coalesce(ordercategoryname, '')) = '09-antibiotics (non iv)'
                ) AS input_antibiotic_noniv_count_24h,
                COUNT(*) FILTER (
                    WHERE lower(coalesce(ordercategoryname, '')) = '10-prophylaxis (iv)'
                ) AS input_prophylaxis_iv_count_24h,
                COUNT(*) FILTER (
                    WHERE lower(coalesce(ordercategoryname, '')) = '11-prophylaxis (non iv)'
                ) AS input_prophylaxis_noniv_count_24h,
                COUNT(*) FILTER (
                    WHERE lower(coalesce(ordercategoryname, '')) = '12-parenteral nutrition'
                ) AS input_parenteral_nutrition_count_24h,
                COUNT(*) FILTER (
                    WHERE lower(coalesce(ordercategoryname, '')) = '13-enteral nutrition'
                ) AS input_enteral_nutrition_count_24h,
                COUNT(*) FILTER (
                    WHERE lower(coalesce(ordercategoryname, '')) = '14-oral/gastric intake'
                ) AS input_oral_gastric_count_24h,
                COUNT(*) FILTER (
                    WHERE lower(coalesce(ordercategoryname, '')) = '16-pre admission/non-icu'
                ) AS input_pre_icu_count_24h,

                SUM(
                    CASE
                        WHEN lower(coalesce(ordercategoryname, '')) = '02-fluids (crystalloids)'
                         AND lower(coalesce(amountuom, '')) = 'ml'
                        THEN amount ELSE 0
                    END
                ) AS input_crystalloid_ml_24h,
                SUM(
                    CASE
                        WHEN lower(coalesce(ordercategoryname, '')) = '03-iv fluid bolus'
                         AND lower(coalesce(amountuom, '')) = 'ml'
                        THEN amount ELSE 0
                    END
                ) AS input_iv_bolus_ml_24h,
                SUM(
                    CASE
                        WHEN lower(coalesce(ordercategoryname, '')) = '04-fluids (colloids)'
                         AND lower(coalesce(amountuom, '')) = 'ml'
                        THEN amount ELSE 0
                    END
                ) AS input_colloid_ml_24h,
                SUM(
                    CASE
                        WHEN lower(coalesce(ordercategoryname, '')) = '07-blood products'
                         AND lower(coalesce(amountuom, '')) = 'ml'
                        THEN amount ELSE 0
                    END
                ) AS input_blood_product_ml_24h,
                SUM(
                    CASE
                        WHEN lower(coalesce(ordercategoryname, '')) = '13-enteral nutrition'
                         AND lower(coalesce(amountuom, '')) = 'ml'
                        THEN amount ELSE 0
                    END
                ) AS input_enteral_ml_24h,
                SUM(
                    CASE
                        WHEN lower(coalesce(ordercategoryname, '')) = '14-oral/gastric intake'
                         AND lower(coalesce(amountuom, '')) = 'ml'
                        THEN amount ELSE 0
                    END
                ) AS input_oral_gastric_ml_24h,

                MAX(
                    CASE WHEN lower(coalesce(ordercategoryname, '')) = '01-drips' THEN 1 ELSE 0 END
                ) AS input_any_drip_24h,
                MAX(
                    CASE WHEN lower(coalesce(ordercategoryname, '')) = '02-fluids (crystalloids)' THEN 1 ELSE 0 END
                ) AS input_any_crystalloid_24h,
                MAX(
                    CASE WHEN lower(coalesce(ordercategoryname, '')) = '03-iv fluid bolus' THEN 1 ELSE 0 END
                ) AS input_any_iv_bolus_24h,
                MAX(
                    CASE WHEN lower(coalesce(ordercategoryname, '')) = '05-med bolus' THEN 1 ELSE 0 END
                ) AS input_any_med_bolus_24h,
                MAX(
                    CASE WHEN lower(coalesce(ordercategoryname, '')) = '07-blood products' THEN 1 ELSE 0 END
                ) AS input_any_blood_product_24h,
                MAX(
                    CASE WHEN lower(coalesce(ordercategoryname, '')) = '08-antibiotics (iv)' THEN 1 ELSE 0 END
                ) AS input_any_antibiotic_iv_24h,
                MAX(
                    CASE WHEN lower(coalesce(ordercategoryname, '')) = '12-parenteral nutrition' THEN 1 ELSE 0 END
                ) AS input_any_parenteral_nutrition_24h,
                MAX(
                    CASE WHEN lower(coalesce(ordercategoryname, '')) = '13-enteral nutrition' THEN 1 ELSE 0 END
                ) AS input_any_enteral_nutrition_24h
            FROM inputevents_24h
            GROUP BY stay_id
        )
        SELECT * FROM base
        """
    )

    features_path = PROCESSED_DATA_DIR / "inputevents_24h_features.parquet"
    con.execute(
        f"COPY inputevents_24h_features TO '{features_path.as_posix()}' (FORMAT PARQUET);"
    )

    print(f"Saved inputevents 24-hour features to: {features_path}")


if __name__ == "__main__":
    main()
