import duckdb
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MIMIC_ROOT = Path("mimic-iv-3.1")
HOSP       = MIMIC_ROOT / "hosp"
ICU        = MIMIC_ROOT / "icu"
OUT_DIR    = Path("data/processed"); OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_LOS_HOURS  = 24
READMIT_WINDOW = 30


def build_cohort(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    log.info("Building cohort...")
    query = f"""
    WITH stays AS (
        SELECT
            i.subject_id,
            i.hadm_id,
            i.stay_id,
            i.intime,
            i.outtime,
            i.los,
            p.anchor_age                AS age,
            p.gender,
            a.admission_type,
            a.admission_location,
            a.discharge_location,
            a.insurance,
            a.race,
            a.language,
            a.marital_status,
            a.hospital_expire_flag      AS died_in_hospital,
            a.admittime                 AS hosp_admittime,
            a.dischtime                 AS hosp_dischtime,
            a.deathtime
        FROM read_csv_auto('{ICU}/icustays.csv.gz')    AS i
        JOIN read_csv_auto('{HOSP}/admissions.csv.gz') AS a
             USING (subject_id, hadm_id)
        JOIN read_csv_auto('{HOSP}/patients.csv.gz')   AS p
             USING (subject_id)
        WHERE i.los * 24 >= {MIN_LOS_HOURS}
    ),
    next_admit AS (
        SELECT
            a1.subject_id,
            a1.hadm_id,
            MIN(a2.admittime) AS next_admittime
        FROM read_csv_auto('{HOSP}/admissions.csv.gz') a1
        JOIN read_csv_auto('{HOSP}/admissions.csv.gz') a2
             ON  a1.subject_id = a2.subject_id
             AND a2.admittime  > a1.dischtime
             AND a2.admittime <= a1.dischtime + INTERVAL '{READMIT_WINDOW} days'
        GROUP BY a1.subject_id, a1.hadm_id
    )
    SELECT
        s.*,
        CASE WHEN na.next_admittime IS NOT NULL THEN 1 ELSE 0 END AS readmit_30d
    FROM stays s
    LEFT JOIN next_admit na USING (subject_id, hadm_id)
    ORDER BY s.subject_id, s.intime
    """
    cohort = con.execute(query).df()
    log.info(f"Cohort size: {len(cohort):,} ICU stays | "
             f"readmit_30d={cohort['readmit_30d'].mean():.2%} | "
             f"mortality={cohort['died_in_hospital'].mean():.2%}")
    return cohort


def add_diagnoses(cohort: pd.DataFrame, con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    log.info("Adding primary diagnoses...")
    diag = con.execute(f"""
        SELECT hadm_id, icd_code AS primary_icd, icd_version
        FROM read_csv_auto('{HOSP}/diagnoses_icd.csv.gz')
        WHERE seq_num = 1
    """).df()
    return cohort.merge(diag, on="hadm_id", how="left")


def add_comorbidity_count(cohort: pd.DataFrame, con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    log.info("Adding comorbidity count...")
    diag = con.execute(f"""
        SELECT hadm_id, COUNT(DISTINCT icd_code) AS comorbidity_count
        FROM read_csv_auto('{HOSP}/diagnoses_icd.csv.gz')
        GROUP BY hadm_id
    """).df()
    return cohort.merge(diag, on="hadm_id", how="left")


def add_drg_severity(cohort: pd.DataFrame, con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    log.info("Adding DRG severity...")
    drg = con.execute(f"""
        SELECT hadm_id, drg_severity, drg_mortality
        FROM read_csv_auto('{HOSP}/drgcodes.csv.gz')
        WHERE drg_type = 'APR'
    """).df()
    drg = drg.groupby("hadm_id").agg(
        drg_severity=("drg_severity", "max"),
        drg_mortality=("drg_mortality", "max")
    ).reset_index()
    return cohort.merge(drg, on="hadm_id", how="left")


def main():
    con = duckdb.connect()
    cohort = build_cohort(con)
    cohort = add_diagnoses(cohort, con)
    cohort = add_comorbidity_count(cohort, con)
    cohort = add_drg_severity(cohort, con)

    out = OUT_DIR / "cohort.parquet"
    cohort.to_parquet(out, index=False)
    log.info(f"Saved → {out}  ({cohort.shape})")
    print(cohort.head())
    print(cohort.dtypes)


if __name__ == "__main__":
    main()