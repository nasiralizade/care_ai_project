import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MIMIC_ROOT = Path("mimic-iv-3.1")
ICU        = MIMIC_ROOT / "icu"
HOSP       = MIMIC_ROOT / "hosp"
DATA_DIR   = Path("data")
OUT_DIR    = DATA_DIR / "processed"; OUT_DIR.mkdir(parents=True, exist_ok=True)

VASOPRESSOR_ITEMS = [221906, 221289, 222315, 221662, 221653]
FLUID_ITEMS       = [220949, 220950, 220952, 225158]
RRT_ITEMS         = [225441, 225802, 225803]
TRANSFUSION_ITEMS = [225168, 226370, 227070]


def load_vasopressors(con, stay_ids):
    ids   = ",".join(map(str, stay_ids))
    items = ",".join(map(str, VASOPRESSOR_ITEMS))
    df = con.execute(f"""
        SELECT stay_id, starttime AS event_time, 1 AS vasopressor
        FROM read_csv_auto('{ICU}/inputevents.csv.gz')
        WHERE stay_id IN ({ids}) AND itemid IN ({items}) AND amount > 0
    """).df()
    df["event_time"] = pd.to_datetime(df["event_time"])
    return df


def load_fluids(con, stay_ids):
    ids   = ",".join(map(str, stay_ids))
    items = ",".join(map(str, FLUID_ITEMS))
    df = con.execute(f"""
        SELECT stay_id,
               DATE_TRUNC('hour', starttime::TIMESTAMP) AS event_time,
               SUM(amount) AS total_ml
        FROM read_csv_auto('{ICU}/inputevents.csv.gz')
        WHERE stay_id IN ({ids}) AND itemid IN ({items}) AND amountuom = 'mL'
        GROUP BY stay_id, DATE_TRUNC('hour', starttime::TIMESTAMP)
        HAVING SUM(amount) >= 500
    """).df()
    df["fluid_bolus"] = 1
    df["event_time"]  = pd.to_datetime(df["event_time"])
    return df[["stay_id", "event_time", "fluid_bolus"]]


def load_rrt(con, stay_ids):
    ids   = ",".join(map(str, stay_ids))
    items = ",".join(map(str, RRT_ITEMS))
    df = con.execute(f"""
        SELECT stay_id, starttime AS event_time, 1 AS rrt
        FROM read_csv_auto('{ICU}/procedureevents.csv.gz')
        WHERE stay_id IN ({ids}) AND itemid IN ({items})
    """).df()
    df["event_time"] = pd.to_datetime(df["event_time"])
    return df


def load_transfusions(con, stay_ids):
    ids   = ",".join(map(str, stay_ids))
    items = ",".join(map(str, TRANSFUSION_ITEMS))
    df = con.execute(f"""
        SELECT stay_id, starttime AS event_time, 1 AS transfusion
        FROM read_csv_auto('{ICU}/inputevents.csv.gz')
        WHERE stay_id IN ({ids}) AND itemid IN ({items})
    """).df()
    df["event_time"] = pd.to_datetime(df["event_time"])
    return df


def load_antibiotics(con, hadm_ids):
    ids = ",".join(map(str, hadm_ids))
    df = con.execute(f"""
        SELECT hadm_id, starttime AS event_time, 1 AS antibiotic
        FROM read_csv_auto('{HOSP}/prescriptions.csv.gz')
        WHERE hadm_id IN ({ids})
          AND starttime IS NOT NULL
          AND (
               LOWER(drug) LIKE '%vancomycin%'
            OR LOWER(drug) LIKE '%piperacillin%'
            OR LOWER(drug) LIKE '%meropenem%'
            OR LOWER(drug) LIKE '%cefepime%'
            OR LOWER(drug) LIKE '%metronidazole%'
            OR LOWER(drug) LIKE '%levofloxacin%'
            OR LOWER(drug) LIKE '%ciprofloxacin%'
            OR LOWER(drug) LIKE '%linezolid%'
            OR LOWER(drug) LIKE '%ceftriaxone%'
            OR LOWER(drug) LIKE '%ampicillin%'
            OR LOWER(drug) LIKE '%azithromycin%'
            OR LOWER(drug) LIKE '%daptomycin%'
          )
    """).df()
    df["event_time"] = pd.to_datetime(df["event_time"])
    return df


def align_to_hourly_grid(events_df, cohort_batch, action_col, join_col="stay_id"):
    merged = events_df.merge(
        cohort_batch[[join_col, "intime"]].drop_duplicates(), on=join_col
    )
    merged["intime"]      = pd.to_datetime(merged["intime"])
    merged["event_time"]  = pd.to_datetime(merged["event_time"])
    merged["hour_bucket"] = (
        (merged["event_time"] - merged["intime"]).dt.total_seconds() / 3600
    ).astype(int)
    merged = merged[merged["hour_bucket"] >= 0]
    return merged.groupby([join_col, "hour_bucket"])[action_col].max().reset_index()


def build_action_matrix(cohort, con):
    stay_ids = cohort["stay_id"].tolist()
    hadm_ids = cohort["hadm_id"].tolist()

    log.info("Loading interventions...")
    vaso  = load_vasopressors(con, stay_ids)
    fluid = load_fluids(con, stay_ids)
    rrt   = load_rrt(con, stay_ids)
    trans = load_transfusions(con, stay_ids)
    abx   = load_antibiotics(con, hadm_ids)

    log.info(f"  Raw event counts: vasopressor={len(vaso):,} fluid={len(fluid):,} "
             f"rrt={len(rrt):,} transfusion={len(trans):,} antibiotic={len(abx):,}")

    vaso_h  = align_to_hourly_grid(vaso,  cohort, "vasopressor")
    fluid_h = align_to_hourly_grid(fluid, cohort, "fluid_bolus")
    rrt_h   = align_to_hourly_grid(rrt,   cohort, "rrt")
    trans_h = align_to_hourly_grid(trans, cohort, "transfusion")

    abx = abx.merge(cohort[["stay_id", "hadm_id", "intime"]], on="hadm_id")
    abx["intime"]      = pd.to_datetime(abx["intime"])
    abx["event_time"]  = pd.to_datetime(abx["event_time"])
    abx["hour_bucket"] = (
        (abx["event_time"] - abx["intime"]).dt.total_seconds() / 3600
    ).astype(int)
    abx = abx[abx["hour_bucket"] >= 0]
    abx_h = abx.groupby(["stay_id", "hour_bucket"])["antibiotic"].max().reset_index()

    base = pd.DataFrame(
        [(int(sid), h)
         for sid, los in cohort.set_index("stay_id")["los"].items()
         for h in range(int(los * 24) + 1)],
        columns=["stay_id", "hour_bucket"]
    )

    for df, col in [(vaso_h, "vasopressor"), (fluid_h, "fluid_bolus"),
                    (rrt_h, "rrt"), (trans_h, "transfusion"), (abx_h, "antibiotic")]:
        base = base.merge(df, on=["stay_id", "hour_bucket"], how="left")

    action_cols = ["vasopressor", "fluid_bolus", "rrt", "transfusion", "antibiotic"]
    base[action_cols] = base[action_cols].fillna(0).astype(int)
    return base


def main():
    cohort = pd.read_parquet(DATA_DIR / "processed/cohort.parquet")
    con    = duckdb.connect()
    con.execute("SET memory_limit='6GB'")

    actions = build_action_matrix(cohort, con)
    out = OUT_DIR / "actions.parquet"
    actions.to_parquet(out, index=False)
    log.info(f"Saved -> {out}  ({actions.shape})")

    action_cols = ["vasopressor", "fluid_bolus", "rrt", "transfusion", "antibiotic"]
    print("\nAction prevalence (percent of hours with action):")
    print((actions[action_cols].mean() * 100).round(2).to_string())
    print("\nStays with action ever used:")
    for col in action_cols:
        n = (actions.groupby("stay_id")[col].max() > 0).sum()
        total = actions.stay_id.nunique()
        print(f"  {col:15s}: {n:,} / {total:,} stays ({n/total*100:.1f}%)")


if __name__ == "__main__":
    main()