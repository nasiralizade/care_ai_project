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

# ICU-specific interventions (from inputevents/procedureevents)
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


def load_anticoagulants(con, hadm_ids):
    ids = ",".join(map(str, hadm_ids))
    df = con.execute(f"""
        SELECT hadm_id, starttime AS event_time, 1 AS anticoagulant
        FROM read_csv_auto('{HOSP}/prescriptions.csv.gz')
        WHERE hadm_id IN ({ids})
          AND starttime IS NOT NULL
          AND drug_type = 'MAIN'
          AND (
               LOWER(drug) LIKE '%heparin%'
            OR LOWER(drug) LIKE '%warfarin%'
            OR LOWER(drug) LIKE '%enoxaparin%'
            OR LOWER(drug) LIKE '%apixaban%'
            OR LOWER(drug) LIKE '%rivaroxaban%'
            OR LOWER(drug) LIKE '%fondaparinux%'
          )
    """).df()
    df["event_time"] = pd.to_datetime(df["event_time"])
    return df


def load_diuretics(con, hadm_ids):
    ids = ",".join(map(str, hadm_ids))
    df = con.execute(f"""
        SELECT hadm_id, starttime AS event_time, 1 AS diuretic
        FROM read_csv_auto('{HOSP}/prescriptions.csv.gz')
        WHERE hadm_id IN ({ids})
          AND starttime IS NOT NULL
          AND drug_type = 'MAIN'
          AND (
               LOWER(drug) LIKE '%furosemide%'
            OR LOWER(drug) LIKE '%torsemide%'
            OR LOWER(drug) LIKE '%bumetanide%'
            OR LOWER(drug) LIKE '%metolazone%'
            OR LOWER(drug) LIKE '%hydrochlorothiazide%'
          )
    """).df()
    df["event_time"] = pd.to_datetime(df["event_time"])
    return df


def load_steroids(con, hadm_ids):
    ids = ",".join(map(str, hadm_ids))
    df = con.execute(f"""
        SELECT hadm_id, starttime AS event_time, 1 AS steroid
        FROM read_csv_auto('{HOSP}/prescriptions.csv.gz')
        WHERE hadm_id IN ({ids})
          AND starttime IS NOT NULL
          AND drug_type = 'MAIN'
          AND (
               LOWER(drug) LIKE '%methylprednisolone%'
            OR LOWER(drug) LIKE '%prednisone%'
            OR LOWER(drug) LIKE '%dexamethasone%'
            OR LOWER(drug) LIKE '%hydrocortisone%'
          )
    """).df()
    df["event_time"] = pd.to_datetime(df["event_time"])
    return df


def load_insulin(con, hadm_ids):
    ids = ",".join(map(str, hadm_ids))
    df = con.execute(f"""
        SELECT hadm_id, starttime AS event_time, 1 AS insulin
        FROM read_csv_auto('{HOSP}/prescriptions.csv.gz')
        WHERE hadm_id IN ({ids})
          AND starttime IS NOT NULL
          AND drug_type = 'MAIN'
          AND LOWER(drug) LIKE '%insulin%'
    """).df()
    df["event_time"] = pd.to_datetime(df["event_time"])
    return df


def load_icu_transfers(con, hadm_ids):
    """Load ICU escalation and stepdown events from transfers."""
    ids = ",".join(map(str, hadm_ids))
    df = con.execute(f"""
        SELECT hadm_id, intime AS event_time, careunit
        FROM read_csv_auto('{HOSP}/transfers.csv.gz')
        WHERE hadm_id IN ({ids})
          AND careunit IS NOT NULL
          AND eventtype != 'discharge'
    """).df()
    df["event_time"] = pd.to_datetime(df["event_time"])
    icu_keywords = ["intensive care", "icu", "ccu", "micu", "sicu", "cvicu", "nsicu"]
    df["is_icu"] = df["careunit"].str.lower().apply(
        lambda x: any(k in x for k in icu_keywords)
    ).astype(int)
    # Escalation = moving INTO ICU, stepdown = moving OUT of ICU
    df = df.sort_values(["hadm_id", "event_time"])
    df["prev_icu"] = df.groupby("hadm_id")["is_icu"].shift(1).fillna(0)
    df["icu_escalation"] = ((df["is_icu"] == 1) & (df["prev_icu"] == 0)).astype(int)
    df["icu_stepdown"]   = ((df["is_icu"] == 0) & (df["prev_icu"] == 1)).astype(int)
    return df[["hadm_id", "event_time", "icu_escalation", "icu_stepdown"]]


def load_discharge(con, hadm_ids):
    """Mark the hour of hospital discharge as an action."""
    ids = ",".join(map(str, hadm_ids))
    df = con.execute(f"""
        SELECT hadm_id, dischtime AS event_time, 1 AS discharged
        FROM read_csv_auto('{HOSP}/admissions.csv.gz')
        WHERE hadm_id IN ({ids})
          AND dischtime IS NOT NULL
    """).df()
    df["event_time"] = pd.to_datetime(df["event_time"])
    return df


def align_to_hourly_grid(events_df, cohort_batch, action_col, join_col="stay_id"):
    """Align events to hourly grid, rebased from hosp_admittime."""
    ref_time_col = "hosp_admittime"
    merged = events_df.merge(
        cohort_batch[[join_col, ref_time_col]].drop_duplicates(), on=join_col
    )
    merged[ref_time_col]  = pd.to_datetime(merged[ref_time_col])
    merged["event_time"]  = pd.to_datetime(merged["event_time"])
    merged["hour_bucket"] = (
        (merged["event_time"] - merged[ref_time_col]).dt.total_seconds() / 3600
    ).astype(int)
    merged = merged[merged["hour_bucket"] >= 0]
    return merged.groupby([join_col, "hour_bucket"])[action_col].max().reset_index()


def align_hadm_to_hourly_grid(events_df, cohort_batch, action_col):
    """Same as align_to_hourly_grid but joins on hadm_id."""
    merged = events_df.merge(
        cohort_batch[["hadm_id", "stay_id", "hosp_admittime"]].drop_duplicates(),
        on="hadm_id"
    )
    merged["hosp_admittime"] = pd.to_datetime(merged["hosp_admittime"])
    merged["event_time"]     = pd.to_datetime(merged["event_time"])
    merged["hour_bucket"]    = (
        (merged["event_time"] - merged["hosp_admittime"]).dt.total_seconds() / 3600
    ).astype(int)
    merged = merged[merged["hour_bucket"] >= 0]
    return merged.groupby(["stay_id", "hour_bucket"])[action_col].max().reset_index()


def build_action_matrix(cohort, con):
    stay_ids = cohort["stay_id"].tolist()
    hadm_ids = cohort["hadm_id"].tolist()

    log.info("Loading interventions...")
    vaso   = load_vasopressors(con, stay_ids)
    fluid  = load_fluids(con, stay_ids)
    rrt    = load_rrt(con, stay_ids)
    trans  = load_transfusions(con, stay_ids)
    abx    = load_antibiotics(con, hadm_ids)
    anticoag = load_anticoagulants(con, hadm_ids)
    diur   = load_diuretics(con, hadm_ids)
    ster   = load_steroids(con, hadm_ids)
    ins    = load_insulin(con, hadm_ids)
    transf = load_icu_transfers(con, hadm_ids)
    disch  = load_discharge(con, hadm_ids)

    log.info(f"  Raw event counts: "
             f"vasopressor={len(vaso):,} fluid={len(fluid):,} "
             f"rrt={len(rrt):,} transfusion={len(trans):,} "
             f"antibiotic={len(abx):,} anticoagulant={len(anticoag):,} "
             f"diuretic={len(diur):,} steroid={len(ster):,} "
             f"insulin={len(ins):,} transfers={len(transf):,}")

    # Align ICU actions (joined via stay_id)
    vaso_h  = align_to_hourly_grid(vaso,  cohort, "vasopressor")
    fluid_h = align_to_hourly_grid(fluid, cohort, "fluid_bolus")
    rrt_h   = align_to_hourly_grid(rrt,   cohort, "rrt")
    trans_h = align_to_hourly_grid(trans, cohort, "transfusion")

    # Align hospital actions (joined via hadm_id → stay_id)
    abx_h      = align_hadm_to_hourly_grid(abx,      cohort, "antibiotic")
    anticoag_h = align_hadm_to_hourly_grid(anticoag, cohort, "anticoagulant")
    diur_h     = align_hadm_to_hourly_grid(diur,     cohort, "diuretic")
    ster_h     = align_hadm_to_hourly_grid(ster,     cohort, "steroid")
    ins_h      = align_hadm_to_hourly_grid(ins,      cohort, "insulin")

    # ICU escalation/stepdown — joined via hadm_id
    esc_h  = align_hadm_to_hourly_grid(
        transf[transf.icu_escalation == 1][["hadm_id", "event_time", "icu_escalation"]],
        cohort, "icu_escalation"
    )
    step_h = align_hadm_to_hourly_grid(
        transf[transf.icu_stepdown == 1][["hadm_id", "event_time", "icu_stepdown"]],
        cohort, "icu_stepdown"
    )

    # Discharge flag
    disch_h = align_hadm_to_hourly_grid(disch, cohort, "discharged")

    # Build base grid: one row per (stay_id, hour_bucket)
    # Now based on full hospital LOS, not just ICU LOS
    cohort["hosp_los_hours"] = (
        (pd.to_datetime(cohort["hosp_dischtime"]) -
         pd.to_datetime(cohort["hosp_admittime"]))
        .dt.total_seconds() / 3600
    ).astype(int)

    base = pd.DataFrame(
        [(int(sid), h)
         for sid, los in cohort.set_index("stay_id")["hosp_los_hours"].items()
         for h in range(los + 1)],
        columns=["stay_id", "hour_bucket"]
    )

    # Merge all actions
    action_dfs = [
        (vaso_h,      "vasopressor"),
        (fluid_h,     "fluid_bolus"),
        (rrt_h,       "rrt"),
        (trans_h,     "transfusion"),
        (abx_h,       "antibiotic"),
        (anticoag_h,  "anticoagulant"),
        (diur_h,      "diuretic"),
        (ster_h,      "steroid"),
        (ins_h,       "insulin"),
        (esc_h,       "icu_escalation"),
        (step_h,      "icu_stepdown"),
        (disch_h,     "discharged"),
    ]

    for df, col in action_dfs:
        base = base.merge(df, on=["stay_id", "hour_bucket"], how="left")

    action_cols = [col for _, col in action_dfs]
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

    action_cols = ["vasopressor", "fluid_bolus", "rrt", "transfusion",
                   "antibiotic", "anticoagulant", "diuretic", "steroid",
                   "insulin", "icu_escalation", "icu_stepdown", "discharged"]
    print("\nAction prevalence (percent of hours with action):")
    print((actions[action_cols].mean() * 100).round(2).to_string())
    print("\nStays with action ever used:")
    for col in action_cols:
        n = (actions.groupby("stay_id")[col].max() > 0).sum()
        total = actions.stay_id.nunique()
        print(f"  {col:18s}: {n:,} / {total:,} stays ({n/total*100:.1f}%)")


if __name__ == "__main__":
    main()