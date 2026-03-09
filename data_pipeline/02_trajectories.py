import duckdb
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import logging
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MIMIC_ROOT = Path("mimic-iv-3.1")
HOSP       = MIMIC_ROOT / "hosp"
ICU        = MIMIC_ROOT / "icu"
DATA_DIR   = Path("data")
TRAJ_DIR   = DATA_DIR / "trajectories"; TRAJ_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR  = DATA_DIR / "cache";        CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ICU vitals — only available during ICU stay
CHART_ITEMS = {
    220045: "heart_rate",
    220052: "arterial_bp_mean",
    220210: "respiratory_rate",
    220277: "spo2",
    223761: "temperature_f",
    220739: "gcs_eye",
    223900: "gcs_verbal",
    223901: "gcs_motor",
}

# Labs — available across full hospital stay
LAB_ITEMS = {
    50813: "lactate",
    50912: "creatinine",
    51006: "bun",
    51301: "wbc",
    51222: "hemoglobin",
    50882: "bicarbonate",
    50931: "glucose_lab",
    50971: "potassium",
    50983: "sodium",
    50868: "anion_gap",
    50862: "albumin",
    51265: "platelets",
    50885: "bilirubin_total",
    50960: "magnesium",
}

RESAMPLE_FREQ        = "1h"
MAX_IMPUTE_GAP_HOURS = 4
MIN_OBS_FRACTION     = 0.10   # lowered from 0.30 — full hospital stay is sparser than ICU-only
MIN_HOURS            = 24     # exclude same-day discharges; aligns with literature standard
MAX_HOURS            = 336    # 14 days; removes extreme outliers that skew sequence models
BATCH_SIZE           = 100


def make_con():
    con = duckdb.connect()
    con.execute("SET memory_limit='6GB'")
    con.execute("SET threads=2")
    con.execute("SET preserve_insertion_order=false")
    return con


def prefilter_labs(con, cohort):
    cache_path = CACHE_DIR / "labs_filtered.parquet"
    if cache_path.exists():
        log.info(f"Lab cache found → {cache_path}")
        return str(cache_path)
    log.info("Pre-filtering labevents once (~5 min)...")
    hadm_ids = ",".join(map(str, cohort["hadm_id"].unique().tolist()))
    items    = ",".join(map(str, LAB_ITEMS.keys()))
    con.execute(f"""
        COPY (
            SELECT hadm_id, charttime, itemid, valuenum
            FROM read_csv_auto('{HOSP}/labevents.csv.gz')
            WHERE hadm_id IN ({hadm_ids})
              AND itemid   IN ({items})
              AND valuenum IS NOT NULL
        ) TO '{cache_path}' (FORMAT PARQUET)
    """)
    log.info(f"Lab cache saved → {cache_path}")
    return str(cache_path)


def prefilter_charts(con, cohort):
    cache_path = CACHE_DIR / "charts_filtered.parquet"
    if cache_path.exists():
        log.info(f"Chart cache found → {cache_path}")
        return str(cache_path)
    log.info("Pre-filtering chartevents once (~10-15 min)...")
    stay_ids = ",".join(map(str, cohort["stay_id"].unique().tolist()))
    items    = ",".join(map(str, CHART_ITEMS.keys()))
    con.execute(f"""
        COPY (
            SELECT stay_id, charttime, itemid, valuenum
            FROM read_csv_auto('{ICU}/chartevents.csv.gz')
            WHERE stay_id  IN ({stay_ids})
              AND itemid   IN ({items})
              AND valuenum IS NOT NULL
              AND valuenum > 0
        ) TO '{cache_path}' (FORMAT PARQUET)
    """)
    log.info(f"Chart cache saved → {cache_path}")
    return str(cache_path)


def prefilter_transfers(con, cohort):
    """Load unit location for all admissions."""
    cache_path = CACHE_DIR / "transfers_filtered.parquet"
    if cache_path.exists():
        log.info(f"Transfers cache found → {cache_path}")
        return str(cache_path)
    log.info("Pre-filtering transfers...")
    hadm_ids = ",".join(map(str, cohort["hadm_id"].unique().tolist()))
    con.execute(f"""
        COPY (
            SELECT hadm_id, intime, outtime, careunit, eventtype
            FROM read_csv_auto('{HOSP}/transfers.csv.gz')
            WHERE hadm_id IN ({hadm_ids})
              AND eventtype != 'discharge'
              AND careunit IS NOT NULL
        ) TO '{cache_path}' (FORMAT PARQUET)
    """)
    log.info(f"Transfers cache saved → {cache_path}")
    return str(cache_path)


def prefilter_services(con, cohort):
    """Load clinical service assignments for all admissions."""
    cache_path = CACHE_DIR / "services_filtered.parquet"
    if cache_path.exists():
        log.info(f"Services cache found → {cache_path}")
        return str(cache_path)
    log.info("Pre-filtering services...")
    hadm_ids = ",".join(map(str, cohort["hadm_id"].unique().tolist()))
    con.execute(f"""
        COPY (
            SELECT hadm_id, transfertime, curr_service
            FROM read_csv_auto('{HOSP}/services.csv.gz')
            WHERE hadm_id IN ({hadm_ids})
        ) TO '{cache_path}' (FORMAT PARQUET)
    """)
    log.info(f"Services cache saved → {cache_path}")
    return str(cache_path)


def prefilter_microbiology(con, cohort):
    """Load culture orders and results."""
    cache_path = CACHE_DIR / "microbiology_filtered.parquet"
    if cache_path.exists():
        log.info(f"Microbiology cache found → {cache_path}")
        return str(cache_path)
    log.info("Pre-filtering microbiology events...")
    hadm_ids = ",".join(map(str, cohort["hadm_id"].unique().tolist()))
    con.execute(f"""
        COPY (
            SELECT hadm_id, charttime,
                   1 AS culture_ordered,
                   CASE WHEN org_name IS NOT NULL THEN 1 ELSE 0 END AS positive_culture
            FROM read_csv_auto('{HOSP}/microbiologyevents.csv.gz')
            WHERE hadm_id IN ({hadm_ids})
              AND charttime IS NOT NULL
        ) TO '{cache_path}' (FORMAT PARQUET)
    """)
    log.info(f"Microbiology cache saved → {cache_path}")
    return str(cache_path)


def load_chart_for_batch(con, stay_ids, charts_parquet):
    ids = ",".join(map(str, stay_ids))
    df = con.execute(f"""
        SELECT stay_id, charttime, itemid, valuenum
        FROM read_parquet('{charts_parquet}')
        WHERE stay_id IN ({ids})
    """).df()
    df["charttime"] = pd.to_datetime(df["charttime"])
    df["variable"]  = df["itemid"].map(CHART_ITEMS)
    return df


def load_labs_for_batch(con, hadm_ids, labs_parquet):
    ids = ",".join(map(str, hadm_ids))
    df = con.execute(f"""
        SELECT hadm_id, charttime, itemid, valuenum
        FROM read_parquet('{labs_parquet}')
        WHERE hadm_id IN ({ids})
    """).df()
    df["charttime"] = pd.to_datetime(df["charttime"])
    df["variable"]  = df["itemid"].map(LAB_ITEMS)
    return df


def load_transfers_for_batch(con, hadm_ids, transfers_parquet):
    ids = ",".join(map(str, hadm_ids))
    df = con.execute(f"""
        SELECT hadm_id, intime, outtime, careunit
        FROM read_parquet('{transfers_parquet}')
        WHERE hadm_id IN ({ids})
    """).df()
    df["intime"]  = pd.to_datetime(df["intime"])
    df["outtime"] = pd.to_datetime(df["outtime"])
    return df


def load_services_for_batch(con, hadm_ids, services_parquet):
    ids = ",".join(map(str, hadm_ids))
    df = con.execute(f"""
        SELECT hadm_id, transfertime, curr_service
        FROM read_parquet('{services_parquet}')
        WHERE hadm_id IN ({ids})
    """).df()
    df["transfertime"] = pd.to_datetime(df["transfertime"])
    return df


def load_microbiology_for_batch(con, hadm_ids, micro_parquet):
    ids = ",".join(map(str, hadm_ids))
    df = con.execute(f"""
        SELECT hadm_id, charttime, culture_ordered, positive_culture
        FROM read_parquet('{micro_parquet}')
        WHERE hadm_id IN ({ids})
    """).df()
    df["charttime"] = pd.to_datetime(df["charttime"])
    return df


def get_careunit_at_hour(transfers_df, hadm_id, time_index):
    """For each hour, find which careunit the patient was in."""
    t = transfers_df[transfers_df.hadm_id == hadm_id].copy()
    if len(t) == 0:
        return pd.Series("unknown", index=time_index), pd.Series(0, index=time_index)

    careunits    = []
    is_icu_list  = []
    icu_keywords = ["intensive care", "icu", "ccu", "micu", "sicu", "cvicu", "nsicu"]

    for ts in time_index:
        match = t[(t.intime <= ts) & ((t.outtime >= ts) | t.outtime.isna())]
        unit  = match.iloc[-1]["careunit"] if len(match) > 0 else "unknown"
        careunits.append(unit)
        is_icu_list.append(1 if any(k in unit.lower() for k in icu_keywords) else 0)

    return pd.Series(careunits, index=time_index), pd.Series(is_icu_list, index=time_index)


def get_service_at_hour(services_df, hadm_id, time_index):
    """For each hour, find which clinical service the patient was under."""
    s = services_df[services_df.hadm_id == hadm_id].sort_values("transfertime")
    if len(s) == 0:
        return pd.Series("unknown", index=time_index)

    services = []
    for ts in time_index:
        past = s[s.transfertime <= ts]
        services.append(past.iloc[-1]["curr_service"] if len(past) > 0 else "unknown")
    return pd.Series(services, index=time_index)


def build_stay_timeseries(
    stay_id, hadm_id, hosp_admittime, hosp_dischtime, icu_intime, icu_outtime,
    chart_df, lab_df, transfers_df, services_df, micro_df
):
    # Episode window: full hospital stay
    time_index = pd.date_range(
        start=hosp_admittime.floor("h"),
        end=hosp_dischtime.ceil("h"),
        freq=RESAMPLE_FREQ
    )

    n_hours = len(time_index)

    # Apply min/max hour filters before doing any work
    if n_hours < MIN_HOURS:
        return None
    if n_hours > MAX_HOURS:
        return None

    # ── ICU vitals (only during ICU stay) ─────────────────────────────────────
    stay_chart = chart_df[chart_df.stay_id == stay_id]
    if len(stay_chart) > 0:
        chart_wide = (stay_chart
            .groupby(["charttime", "variable"])["valuenum"].mean()
            .unstack("variable").reindex(time_index))
    else:
        chart_wide = pd.DataFrame(index=time_index, columns=list(CHART_ITEMS.values()))

    # GCS components → total
    gcs_cols = ["gcs_eye", "gcs_verbal", "gcs_motor"]
    if all(c in chart_wide.columns for c in gcs_cols):
        chart_wide["gcs_total"] = chart_wide[gcs_cols].sum(axis=1, min_count=1)
        chart_wide.drop(columns=gcs_cols, inplace=True)

    # ── Labs (full hospital stay) ──────────────────────────────────────────────
    stay_labs = lab_df[lab_df.hadm_id == hadm_id]
    if len(stay_labs) > 0:
        labs_wide = (stay_labs
            .groupby(["charttime", "variable"])["valuenum"].mean()
            .unstack("variable").reindex(time_index))
    else:
        labs_wide = pd.DataFrame(index=time_index, columns=list(LAB_ITEMS.values()))

    ts = pd.concat([chart_wide, labs_wide], axis=1)

    # ── Location features ──────────────────────────────────────────────────────
    careunit_series, is_icu_series = get_careunit_at_hour(transfers_df, hadm_id, time_index)
    ts["is_icu"] = is_icu_series.values

    # ── Clinical service ───────────────────────────────────────────────────────
    service_series = get_service_at_hour(services_df, hadm_id, time_index)
    icu_services   = ["micu", "sicu", "tsicu", "csru", "ccu"]
    ts["is_icu_service"] = service_series.apply(
        lambda s: 1 if any(k in str(s).lower() for k in icu_services) else 0
    ).values

    # ── Infection signal ───────────────────────────────────────────────────────
    stay_micro   = micro_df[micro_df.hadm_id == hadm_id]
    micro_hourly = pd.DataFrame(index=time_index)
    if len(stay_micro) > 0:
        # Aggregate duplicate timestamps — take max per charttime
        stay_micro_agg = (stay_micro
                          .groupby("charttime")[["culture_ordered", "positive_culture"]]
                          .max())
        culture_ordered = (stay_micro_agg["culture_ordered"]
                           .reindex(time_index, method="nearest",
                                    tolerance=pd.Timedelta("1h"))
                           .fillna(0))
        positive_culture = (stay_micro_agg["positive_culture"]
                            .reindex(time_index, method="nearest",
                                     tolerance=pd.Timedelta("1h"))
                            .fillna(0).cummax())  # once positive stays positive
        micro_hourly["culture_ordered"]  = culture_ordered.values
        micro_hourly["positive_culture"] = positive_culture.values
    else:
        micro_hourly["culture_ordered"]  = 0
        micro_hourly["positive_culture"] = 0

    ts = pd.concat([ts, micro_hourly], axis=1)

    # ── Observation mask (clinical vars only) ──────────────────────────────────
    non_clinical  = ["is_icu", "is_icu_service", "culture_ordered", "positive_culture"]
    clinical_cols = [c for c in ts.columns if c not in non_clinical]
    mask          = ts[clinical_cols].notna().astype(int)
    mask.columns  = [f"{c}_obs" for c in mask.columns]

    if mask.values.mean() < MIN_OBS_FRACTION:
        return None

    # ── Imputation ─────────────────────────────────────────────────────────────
    ts[clinical_cols] = (ts[clinical_cols]
                         .infer_objects()
                         .ffill(limit=MAX_IMPUTE_GAP_HOURS)
                         .fillna(ts[clinical_cols].median()))

    # ── Normalization ──────────────────────────────────────────────────────────
    ts_z = (ts[clinical_cols] - ts[clinical_cols].mean()) / (
             ts[clinical_cols].std().replace(0, 1))
    ts_z = ts_z.fillna(0.0)

    # Rename columns
    ts_raw         = ts[clinical_cols].copy()
    ts_raw.columns = [f"{c}_raw"  for c in clinical_cols]
    ts_z.columns   = [f"{c}_norm" for c in clinical_cols]

    result = pd.concat([
        ts_raw,
        ts_z,
        mask,
        ts[non_clinical]
    ], axis=1)

    result.index.name = "time"
    result["stay_id"] = stay_id
    result["hadm_id"] = hadm_id
    result["hour"]    = (result.index - result.index[0]).total_seconds() / 3600

    # Tag phase
    result["phase"] = "ward"
    result.loc[result.index >= icu_intime,  "phase"] = "icu"
    result.loc[result.index >= icu_outtime, "phase"] = "post_icu"

    return result


def process_batch(con, batch, charts_parquet, labs_parquet,
                  transfers_parquet, services_parquet, micro_parquet):
    stay_ids = batch["stay_id"].tolist()
    hadm_ids = batch["hadm_id"].tolist()

    chart_df     = load_chart_for_batch(con, stay_ids, charts_parquet)
    lab_df       = load_labs_for_batch(con, hadm_ids, labs_parquet)
    transfers_df = load_transfers_for_batch(con, hadm_ids, transfers_parquet)
    services_df  = load_services_for_batch(con, hadm_ids, services_parquet)
    micro_df     = load_microbiology_for_batch(con, hadm_ids, micro_parquet)

    index_rows = []
    for _, row in batch.iterrows():
        ts = build_stay_timeseries(
            stay_id        = int(row["stay_id"]),
            hadm_id        = int(row["hadm_id"]),
            hosp_admittime = pd.Timestamp(row["hosp_admittime"]),
            hosp_dischtime = pd.Timestamp(row["hosp_dischtime"]),
            icu_intime     = pd.Timestamp(row["intime"]),
            icu_outtime    = pd.Timestamp(row["outtime"]),
            chart_df       = chart_df,
            lab_df         = lab_df,
            transfers_df   = transfers_df,
            services_df    = services_df,
            micro_df       = micro_df,
        )
        if ts is None:
            continue
        fname = TRAJ_DIR / f"{row['stay_id']}.parquet"
        ts.to_parquet(fname)
        index_rows.append({
            "stay_id":    int(row["stay_id"]),
            "hadm_id":    int(row["hadm_id"]),
            "subject_id": int(row["subject_id"]),
            "n_hours":    len(ts),
            "path":       str(fname),
        })
    return index_rows


def main():
    cohort = pd.read_parquet(DATA_DIR / "processed/cohort.parquet")
    log.info(f"Building trajectories for {len(cohort):,} stays...")
    log.info(f"Filters: MIN_OBS={MIN_OBS_FRACTION:.0%}  |  "
             f"MIN_HOURS={MIN_HOURS}  |  MAX_HOURS={MAX_HOURS} ({MAX_HOURS//24} days)")

    con = make_con()
    labs_parquet      = prefilter_labs(con, cohort)
    charts_parquet    = prefilter_charts(con, cohort)
    transfers_parquet = prefilter_transfers(con, cohort)
    services_parquet  = prefilter_services(con, cohort)
    micro_parquet     = prefilter_microbiology(con, cohort)

    already_done = {}
    for p in TRAJ_DIR.glob("*.parquet"):
        sid = int(p.stem)
        try:
            n = len(pd.read_parquet(p, columns=["hour"]))
        except Exception:
            n = 0
        already_done[sid] = (p, n)

    remaining = cohort[~cohort["stay_id"].astype(int).isin(already_done.keys())]
    log.info(f"Already done: {len(already_done):,}  |  Remaining: {len(remaining):,}")

    all_index = []
    for sid, (p, n) in already_done.items():
        row = cohort[cohort.stay_id == sid]
        if len(row):
            all_index.append({
                "stay_id":    sid,
                "hadm_id":    int(row.iloc[0]["hadm_id"]),
                "subject_id": int(row.iloc[0]["subject_id"]),
                "n_hours":    n,
                "path":       str(p),
            })

    batches = [remaining.iloc[i:i+BATCH_SIZE]
               for i in range(0, len(remaining), BATCH_SIZE)]
    for batch in tqdm(batches, desc="Processing batches"):
        rows = process_batch(con, batch, charts_parquet, labs_parquet,
                             transfers_parquet, services_parquet, micro_parquet)
        all_index.extend(rows)

    index_df = pd.DataFrame(all_index).drop_duplicates("stay_id")
    index_df.to_parquet(DATA_DIR / "processed/trajectory_index.parquet", index=False)

    log.info(f"Done. {len(index_df):,} trajectories saved → {TRAJ_DIR}")
    log.info(f"  n_hours: min={index_df.n_hours.min()}, "
             f"median={index_df.n_hours.median():.0f}, "
             f"max={index_df.n_hours.max()}")
    print(index_df.head())


if __name__ == "__main__":
    main()