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

CHART_ITEMS = {
    220045: "heart_rate",
    220050: "arterial_bp_systolic",
    220051: "arterial_bp_diastolic",
    220052: "arterial_bp_mean",
    220210: "respiratory_rate",
    220277: "spo2",
    223761: "temperature_f",
    220739: "gcs_eye",
    223900: "gcs_verbal",
    223901: "gcs_motor",
}

LAB_ITEMS = {
    50813: "lactate",
    50912: "creatinine",
    51301: "wbc",
    51222: "hemoglobin",
    50882: "bicarbonate",
    50931: "glucose_lab",
    50971: "potassium",
    50983: "sodium",
}

RESAMPLE_FREQ        = "1h"
MAX_IMPUTE_GAP_HOURS = 4
MIN_OBS_FRACTION     = 0.3
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


def build_stay_timeseries(stay_id, intime, outtime, chart_df, lab_df, hadm_id):
    time_index = pd.date_range(
        start=intime.floor("h"), end=outtime.ceil("h"), freq=RESAMPLE_FREQ
    )
    if len(time_index) < 6:
        return None

    stay_chart = chart_df[chart_df.stay_id == stay_id]
    if len(stay_chart) > 0:
        chart_wide = (stay_chart
            .groupby(["charttime", "variable"])["valuenum"].mean()
            .unstack("variable").reindex(time_index))
    else:
        chart_wide = pd.DataFrame(index=time_index, columns=list(CHART_ITEMS.values()))

    stay_labs = lab_df[lab_df.hadm_id == hadm_id]
    if len(stay_labs) > 0:
        labs_wide = (stay_labs
            .groupby(["charttime", "variable"])["valuenum"].mean()
            .unstack("variable").reindex(time_index))
    else:
        labs_wide = pd.DataFrame(index=time_index, columns=list(LAB_ITEMS.values()))

    ts = pd.concat([chart_wide, labs_wide], axis=1)

    gcs_cols = ["gcs_eye", "gcs_verbal", "gcs_motor"]
    if all(c in ts.columns for c in gcs_cols):
        ts["gcs_total"] = ts[gcs_cols].sum(axis=1, min_count=1)
        ts.drop(columns=gcs_cols, inplace=True)

    mask = ts.notna().astype(int)
    mask.columns = [f"{c}_obs" for c in mask.columns]

    if mask.values.mean() < MIN_OBS_FRACTION:
        return None

    ts = ts.infer_objects(copy=False).ffill(limit=MAX_IMPUTE_GAP_HOURS)
    ts = ts.fillna(ts.median())

    ts_z = (ts - ts.mean()) / (ts.std().replace(0, 1))
    # TODO FIX 1: fill any residual NaN in norm space with 0 (mean in z-score space)
    ts_z = ts_z.fillna(0.0)

    ts.columns   = [f"{c}_raw"  for c in ts.columns]
    ts_z.columns = [f"{c}_norm" for c in ts_z.columns]

    result = pd.concat([ts, ts_z, mask], axis=1)
    result.index.name = "time"
    result["stay_id"] = stay_id
    result["hour"]    = (result.index - result.index[0]).total_seconds() / 3600
    return result


def process_batch(con, batch, charts_parquet, labs_parquet):
    stay_ids = batch["stay_id"].tolist()
    hadm_ids = batch["hadm_id"].tolist()
    chart_df = load_chart_for_batch(con, stay_ids, charts_parquet)
    lab_df   = load_labs_for_batch(con, hadm_ids, labs_parquet)

    index_rows = []
    for _, row in batch.iterrows():
        ts = build_stay_timeseries(
            stay_id=int(row["stay_id"]),
            intime=pd.Timestamp(row["intime"]),
            outtime=pd.Timestamp(row["outtime"]),
            chart_df=chart_df,
            lab_df=lab_df,
            hadm_id=int(row["hadm_id"]),
        )
        if ts is None:
            continue
        fname = TRAJ_DIR / f"{row['stay_id']}.parquet"
        ts.to_parquet(fname)
        index_rows.append({
            "stay_id":    int(row["stay_id"]),
            "hadm_id":    int(row["hadm_id"]),
            "subject_id": int(row["subject_id"]),
            "n_hours":    len(ts),   # TODO FIX: actual length, not 0
            "path":       str(fname),
        })
    return index_rows


def main():
    cohort = pd.read_parquet(DATA_DIR / "processed/cohort.parquet")
    log.info(f"Building trajectories for {len(cohort):,} stays...")

    con = make_con()
    labs_parquet   = prefilter_labs(con, cohort)
    charts_parquet = prefilter_charts(con, cohort)

    # TODO FIX: rebuild index with real n_hours from existing files
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

    # Seed index with correct n_hours from existing files
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

    batches = [remaining.iloc[i:i+BATCH_SIZE] for i in range(0, len(remaining), BATCH_SIZE)]
    for batch in tqdm(batches, desc="Processing batches"):
        rows = process_batch(con, batch, charts_parquet, labs_parquet)
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