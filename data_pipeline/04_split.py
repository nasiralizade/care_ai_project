import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path("data")
OUT_DIR  = DATA_DIR / "processed"

TRAIN_FRAC  = 0.70
VAL_FRAC    = 0.15
TEST_FRAC   = 0.15
RANDOM_SEED = 42


def main():
    index_df = pd.read_parquet(OUT_DIR / "trajectory_index.parquet")
    cohort   = pd.read_parquet(OUT_DIR / "cohort.parquet")

    log.info(f"Trajectory stays: {len(index_df):,} | "
             f"Unique patients: {index_df['subject_id'].nunique():,}")

    # Split at patient level to prevent leakage
    patients = index_df["subject_id"].unique()
    rng      = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(patients)

    n       = len(patients)
    n_train = int(n * TRAIN_FRAC)
    n_val   = int(n * VAL_FRAC)

    train_patients = set(patients[:n_train])
    val_patients   = set(patients[n_train:n_train + n_val])
    test_patients  = set(patients[n_train + n_val:])

    def assign_split(subject_id):
        if subject_id in train_patients:
            return "train"
        elif subject_id in val_patients:
            return "val"
        return "test"

    index_df["split"] = index_df["subject_id"].map(assign_split)

    splits = index_df[["stay_id", "hadm_id", "subject_id", "split"]]
    out    = OUT_DIR / "splits.parquet"
    splits.to_parquet(out, index=False)

    log.info(f"Splits saved → {out}")
    log.info(f"  train: {(splits.split == 'train').sum():,} stays "
             f"({len(train_patients):,} patients)")
    log.info(f"  val:   {(splits.split == 'val').sum():,} stays "
             f"({len(val_patients):,} patients)")
    log.info(f"  test:  {(splits.split == 'test').sum():,} stays "
             f"({len(test_patients):,} patients)")

    # Sanity check — no patient overlap between splits
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        pa = set(splits[splits.split == a]["subject_id"])
        pb = set(splits[splits.split == b]["subject_id"])
        assert len(pa & pb) == 0, f"Overlap between {a} and {b}"
    log.info("Sanity check passed — no patient overlap between splits")

    # Label distribution check
    cohort_labels = cohort[["stay_id", "readmit_30d", "died_in_hospital"]]
    splits_check  = splits.merge(cohort_labels, on="stay_id")
    for split in ["train", "val", "test"]:
        s = splits_check[splits_check.split == split]
        log.info(f"  {split:5s} — readmit_30d={s['readmit_30d'].mean():.2%}  "
                 f"mortality={s['died_in_hospital'].mean():.2%}")


if __name__ == "__main__":
    main()