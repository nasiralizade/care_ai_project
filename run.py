import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("care_ai.log"),
    ],
)
log = logging.getLogger(__name__)


def step_data():
    log.info("═══ STEP 1: Data Pipeline ═══")
    import subprocess
    for script in [
        "data_pipeline/01_cohort.py",
        "data_pipeline/02_trajectories.py",
        "data_pipeline/03_actions.py",
    ]:
        log.info(f"Running {script}...")
        result = subprocess.run([sys.executable, script], capture_output=False)
        if result.returncode != 0:
            log.error(f"Failed: {script}")
            sys.exit(1)
    log.info("Data pipeline complete")


step_data()
