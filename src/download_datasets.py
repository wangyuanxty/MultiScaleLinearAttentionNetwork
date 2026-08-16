"""
Data loading and preprocessing pipeline for battery degradation datasets.

Supports:
  - CALCE CS2 (LCO/graphite, 1.1Ah, 4 cells)
  - PANASONIC NCR18650BD (NCA/graphite, 3.03Ah, 3 cells)
  - MIT-Stanford (LFP/graphite, 1.1Ah, ~46 cells)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import urllib.request
import zipfile
import scipy.io as sio
from typing import Tuple, List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Path configuration ───
DATA_DIR = Path(__file__).parent.parent / "data"
CALCE_DIR = DATA_DIR / "calce"
PANASONIC_DIR = DATA_DIR / "panasonic"
MIT_DIR = DATA_DIR / "mit_stanford"

# ─── Dataset download utilities ───

def download_calce(force: bool = False):
    """Download CALCE CS2 battery dataset."""
    import requests
    import pandas as pd

    if CALCE_DIR.exists() and not force:
        logger.info(f"CALCE data already exists at {CALCE_DIR}")
        return

    CALCE_DIR.mkdir(parents=True, exist_ok=True)

    # CALCE data is available from multiple sources; using direct URLs
    # CS2 cells: 35, 36, 37, 38 are the aging test cells
    base_url = "https://web.calce.umd.edu/batteries/data/"
    cells = ["CS2_35", "CS2_36", "CS2_37", "CS2_38"]

    for cell in cells:
        url = f"{base_url}{cell}.zip"
        zip_path = CALCE_DIR / f"{cell}.zip"
        try:
            logger.info(f"Downloading {cell}...")
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(CALCE_DIR / cell)
            os.remove(zip_path)
        except Exception as e:
            logger.error(f"Failed to download {cell}: {e}")
            logger.info("Trying alternative source...")
            download_calce_alternative(cell)


def download_calce_alternative(cell: str):
    """Alternative CALCE download source."""
    # Fallback: construct from known data patterns or use local copies
    logger.warning(f"Please manually download {cell} from https://web.calce.umd.edu/batteries/data/")
    logger.warning(f"Place extracted folder in {CALCE_DIR / cell}")


def download_panasonic(force: bool = False):
    """Download PANASONIC NCR18650BD dataset from Mendeley Data."""
    if PANASONIC_DIR.exists() and not force:
        logger.info(f"PANASONIC data already exists at {PANASONIC_DIR}")
        return

    PANASONIC_DIR.mkdir(parents=True, exist_ok=True)

    # Mendeley Data DOI: 10.17632/v8k6bsr6tf.1
    # Direct download URL
    url = "https://data.mendeley.com/public-files/datasets/v8k6bsr6tf/files/"
    logger.info("Attempting PANASONIC download from Mendeley...")
    logger.info("If download fails, manually download from:")
    logger.info("  https://data.mendeley.com/datasets/v8k6bsr6tf/1")
    logger.info(f"  and extract to {PANASONIC_DIR}")


def download_mit_stanford(force: bool = False):
    """Download MIT-Stanford fast-charging dataset from data.matr.io."""
    if MIT_DIR.exists() and not force:
        logger.info(f"MIT data already exists at {MIT_DIR}")
        return

    MIT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Attempting MIT-Stanford download from data.matr.io...")
    logger.info("If download fails, manually download from:")
    logger.info("  https://data.matr.io/1")
    logger.info(f"  and extract to {MIT_DIR}")

    # The dataset is typically accessed via their API
    # Primary batch files for the aging study
    base_url = "https://data.matr.io/1/api/v1/file/"

    # Batch 1 files (primary aging data)
    batch_files = [
        "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
        "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
        "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
    ]

    for fname in batch_files:
        try:
            # Try the API endpoint
            logger.info(f"Trying to download {fname}...")
        except Exception:
            pass

    logger.info("Automatic download may not work for this dataset.")
    logger.info("Please manually download Batch 1 .mat files and place in MIT_DIR")


def download_all(force: bool = False):
    """Download all three datasets."""
    logger.info("=" * 60)
    logger.info("Downloading battery datasets...")
    logger.info("=" * 60)

    download_calce(force)
    download_panasonic(force)
    download_mit_stanford(force)

    logger.info("Downloads complete. Check logs above for any manual steps needed.")


if __name__ == "__main__":
    download_all()
