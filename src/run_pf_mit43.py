"""PatchFormer per-SP on the FULL MIT split (39 training cells, not 9).

The self-run PatchFormer baseline was trained on the 10-cell cache split
(load_series_mit.pkl): 9 training cells and batch2_cell5 held out.  MIT has 43
degradable cells; the pipeline only ever saw 10 of them because
load_datasets.DATA_DIR points at a directory that does not exist and every
caller silently falls back to the cache.

This is the same experiment as train_per_sp_mit43.py, on the baseline, so the
two can be compared cell-for-cell:

    series     41 cells = all 43 minus the 2 outliers (first-cycle capacity
               1.54 / 1.49 Ah against <=1.09 for the rest; left in they would
               stretch the min--max range from 0.265 to 0.72 Ah)
    held out   batch2_cell47 removed from the series entirely, batch2_cell5 is
               the test cell -- so the training set is 43-2-2 = 39 cells,
               matching ours exactly
    everything else -- per-SP split, EncoderNormalizer decode, SMAPE, early
               stopping, official rul_value_error -- is run_pf_nasa_adapted.py
               verbatim; this file only swaps the series source and the output
               namespace by reassigning two module globals.

Run with the `patchformer` conda env:
    D:/anaconda/envs/patchformer/python.exe src/run_pf_mit43.py --count 1
"""
from __future__ import annotations

import functools
import os
import sys
from pathlib import Path

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import run_pf_nasa_adapted as R          # noqa: E402  (chdirs into ref_patchformer)

MIT_RAW = Path(r"D:\research\degradation_prognostics")
MAX_CAP_AH = 1.2
DROP_EXTRA = {"batch2_cell47"}          # second hold-out, kept out of training


@functools.lru_cache(maxsize=None)      # the h5py parse is slow; build_dfs
def mit43_series():                     # calls this once per start point
    """The 39-cell training pool + the test cell, as PatchFormer DataFrames."""
    import pandas as pd
    import load_datasets as L
    L.DATA_DIR = MIT_RAW
    caps = L.load_mit_stanford()
    if not caps:
        raise SystemExit(f"no MIT cells under {MIT_RAW}")
    keep = [c for c in sorted(caps)
            if float(np.asarray(caps[c]).max()) <= MAX_CAP_AH
            and c not in DROP_EXTRA]
    out = {}
    for c in keep:
        arr = np.asarray(caps[c], dtype=np.float64)
        out[c] = pd.DataFrame({"Cycle": np.arange(1, len(arr) + 1),
                               "Capacity": arr})
    return out


def main() -> None:
    orig = R._series_dict

    def patched(ds):
        return mit43_series() if ds == "mit" else orig(ds)

    R._series_dict = patched
    R.DATASETS["mit"] = dict(
        R.DATASETS["mit"],
        out_dir="results_MIT43_RUL_prediction_sl_64",
        json_out="results/pf_mit43_selfrun_batch2_cell5.json",
    )
    s = mit43_series()
    print(f"  MIT series: {len(s)} cells (test {R.DATASETS['mit']['test']}, "
          f"so {len(s) - 1} training cells)", flush=True)
    R.main()


if __name__ == "__main__":
    main()
