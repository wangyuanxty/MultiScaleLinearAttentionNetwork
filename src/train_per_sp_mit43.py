"""MIT per-SP using ALL 43 degradable cells instead of the 10-cell subset.

Test of one hypothesis: the autoregressive rollout fails because the per-SP
protocol trains on too few cells (CALCE 3, NASA 3, TJU/PANASONIC/GOTION 2).

MIT is the only dataset that can answer it without a confound -- same chemistry,
same instrumentation, same lab -- and it has far more cells available than the
pipeline uses.  The discrepancy is a path bug, not a data shortage:

  * load_datasets.DATA_DIR is repo/data/mit_stanford, which does NOT exist;
    load_mit_stanford() therefore returns {} and every script silently falls
    back to checkpoints/data_cache/load_series_mit.pkl, which holds 10 cells.
  * the raw .mat files are outside the repo, at
    D:\\research\\degradation_prognostics\\mit_stanford\\ ; pointing DATA_DIR
    there yields 43 degradable cells (all batch2, 170-745 cycles each).

So this script loads all 43, holds out MIT_TEST_CELLS (2, unchanged), and
trains on the other 41 -- 4.1x the training cells of the cached split, with
nothing else different.

Protocol, architecture, target and metrics are imported from train_per_sp.py
verbatim; only the cell list and the output namespace differ.

Outputs (new paths; nothing existing is touched):
  checkpoints/per_sp/mit43/SP{sp}_seed{seed}.pt
  results/per_sp_train_mit43.json

    D:/anaconda/envs/py312/python.exe src/train_per_sp_mit43.py \
        --seeds 1 --start-seed 1 --sps 300
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import load_datasets as L                                   # noqa: E402
from train_per_sp import build_windows, train_one, eval_sp  # noqa: E402

MIT_RAW = Path(r"D:\research\degradation_prognostics")
W = 64
EOL_AH = 0.86          # rated 1.074 x 0.80, as in load_series('mit')
# Two of the 43 cells report a first-cycle capacity far above the rest (1.54 and
# 1.49 Ah against <=1.09 for the other 41).  Left in, they stretch the min--max
# range from 0.265 to 0.72 Ah, so every window is normalised into ~37% of [0,1]
# and the EOL threshold lands at 0.049 instead of 0.13.  The cached 10-cell
# split never contained them, which is why this only surfaces now.
MAX_CAP_AH = 1.2


def load_all_mit():
    """All degradable MIT cells, plus the 2-cell hold-out."""
    L.DATA_DIR = MIT_RAW           # the loader's default path does not exist
    caps = L.load_mit_stanford()
    if not caps:
        raise SystemExit(f"no MIT cells under {MIT_RAW}")
    from load_datasets import MIT_TEST_CELLS
    test_cells = list(MIT_TEST_CELLS)
    dropped = sorted(c for c in caps
                     if c not in set(test_cells)
                     and float(np.asarray(caps[c]).max()) > MAX_CAP_AH)
    train_cells = sorted(c for c in caps
                         if c not in set(test_cells) and c not in set(dropped))
    if dropped:
        print(f"  dropped {len(dropped)} outlier cells "
              f"(max > {MAX_CAP_AH} Ah): {dropped}", flush=True)
    return ({c: np.asarray(caps[c], dtype=np.float32) for c in caps},
            train_cells, test_cells)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--start-seed", type=int, default=1)
    ap.add_argument("--sps", type=int, nargs="+", default=[200, 300, 400])
    args = ap.parse_args()

    caps, train_cells, test_cells = load_all_mit()
    sps = [sp for sp in args.sps if sp > W]
    all_tr = np.concatenate([caps[c] for c in train_cells])
    lo, hi = float(all_tr.min()), float(all_tr.max())

    out_dir = os.path.join("..", "checkpoints", "per_sp", "mit43")
    os.makedirs(out_dir, exist_ok=True)
    out_path = "results/per_sp_train_mit43.json"
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}

    print(f"=== MIT all-cells  train={len(train_cells)} "
          f"test={len(test_cells)}  W={W}  eol={EOL_AH} Ah  SPs {sps} ===",
          flush=True)
    print(f"  cached split uses 8 train cells; this uses {len(train_cells)}",
          flush=True)
    for sp in sps:
        Xtr, Ytr = build_windows(caps, train_cells, lo, hi, W)
        Xts, Yts = build_windows(caps, test_cells, lo, hi, W, max_cycle=sp)
        X = np.vstack([Xtr, Xts])
        Y = np.concatenate([Ytr, Yts])
        print(f"  SP{sp}: {len(X)} windows ({len(Xtr)} train + {len(Xts)} "
              f"test-cell pre-SP)", flush=True)
        for seed in range(args.start_seed, args.start_seed + args.seeds):
            if str(seed) in out.setdefault("mit", {}).setdefault(str(sp), {}):
                print(f"    seed{seed}: SKIP (already in json)", flush=True)
                continue
            t0 = time.time()
            # prefix lets train_one drop an intermediate checkpoint every
            # PROG_SAVE epochs, so the rollout can be measured part-way
            prefix = os.path.join(out_dir, f"SP{sp}_seed{seed}")
            model = train_one(seed, X, Y, W, save_prefix=prefix,
                              save_meta={"lo": lo, "hi": hi, "sp": sp,
                                         "eol_ah": EOL_AH,
                                         "test_cells": test_cells,
                                         "train_cells": train_cells})
            torch.save({"state_dict": model.state_dict(), "seed": seed,
                        "lo": lo, "hi": hi, "W": W, "sp": sp, "eol_ah": EOL_AH,
                        "test_cells": test_cells, "train_cells": train_cells},
                       f"{prefix}.pt")
            rows = eval_sp(model, caps, test_cells, lo, hi, W, sp, EOL_AH)
            out["mit"][str(sp)][str(seed)] = rows
            json.dump(out, open(out_path, "w"), indent=2)
            g = lambda f: float(np.mean([r[f] for r in rows]))  # noqa: E731
            print(f"    SP{sp} seed{seed}: MAE={g('MAE'):.5f} "
                  f"R2={g('R2'):.4f} AE={g('AE'):.1f} "
                  f"[{time.time() - t0:.0f}s]", flush=True)
            del model
            torch.cuda.empty_cache()

    print("=== per-SP means over seeds ===", flush=True)
    for sp in sps:
        rs = [r for s in out["mit"].get(str(sp), {})
              for r in out["mit"][str(sp)][s]]
        if rs:
            print(f"  SP{sp}: MAE={np.mean([r['MAE'] for r in rs]):.5f} "
                  f"R2={np.mean([r['R2'] for r in rs]):.4f} "
                  f"AE={np.mean([r['AE'] for r in rs]):.2f}", flush=True)


if __name__ == "__main__":
    main()
