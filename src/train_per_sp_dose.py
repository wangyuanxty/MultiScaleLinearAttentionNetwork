"""MIT cell-count dose-response: does the autoregressive rollout improve with N?

The within-dataset counterpart to the synthetic learning curve.  The synthetic
pool can push N to 100, but it is a simulation and a reviewer will say so; this
is 43 real cells from one lab, one chemistry, one instrument.

N is the only thing that moves:

  * training sets are NESTED prefixes pool[:N] -- N=4 is a subset of N=8 is a
    subset of N=20 -- so going up in N only ADDS cells.  Independent draws would
    change the data and the count together, the confound that voided the NASA
    augmentation run.
  * lo/hi come from the full pool and are held constant across every N.  Taken
    from each arm's own prefix they would drift with N and move the
    normalisation scale at the same time as the cell count.  This also means the
    N=39 checkpoint trained earlier is NOT comparable here -- it used lo/hi from
    its own 39 cells -- which is why N=39 is left out rather than reused.
  * the per-SP split, architecture, epochs and target are train_per_sp.py's,
    unchanged.

Metrics follow the locked convention: MSE and AE, never a crossing rate.  The AR
rollout is 150 steps from EOL-50 regardless of how long the series is, so AE is
comparable across cells (see run_synth.eval_cell for why "run to the end of the
data" is not good enough).

Writes only new paths: checkpoints/per_sp/mitdose/ and results/per_sp_mit_dose.json.
Run from src/.

    D:/anaconda/envs/py312/python.exe train_per_sp_dose.py --ns 4 8 20 --seeds 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from run_synth import eval_cell                                # noqa: E402
from synth_battery import first_crossing                       # noqa: E402
from train_per_sp import (DEV, EPS, build_windows, eval_sp,    # noqa: E402
                          train_one)
from train_per_sp_mit43 import load_all_mit                    # noqa: E402

W = 64
EOL_AH = 0.86           # rated 1.074 x 0.80, as load_series('mit')
AR_STEPS = 150


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[4, 8, 20],
                    help="training-cell counts, taken as nested prefixes")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--sp", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()

    caps, train_cells, test_cells = load_all_mit()
    pool = list(train_cells)                    # already a fixed order; do not re-sort
    all_tr = np.concatenate([caps[c] for c in pool])
    lo, hi = float(all_tr.min()), float(all_tr.max())

    print("=" * 100)
    print(f"  MIT cell-count dose-response  pool={len(pool)} cells  "
          f"test={test_cells}  W={W}  SP{args.sp}  epochs={args.epochs}")
    print(f"  lo={lo:.4f} hi={hi:.4f} from ALL {len(pool)} cells, constant across "
          f"every N   N={args.ns}  seeds={args.seeds}")
    print("=" * 100)
    hdr = (f"  {'N':>4}{'seed':>5}{'win':>7}{'TF MAE':>10}{'TF R2':>9}"
           f"{'AR MSE@50':>12}{'AR MSE':>11}{'AR AE':>8}{'crossed':>9}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    os.makedirs(os.path.join("..", "checkpoints", "per_sp", "mitdose"),
                exist_ok=True)
    out = []
    for n in args.ns:
        cells = pool[:n]
        Xtr, Ytr = build_windows(caps, cells, lo, hi, W)
        Xts, Yts = build_windows(caps, test_cells, lo, hi, W, max_cycle=args.sp)
        X = np.vstack([Xtr, Xts])
        Y = np.concatenate([Ytr, Yts])
        for seed in args.seeds:
            t_wall = time.time()
            model = train_one(seed, X, Y, W)
            model.eval()
            rows = []
            for tc in test_cells:
                seq = (np.asarray(caps[tc], dtype=np.float64) - lo) / (hi - lo + EPS)
                eol = first_crossing(seq, (EOL_AH - lo) / (hi - lo + EPS)) + 1
                if eol <= 0:
                    continue
                r = eval_cell(model, caps[tc], eol, EOL_AH, lo, hi,
                              launch=50, steps=AR_STEPS)
                r["cell"] = tc
                rows.append(r)
            g = lambda f: float(np.mean([r[f] for r in rows]))      # noqa: E731
            ncross = sum(r["crossed"] for r in rows)
            print(f"  {n:>4}{seed:>5}{len(X):>7}{g('mae'):>10.5f}{g('r2'):>9.4f}"
                  f"{g('ar_mse50'):>12.3e}{g('ar_mse'):>11.3e}{g('ae'):>8.1f}"
                  f"{ncross:>5}/{len(rows):<3}  [{time.time() - t_wall:.0f}s]",
                  flush=True)
            out.append(dict(N=n, seed=seed, n_win=len(X), sp=args.sp,
                            epochs=args.epochs, lo=lo, hi=hi,
                            mae=g("mae"), r2=g("r2"),
                            ar_mse50=g("ar_mse50"), ar_mse=g("ar_mse"),
                            ae=g("ae"), n_crossed=ncross, n_test=len(rows),
                            per_cell=rows))
            torch.save({"state_dict": model.state_dict(), "W": W, "N": n,
                        "seed": seed, "lo": lo, "hi": hi, "sp": args.sp},
                       os.path.join("..", "checkpoints", "per_sp", "mitdose",
                                    f"SP{args.sp}_N{n}_seed{seed}.pt"))
            del model
            torch.cuda.empty_cache()
        print()

    p = "results/per_sp_mit_dose.json"
    prev = json.load(open(p)) if os.path.exists(p) else []
    json.dump(prev + out, open(p, "w"), indent=2)
    print(f"  saved {len(out)} rows to {p}")
    print(f"  AE is the mean over {len(test_cells)} held-out cells; the rollout is")
    print(f"  {AR_STEPS} steps from EOL-50, so a never-crossed row carries the")
    print(f"  bound {AR_STEPS - 50}.  'crossed' is reported but is NOT the metric.")


if __name__ == "__main__":
    main()
