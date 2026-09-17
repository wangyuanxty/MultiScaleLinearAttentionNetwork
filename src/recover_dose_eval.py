"""Recover dose-response rows from checkpoints whose run was killed early.

`train_per_sp_dose.py` saves a checkpoint per (N, seed) the moment that arm
finishes, but writes the JSON only once, after every arm.  A run killed in
between therefore leaves finished arms on disk with no record of their
metrics.  This re-derives those rows from the checkpoints, so the finished
training is not thrown away.

Nothing is re-implemented.  The rollout and the metrics come from the same
`run_synth.eval_cell` / `synth_battery.first_crossing` / `train_per_sp.
build_windows` that `train_per_sp_dose.py` itself calls, and the normalisation
comes from each checkpoint's own stored `lo`/`hi`, so a recovered row is the
row the interrupted run would have written.

Rows already present in results/per_sp_mit_dose.json are left alone; only
(N, seed) pairs missing from it are appended.

    D:/anaconda/envs/py312/python.exe src/recover_dose_eval.py
    D:/anaconda/envs/py312/python.exe src/recover_dose_eval.py --dry-run
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import torch

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from run_synth import eval_cell                          # noqa: E402
from synth_battery import first_crossing                 # noqa: E402
from train_per_sp import DEV, EPS, build_windows         # noqa: E402
from train_per_sp_mit43 import load_all_mit              # noqa: E402
from gdn_model import build_gdn_model                    # noqa: E402

EOL_AH = 0.86           # rated 1.074 x 0.80, as load_series('mit')
AR_STEPS = 150          # the dose script's fixed rollout length
CKPT_DIR = os.path.join("..", "checkpoints", "per_sp", "mitdose")
OUT = os.path.join("results", "per_sp_mit_dose.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="evaluate and print, but do not touch the JSON")
    args = ap.parse_args()

    caps, train_cells, test_cells = load_all_mit()
    pool = list(train_cells)                    # same fixed order as the trainer

    out = json.load(open(OUT)) if os.path.exists(OUT) else []
    have = {(r["N"], r["seed"]) for r in out}
    print(f"  json holds {len(out)} rows: "
          f"{sorted((r['N'], r['seed']) for r in out)}", flush=True)

    paths = sorted(glob.glob(os.path.join(CKPT_DIR, "SP*_N*_seed*.pt")))
    if not paths:
        raise SystemExit(f"no checkpoints under {CKPT_DIR}")

    added = []
    for p in paths:
        m = re.search(r"_N(\d+)_seed(\d+)\.pt$", p)
        if not m:
            continue
        n, seed = int(m.group(1)), int(m.group(2))
        if (n, seed) in have:
            print(f"  N={n:<3} seed={seed}: already in json, skip", flush=True)
            continue

        ck = torch.load(p, map_location=DEV, weights_only=False)
        W = int(ck["W"])
        lo, hi = float(ck["lo"]), float(ck["hi"])
        sp = int(ck.get("sp", 300))

        model = build_gdn_model(
            multiscale=True, stage_query=True, input_dim=1, window_size=W,
            output_len=1, readout="last",
        ).to(DEV)
        model.load_state_dict(ck["state_dict"])
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

        # window count the trainer would have reported for this arm
        Xtr, _ = build_windows(caps, pool[:n], lo, hi, W)
        Xts, _ = build_windows(caps, test_cells, lo, hi, W, max_cycle=sp)
        n_win = len(Xtr) + len(Xts)

        g = lambda f: float(np.mean([r[f] for r in rows]))       # noqa: E731
        ncross = sum(r["crossed"] for r in rows)
        row = dict(N=n, seed=seed, n_win=n_win, sp=sp, epochs=100, lo=lo, hi=hi,
                   mae=g("mae"), r2=g("r2"), ar_mse50=g("ar_mse50"),
                   ar_mse=g("ar_mse"), ae=g("ae"), n_crossed=ncross,
                   n_test=len(rows), per_cell=rows)
        print(f"  N={n:<3} seed={seed}: MAE={row['mae']:.5f} R2={row['r2']:.4f} "
              f"AR_AE={row['ae']:.1f} crossed={ncross}/{len(rows)} "
              f"win={n_win}  [recovered from {os.path.basename(p)}]", flush=True)
        added.append(row)
        have.add((n, seed))

    if args.dry_run:
        print(f"\n  DRY RUN: {len(added)} row(s) would be added; JSON untouched")
        return
    if not added:
        print("\n  nothing to add")
        return

    json.dump(out + added, open(OUT, "w"), indent=2)
    print(f"\n  appended {len(added)} row(s) to {OUT}; now {len(out) + len(added)} total")
    print("  note: the rollout is 150 steps from EOL-50, so a never-crossed row "
          "carries the bound 100; 'crossed' is reported but is NOT the metric.")


if __name__ == "__main__":
    main()
