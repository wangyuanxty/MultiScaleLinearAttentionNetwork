"""Evaluate existing per-SP checkpoints (train_per_sp.py artifacts).

Re-runs only the evaluation on saved per-SP checkpoints and prints
10-seed per-SP means — used when an SP was trained but its numbers
never made it into a stored JSON (e.g. our PANASONIC [300,400,500]
set, whose 10 seeds exist as SP300/SP400/SP500 checkpoint files).

Run:  python src/eval_per_sp_existing.py --dataset panasonic
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from gdn_model import build_gdn_model
from make_figures import load_series
from train_per_sp import eval_sp

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--start-seed", type=int, default=1)
    ap.add_argument("--sps", type=int, nargs="+", default=None)
    args = ap.parse_args()

    ds = args.dataset
    caps, train_cells, test_cell, W, sps, eol_ah = load_series(ds)
    if args.sps:
        sps = args.sps
    caps = {c: caps[c].astype(np.float32) for c in caps}
    test_cells = [test_cell] if isinstance(test_cell, str) else list(test_cell)

    all_tr = np.concatenate([caps[c] for c in train_cells])
    lo, hi = float(all_tr.min()), float(all_tr.max())

    ckpt_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "checkpoints", "per_sp", ds))

    for sp in sps:
        rows_all = []
        for seed in range(args.start_seed, args.start_seed + args.seeds):
            path = f"{ckpt_dir}/SP{sp}_seed{seed}.pt"
            if not os.path.exists(path):
                print(f"SP{sp} seed{seed}: NO CKPT", flush=True)
                continue
            ck = torch.load(path, map_location=DEV, weights_only=False)
            model = build_gdn_model(
                multiscale=True, stage_query=True, input_dim=1,
                window_size=W, output_len=1, readout="last").to(DEV)
            model.load_state_dict(ck["state_dict"])
            rows = eval_sp(model, caps, test_cells, lo, hi, W, sp, eol_ah)
            rows_all.append(rows)
            print(f"SP{sp} seed{seed}: AE={[r['AE'] for r in rows]} "
                  f"MAE={np.mean([r['MAE'] for r in rows]):.4f}",
                  flush=True)
        if not rows_all:
            continue
        vals = {m: [] for m in ["AE", "MAE", "RMSE", "R2", "TRUL", "PRUL"]}
        for rows in rows_all:
            for r in rows:
                for m in vals:
                    vals[m].append(r[m])
        print(f"== {ds} SP{sp}: AE={np.mean(vals['AE']):.2f} "
              f"MAE={np.mean(vals['MAE']):.4f} "
              f"RMSE={np.mean(vals['RMSE']):.4f} "
              f"R2={np.mean(vals['R2']):.4f} "
              f"TRUL={np.mean(vals['TRUL']):.1f} (n={len(rows_all)})", flush=True)


if __name__ == "__main__":
    main()
