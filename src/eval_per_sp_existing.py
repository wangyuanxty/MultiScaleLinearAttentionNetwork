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
from eval_multiseed import true_rul

EPS = 1e-6
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_sp_batched(model, caps, test_cells, lo, hi, W, sp, eol_ah):
    """Batch-forwards of train_per_sp.eval_sp (identical math:
    per-window z-score de-normalization, same crossing rule)."""
    model.eval()
    th = (eol_ah - lo) / (hi - lo + EPS)
    rows = []
    with torch.no_grad():
        for tc in test_cells:
            seq = (caps[tc] - lo) / (hi - lo + EPS)
            idx = np.arange(sp, len(seq))
            X = np.stack([seq[i - W:i, None] for i in idx]).astype(np.float32)
            wmean = X[:, :, 0].mean(axis=1)
            wstd = X[:, :, 0].std(axis=1) + EPS
            pred = model(torch.tensor(X, device=DEV)).squeeze(-1).cpu().numpy()
            seg_p = pred * wstd + wmean
            tv = seq[sp:]
            n = min(len(tv), len(seg_p))
            tv, seg_p = tv[:n], seg_p[:n]
            trul = true_rul(tv, th)
            prul = true_rul(seg_p, th)
            rows.append({
                "TRUL": trul, "PRUL": prul, "AE": abs(trul - prul),
                "MAE": float(np.mean(np.abs(tv - seg_p))),
                "RMSE": float(np.sqrt(np.mean((tv - seg_p) ** 2))),
                "R2": float(1 - np.sum((tv - seg_p) ** 2) /
                            (np.sum((tv - tv.mean()) ** 2) + EPS)),
            })
    return rows


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
    if ds == "mit":
        from load_datasets import MIT_TEST_CELLS
        test_cells = MIT_TEST_CELLS

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
            rows = eval_sp_batched(model, caps, test_cells, lo, hi, W, sp,
                                   eol_ah)
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
        mae_s, rmse_s, r2_s = (float(np.std(vals[m])) for m in
                               ["MAE", "RMSE", "R2"])
        print(f"== {ds} SP{sp}: AE={np.mean(vals['AE']):.2f} "
              f"MAE={np.mean(vals['MAE']):.4f}±{mae_s:.4f} "
              f"RMSE={np.mean(vals['RMSE']):.4f}±{rmse_s:.4f} "
              f"R2={np.mean(vals['R2']):.4f}±{r2_s:.4f} "
              f"TRUL={np.mean(vals['TRUL']):.1f} (n={len(rows_all)})", flush=True)


if __name__ == "__main__":
    main()
