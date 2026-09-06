"""Continue per-SP checkpoints from 100 to 200 epochs (CALCE / NASA).

Loads the existing 100-epoch per-SP checkpoint (train_per_sp.py
pipeline), trains 100 more epochs with the same protocol (fresh Adam
lr=1e-3, batch 64, per-window z-score targets, same train split only
-- no evaluation leakage: train windows use targets < SP, eval starts
at >= SP), and saves the 200-epoch model separately as
`SP{sp}_seed{seed}_ep200.pt`; the 100-epoch files are kept untouched
for rollback.  Prints BEFORE/AFTER MAE/RMSE/R2/AE per model.

Run:  python src/continue_per_sp_200.py --dataset calce
      python src/continue_per_sp_200.py --dataset nasa
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

import train_per_sp as T
from gdn_model import build_gdn_model
from make_figures import load_series

DEV = T.DEV


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--start-seed", type=int, default=1)
    ap.add_argument("--sps", type=int, nargs="+", default=None)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--from-suffix", default="",
                    help="base checkpoint suffix (e.g. _ep200)")
    ap.add_argument("--save-suffix", default="_ep200",
                    help="output checkpoint suffix (e.g. _ep300)")
    ap.add_argument("--extra", type=int, default=100,
                    help="epochs to continue")
    ap.add_argument("--final-epochs", type=int, default=200,
                    help="epoch total recorded in ckpt metadata")
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
        Xtr, Ytr = T.build_windows(caps, train_cells, lo, hi, W)
        Xts, Yts = T.build_windows(caps, test_cells, lo, hi, W, max_cycle=sp)
        X = np.vstack([Xtr, Xts])
        Y = np.concatenate([Ytr, Yts])
        for seed in range(args.start_seed, args.start_seed + args.seeds):
            base = f"{ckpt_dir}/SP{sp}_seed{seed}{args.from_suffix}.pt"
            save = f"{ckpt_dir}/SP{sp}_seed{seed}{args.save_suffix}.pt"
            if not os.path.exists(base):
                print(f"SP{sp} seed{seed}: NO base ckpt", flush=True)
                continue
            if args.skip_existing and os.path.exists(save):
                print(f"SP{sp} seed{seed}: SKIP (_ep200 exists)", flush=True)
                continue

            ck = torch.load(base, map_location=DEV, weights_only=False)
            model = build_gdn_model(
                multiscale=True, stage_query=True, input_dim=1,
                window_size=W, output_len=1, readout="last").to(DEV)
            model.load_state_dict(ck["state_dict"])
            before = T.eval_sp(model, caps, test_cells, lo, hi, W, sp,
                               eol_ah)
            b = {m: float(np.mean([r[m] for r in before]))
                 for m in ("MAE", "RMSE", "R2")}
            b["AE"] = float(np.mean([r["AE"] for r in before]))

            # continuation: fresh optimizer, same training loop as train_one
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            model.train()
            N = len(X)
            for _ in range(args.extra):
                perm = np.random.permutation(N)
                for s in range(0, N, T.BATCH):
                    idx = perm[s:s + T.BATCH]
                    x = torch.tensor(X[idx]).to(DEV)
                    y = torch.tensor(Y[idx]).to(DEV)
                    opt.zero_grad()
                    pred = model(x).squeeze(-1)
                    wmean = x[:, :, 0].mean(dim=1)
                    wstd = x[:, :, 0].std(dim=1) + T.EPS
                    tgt = (y - wmean) / wstd
                    loss = T.masked_mae(pred, tgt, torch.ones_like(y))
                    loss.backward()
                    opt.step()

            model.eval()
            after = T.eval_sp(model, caps, test_cells, lo, hi, W, sp,
                              eol_ah)
            a = {m: float(np.mean([r[m] for r in after]))
                 for m in ("MAE", "RMSE", "R2")}
            a["AE"] = float(np.mean([r["AE"] for r in after]))
            torch.save({"state_dict": model.state_dict(), "seed": seed,
                        "sp": sp, "epochs": args.final_epochs, "lo": lo,
                        "hi": hi, "W": W, "eol_ah": eol_ah}, save)
            print(f"SP{sp} seed{seed}: BEFORE MAE={b['MAE']:.4f} R2={b['R2']:.4f} "
                  f"AE={b['AE']:.0f} -> AFTER MAE={a['MAE']:.4f} "
                  f"R2={a['R2']:.4f} AE={a['AE']:.0f}", flush=True)


if __name__ == "__main__":
    main()
