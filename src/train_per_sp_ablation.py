"""Per-SP ablation pilot — same protocol as the main experiments (train_per_sp.py).

Pilot (user decision 2026-08-30): 3 configs (single / multi / xchg) x
CALCE SP 300/400/500 x seeds 1..3. Protocol == train_per_sp.py:
fresh model per (SP, seed), train = non-test cells' full sequences +
test cell's cycles BEFORE the SP (PatchFormer convention), per-window
z-score target, global min--max from train split; eval on the segment
starting at SP (per-SP MAE/RMSE/R2/AE rows).

Outputs NEVER touch the full-sequence ablation files:
  checkpoints/per_sp_abl/calce/{config}/SP{sp}_seed{seed}.pt
  results/per_sp_ablation_calce_{config}.json
"""
import argparse
import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, '.')
from gdn_model import build_gdn_model, masked_mae
from make_figures import load_series
from eval_multiseed import true_rul
from train_per_sp import build_windows, eval_sp

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH, EPOCHS = 64, 100
EPS = 1e-6

CONFIGS = {
    "single": {"multiscale": False, "stage_query": False},
    "multi": {"multiscale": True, "stage_query": False},
    "xchg": {"multiscale": True, "stage_query": True},
}


def train_one(seed, X, Y, W, cfg):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_gdn_model(
        multiscale=cfg["multiscale"], stage_query=cfg["stage_query"],
        input_dim=1, window_size=W, output_len=1, readout="last",
    ).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    N = len(X)
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(N)
        for s in range(0, N, BATCH):
            idx = perm[s:s + BATCH]
            x = torch.tensor(X[idx]).to(DEV)
            y = torch.tensor(Y[idx]).to(DEV)
            opt.zero_grad()
            pred = model(x).squeeze(-1)
            wmean = x[:, :, 0].mean(dim=1)
            wstd = x[:, :, 0].std(dim=1) + EPS
            tgt = (y - wmean) / wstd
            loss = masked_mae(pred, tgt, torch.ones_like(y))
            loss.backward()
            opt.step()
        if ep % 25 == 0:
            print(f"  ep{ep} loss={loss.item():.4f}", flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", nargs="+", required=True, choices=list(CONFIGS))
    ap.add_argument("--seeds", type=int, default=3)
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

    for cfg_name in args.config:
        cfg = CONFIGS[cfg_name]
        ckpt_dir = f"../checkpoints/per_sp_abl/calce/{cfg_name}"
        os.makedirs(ckpt_dir, exist_ok=True)
        out_path = f"results/per_sp_ablation_calce_{cfg_name}.json"
        out = json.load(open(out_path)) if os.path.exists(out_path) else {
            "dataset": ds, "config": cfg_name, "protocol": "per-SP, pilot seeds 1-3",
        }
        for sp in sps:
            Xtr, Ytr = build_windows(caps, train_cells, lo, hi, W)
            Xts, Yts = build_windows(caps, test_cells, lo, hi, W, max_cycle=sp)
            X_all = np.vstack([Xtr, Xts])
            Y_all = np.concatenate([Ytr, Yts])
            for seed in range(args.start_seed, args.start_seed + args.seeds):
                skey = str(seed)
                if skey in out.setdefault(str(sp), {}):
                    print(f"[{cfg_name}] {ds} SP{sp} seed{seed}: SKIP", flush=True)
                    continue
                model = train_one(seed, X_all, Y_all, W, cfg)
                torch.save(
                    {"state_dict": model.state_dict(), "seed": seed,
                     "lo": lo, "hi": hi, "W": W, "sp": sp, "eol_ah": eol_ah,
                     "test_cells": test_cells, "train_cells": train_cells},
                    f"{ckpt_dir}/SP{sp}_seed{seed}.pt")
                rows = eval_sp(model, caps, test_cells, lo, hi, W, sp, eol_ah)
                out.setdefault(str(sp), {})[skey] = rows
                print(f"[{cfg_name}] {ds} SP{sp} seed{seed}: "
                      f"AE={[r['AE'] for r in rows]} "
                      f"MAE={np.mean([r['MAE'] for r in rows]):.4f} "
                      f"R2={np.mean([r['R2'] for r in rows]):.4f}",
                      flush=True)
                json.dump(out, open(out_path, "w"), indent=2)


if __name__ == "__main__":
    main()
