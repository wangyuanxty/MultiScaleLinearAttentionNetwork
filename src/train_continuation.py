"""Continuation training diagnostic: continue an existing 100-epoch model
for `--extra` more epochs, then re-evaluate the same full-seq metrics.

Answers: are the ablation results under-converged at 100 epochs? If the
MAE gain from +100 epochs is much larger for multi/xchg than for single,
architecture differences are masked by under-convergence.

Usage (run from src/):
  python train_continuation.py --dataset nasa --config single multi xchg --seed 2 --extra 100

Outputs:
  checkpoints/abl_seed/{ds}/{config}/seed{seed}_cont{extra}.pt   (never overwrites)
  prints: before/after MAE/R2/regen/AE per config
"""
import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, '.')
from gdn_model import build_gdn_model, masked_mae
from make_figures import load_series
from train_per_sp import build_windows
from train_ablation_fullseq import CONFIGS, eval_fullseq

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 64
EPS = 1e-6


def continue_train(seed, state_dict, X, Y, W, cfg, extra):
    model = build_gdn_model(
        multiscale=cfg["multiscale"], stage_query=cfg["stage_query"],
        input_dim=1, window_size=W, output_len=1, readout="last",
    ).to(DEV)
    model.load_state_dict(state_dict)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    N = len(X)
    for ep in range(extra):
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
            print(f"    cont ep{ep} loss={loss.item():.4f}", flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["calce", "nasa"])
    ap.add_argument("--config", nargs="+", required=True, choices=list(CONFIGS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--extra", type=int, default=100)
    ap.add_argument("--base-suffix", default="",
                    help="Suffix appended to seed{seed} for the input ckpt "
                         "(e.g. _cont100 to continue the 200-epoch model)")
    ap.add_argument("--save-suffix", default=None,
                    help="Suffix for output ckpt (default _cont{extra})")
    args = ap.parse_args()
    save_suffix = args.save_suffix or f"_cont{args.extra}"

    caps, train_cells, test_cell, W, sps, eol_ah = load_series(args.dataset)
    caps = {c: caps[c].astype(np.float32) for c in caps}
    all_tr = np.concatenate([caps[c] for c in train_cells])
    lo, hi = float(all_tr.min()), float(all_tr.max())
    X, Y = build_windows(caps, train_cells, lo, hi, W)

    for cfg_name in args.config:
        ckpt_path = f"../checkpoints/abl_seed/{args.dataset}/{cfg_name}/seed{args.seed}{args.base_suffix}.pt"
        save_path = (f"../checkpoints/abl_seed/{args.dataset}/{cfg_name}"
                     f"/seed{args.seed}{save_suffix}.pt")
        if os.path.exists(ckpt_path) and os.path.exists(save_path):
            print(f"[{cfg_name}] SKIP (cont ckpt exists)", flush=True)
            continue
        if not os.path.exists(ckpt_path):
            print(f"[{cfg_name}] ckpt missing: {ckpt_path}", flush=True)
            continue
        ck = torch.load(ckpt_path, map_location=DEV)
        model0 = build_gdn_model(
            multiscale=CONFIGS[cfg_name]['multiscale'],
            stage_query=CONFIGS[cfg_name]['stage_query'],
            input_dim=1, window_size=W, output_len=1, readout='last').to(DEV)
        model0.load_state_dict(ck['state_dict'])
        before = eval_fullseq(model0, caps, test_cell, W, lo, hi, eol_ah)
        print(f"[{cfg_name}] BEFORE: MAE={before['mae']:.4f} R2={before['r2']:.4f} "
              f"regen={before['regen']:.3f} AE={before['ae']:.0f}", flush=True)
        model = continue_train(args.seed, ck['state_dict'], X, Y, W,
                               CONFIGS[cfg_name], args.extra)
        torch.save({"state_dict": model.state_dict(), "seed": args.seed,
                    "lo": lo, "hi": hi, "W": W, "eol_ah": eol_ah,
                    "config": cfg_name, "extra_epochs": args.extra},
                   save_path)
        after = eval_fullseq(model, caps, test_cell, W, lo, hi, eol_ah)
        print(f"[{cfg_name}] AFTER : MAE={after['mae']:.4f} R2={after['r2']:.4f} "
              f"regen={after['regen']:.3f} AE={after['ae']:.0f} "
              f"(delta MAE {after['mae'] - before['mae']:+.4f})", flush=True)


if __name__ == "__main__":
    main()
