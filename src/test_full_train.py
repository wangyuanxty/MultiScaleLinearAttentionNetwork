"""Full-dataset training: MAIN model (agreed architecture).

  - multiscale: 3 branches (patch 2/4/8)
  - interaction: StageQuery V3 (coarse-branch GDN state as query)
  - head: direct (no physics — physics is the ablation study, done separately)
  - normalization: PatchFormer-identical — input global min-max
    (train lo/hi), target per-window z-score, eval with per-window
    de-normalization

Capacity-only input. 5 datasets: calce, nasa, mit, panasonic, tju.
Usage: python test_full_train.py --seed 42 [--datasets calce,nasa]
Results appended to results/full_train.json.
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

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH, EPOCHS = 64, 100
EPS = 1e-6


def build_windows(caps, cells, lo, hi, W):
    X, Y = [], []
    for c in cells:
        seq = (caps[c] - lo) / (hi - lo + EPS)
        for i in range(W, len(seq)):
            X.append(seq[i - W:i, None])
            Y.append(seq[i])
    return np.stack(X).astype(np.float32), np.array(Y, dtype=np.float32)


def train_eval(ds, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    caps, train_cells, test_cell, W, sps, eol_ah = load_series(ds)
    caps = {c: caps[c].astype(np.float32) for c in caps}

    all_tr = np.concatenate([caps[c] for c in train_cells])
    lo, hi = float(all_tr.min()), float(all_tr.max())

    model = build_gdn_model(
        multiscale=True, stage_query=True, input_dim=1, window_size=W,
        output_len=1, readout="last",
    ).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    X, Y = build_windows(caps, train_cells, lo, hi, W)
    N = len(X)
    tag = f"{ds} seed{seed}"
    print(f"training {tag} (N={N}, W={W}) ...", flush=True)
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
            tgt_norm = (y - wmean) / wstd          # per-window z-score
            loss = masked_mae(pred, tgt_norm, torch.ones_like(y))
            loss.backward()
            opt.step()
        if ep % 25 == 0:
            print(f"    ep{ep} loss={loss.item():.4f}", flush=True)

    model.eval()
    tc = (caps[test_cell] - lo) / (hi - lo + EPS)
    seg_p = []
    with torch.no_grad():
        for i in range(W, len(tc)):
            win = tc[i - W:i, None]
            cin = torch.tensor(win, dtype=torch.float32).unsqueeze(0).to(DEV)
            wmean = float(win[:, 0].mean())
            wstd = float(win[:, 0].std()) + EPS
            seg_p.append(model(cin).item() * wstd + wmean)  # de-norm
    seg_p = np.array(seg_p)
    tv = tc[W:]
    mae = np.mean(np.abs(seg_p - tv))
    r2 = 1 - np.sum((tv - seg_p) ** 2) / (np.sum((tv - tv.mean()) ** 2) + EPS)
    regen = np.mean(np.diff(seg_p) > 0.002)
    th = (eol_ah - lo) / (hi - lo + EPS)
    ae = abs(true_rul(tv, th) - true_rul(seg_p, th))
    print(f"  [{tag}] MAE={mae:.4f} R2={r2:.4f} regen={regen:.3f} AE={ae}",
          flush=True)

    # checkpoint: state + full config so eval scripts can reload without
    # re-deriving normalization/protocol constants
    ckpt_path = f"../checkpoints/full_{ds}_seed{seed}.pt"
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "config": {"multiscale": True, "stage_query": True, "input_dim": 1,
                   "window_size": W, "output_len": 1, "num_quantiles": 1,
                   "readout": "last"},
        "train_cells": list(train_cells), "test_cell": test_cell,
        "lo": float(lo), "hi": float(hi), "eol_ah": float(eol_ah),
        "sps": list(sps),
    }, ckpt_path)
    print(f"  saved {ckpt_path}", flush=True)

    return {"dataset": ds, "seed": seed, "W": W,
            "MAE": round(float(mae), 4), "R2": round(float(r2), 4),
            "AE": int(ae), "regen": round(float(regen), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--datasets", default="calce,nasa,mit,panasonic,tju")
    args = ap.parse_args()
    ds_list = [d.strip() for d in args.datasets.split(",")]

    out_path = "results/full_train.json"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    try:
        with open(out_path) as f:
            results = json.load(f)
    except Exception:
        results = []

    for ds in ds_list:
        try:
            r = train_eval(ds, args.seed)
            # dedup by (dataset, seed) so re-runs replace old entries
            results = [x for x in results
                       if not (x.get("dataset") == ds
                               and x.get("seed") == args.seed)]
            results.append(r)
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            print(f"  {ds} FAILED: {e}", flush=True)


if __name__ == "__main__":
    main()
