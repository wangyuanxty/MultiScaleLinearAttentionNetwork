"""PatchFormer-consistent per-SP protocol training (10 seeds per SP).

Protocol (user decision 2026-08-17, STRICT PatchFormer consistency):

  For every dataset, for EVERY (SP, seed) pair:
    - train a fresh model (10 seeds per SP: 1..10)
    - train = non-test cells' FULL sequences
             PLUS the test cell's cycles BEFORE the SP (Cycle < SP)
             --- exactly the NASADataPreProcess.py convention, applied
                uniformly to all five datasets per user instruction ---
    - per-window z-score target; global min-max input from train split
    - evaluate the segment starting at SP (absolute index), with
      per-window de-normalization; TRUL/PRUL/AE/MAE/RMSE/R2

Outputs:
  checkpoints/per_sp/{ds}/SP{sp}_seed{seed}.pt
  results/per_sp_train.json  (per dataset/SP/seed metrics + 10-seed means)
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


def build_windows(caps, cells, lo, hi, W, max_cycle=None):
    """Windows from the given cells' sequences (or up to max_cycle).

    max_cycle = SP: includes ONLY cycles < SP (PatchFormer convention).
    """
    X, Y = [], []
    for c in cells:
        seq = (caps[c] - lo) / (hi - lo + EPS)
        end = len(seq) if max_cycle is None else min(len(seq), max_cycle)
        for i in range(W, end):
            X.append(seq[i - W:i, None])
            Y.append(seq[i])
    return np.stack(X).astype(np.float32), np.array(Y, dtype=np.float32)


def train_one(seed, X, Y, W):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_gdn_model(
        multiscale=True, stage_query=True, input_dim=1, window_size=W,
        output_len=1, readout="last",
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
    return model


def eval_sp(model, caps, test_cells, lo, hi, W, sp, eol_ah):
    model.eval()
    th = (eol_ah - lo) / (hi - lo + EPS)
    rows = []
    for tc in test_cells:
        seq = (caps[tc] - lo) / (hi - lo + EPS)
        seg_p = []
        with torch.no_grad():
            for i in range(sp, len(seq)):
                win = seq[i - W:i, None]
                cin = torch.tensor(win, dtype=torch.float32).unsqueeze(0).to(DEV)
                wmean = float(win[:, 0].mean())
                wstd = float(win[:, 0].std()) + EPS
                seg_p.append(model(cin).item() * wstd + wmean)
        seg_p = np.array(seg_p)
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
    ap.add_argument("--sps", type=int, nargs="+", default=None,
                    help="Override SP list; default uses load_series")
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

    os.makedirs(f"../checkpoints/per_sp/{ds}", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    out_path = "results/per_sp_train.json"
    out = {}
    if os.path.exists(out_path):
        out = json.load(open(out_path))

    # normalization lo/hi from the train split (non-test cells only)
    all_tr = np.concatenate([caps[c] for c in train_cells])
    lo, hi = float(all_tr.min()), float(all_tr.max())

    for sp in sps:
        # train set: non-test cells full + test cell cycles < SP
        Xtr, Ytr = build_windows(caps, train_cells, lo, hi, W)
        Xts, Yts = build_windows(caps, test_cells, lo, hi, W, max_cycle=sp)
        X_all = np.vstack([Xtr, Xts])
        Y_all = np.concatenate([Ytr, Yts])
        n_tr = len(Xtr)

        for seed in range(args.start_seed, args.start_seed + args.seeds):
            skey = str(seed)
            if skey in out.setdefault(ds, {}).get(str(sp), {}):
                print(f"{ds} SP{sp} seed{seed}: SKIP (already in JSON)", flush=True)
                continue
            model = train_one(seed, X_all, Y_all, W)
            torch.save(
                {"state_dict": model.state_dict(), "seed": seed,
                 "lo": lo, "hi": hi, "W": W, "sp": sp, "eol_ah": eol_ah,
                 "test_cells": test_cells, "train_cells": train_cells},
                f"../checkpoints/per_sp/{ds}/SP{sp}_seed{seed}.pt")
            rows = eval_sp(model, caps, test_cells, lo, hi, W, sp, eol_ah)
            out.setdefault(ds, {}).setdefault(str(sp), {})[str(seed)] = rows
            print(f"{ds} SP{sp} seed{seed}: "
                  f"AE={[r['AE'] for r in rows]} "
                  f"MAE={np.mean([r['MAE'] for r in rows]):.4f} "
                  f"R2={np.mean([r['R2'] for r in rows]):.4f} "
                  f"trainN={n_tr}+{len(Xts)}", flush=True)
            json.dump(out, open(out_path, "w"), indent=2)

    # 10-seed per-SP means
    print("=== per-SP means over seeds ===", flush=True)
    for sp in sps:
        vals = {m: [] for m in ["AE", "MAE", "RMSE", "R2", "TRUL", "PRUL"]}
        for seed in range(args.start_seed, args.start_seed + args.seeds):
            for r in out[ds][str(sp)].get(str(seed), []):
                for m in vals:
                    vals[m].append(r[m])
        print(f"SP{sp}: AE={np.mean(vals['AE']):.2f} "
              f"MAE={np.mean(vals['MAE']):.4f} "
              f"RMSE={np.mean(vals['RMSE']):.4f} "
              f"R2={np.mean(vals['R2']):.4f}", flush=True)


if __name__ == "__main__":
    main()
