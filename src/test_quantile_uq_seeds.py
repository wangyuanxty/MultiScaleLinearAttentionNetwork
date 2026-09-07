"""Multi-seed UQ (pinball + CQR) follow-up — same pipeline as
test_quantile_uq.py, over seeds 1..10.

Trains 10 independent quantile-head models on CS2_37/38, calibrates
each on CS2_36 (conformity scores, (1-alpha)(1+1/n) quantile), and
evaluates raw vs CQR-adjusted coverage/width on CS2_35. Aggregates
mean +/- std across seeds into results/quantile_uq_10seeds.json;
per-seed ckpts saved as checkpoints/quantile_calce_seed{N}.pt.

Run:  python src/test_quantile_uq_seeds.py
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gdn_model import build_gdn_model, PinballLoss
from make_figures import load_series
from test_quantile_uq import (
    BATCH, EPOCHS, EPS, QUANTILES, ALPHA, TRAIN_CELLS, CAL_CELL, TEST_CELL,
    DEV, build_windows, predict_seq)

CKPT = "checkpoints"


def train_seed(seed, caps, lo, hi, W):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_gdn_model(
        multiscale=True, stage_query=True, input_dim=1, window_size=W,
        output_len=1, num_quantiles=3, readout="last").to(DEV)
    loss_fn = PinballLoss(quantiles=QUANTILES).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    X, Y = build_windows(caps, TRAIN_CELLS, lo, hi, W)
    N = len(X)
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(N)
        for s in range(0, N, BATCH):
            idx = perm[s:s + BATCH]
            x = torch.tensor(X[idx]).to(DEV)
            y = torch.tensor(Y[idx]).to(DEV)
            opt.zero_grad()
            pred = model(x)
            wmean = x[:, :, 0].mean(dim=1)
            wstd = x[:, :, 0].std(dim=1) + EPS
            loss = loss_fn(pred, ((y - wmean) / wstd).unsqueeze(-1))
            loss.backward()
            opt.step()
    return model


def main():
    caps, _, _, W, sps, eol_ah = load_series("calce")
    caps = {c: caps[c].astype(np.float32) for c in caps}
    all_tr = np.concatenate([caps[c] for c in TRAIN_CELLS])
    lo, hi = float(all_tr.min()), float(all_tr.max())

    rows = []
    for seed in range(1, 11):
        model = train_seed(seed, caps, lo, hi, W)
        q_cal, y_cal = predict_seq(model, caps, CAL_CELL, lo, hi, W)
        scores = np.maximum(q_cal[:, 0] - y_cal, y_cal - q_cal[:, 2])
        n_cal = len(scores)
        q_adj = float(np.quantile(scores, min((1 - ALPHA) * (1 + 1.0 / n_cal),
                                              1.0)))

        q_test, y_test = predict_seq(model, caps, TEST_CELL, lo, hi, W)
        mae50 = float(np.mean(np.abs(q_test[:, 1] - y_test)))
        raw_in = (y_test >= q_test[:, 0]) & (y_test <= q_test[:, 2])
        cqr_lo, cqr_hi = q_test[:, 0] - q_adj, q_test[:, 2] + q_adj
        cqr_in = (y_test >= cqr_lo) & (y_test <= cqr_hi)
        print(f"[seed {seed}] raw_cov={raw_in.mean():.3f} "
              f"cqr_cov={cqr_in.mean():.3f} "
              f"width={np.mean(cqr_hi - cqr_lo):.4f} "
              f"P50MAE={mae50:.4f} q_adj={q_adj:.4f}", flush=True)

        torch.save(model.state_dict(),
                   os.path.join(CKPT, f"quantile_calce_seed{seed}.pt"))
        rows.append({
            "seed": seed, "q_adj": round(q_adj, 4),
            "P50_MAE": round(mae50, 4),
            "raw_coverage": round(float(raw_in.mean()), 3),
            "cqr_coverage": round(float(cqr_in.mean()), 3),
            "cqr_width": round(float(np.mean(cqr_hi - cqr_lo)), 4),
        })

    agg = {}
    for k in ("P50_MAE", "raw_coverage", "cqr_coverage", "cqr_width"):
        vals = np.array([r[k] for r in rows])
        agg[k] = {"mean": round(float(vals.mean()), 4),
                  "std": round(float(vals.std()), 4)}
    out = {"alpha": ALPHA, "quantiles": list(QUANTILES),
           "train_cells": TRAIN_CELLS, "cal_cell": CAL_CELL,
           "test_cell": TEST_CELL, "seeds": rows, "agg": agg}
    os.makedirs("results", exist_ok=True)
    with open("results/quantile_uq_10seeds.json", "w") as f:
        json.dump(out, f, indent=2)
    print("agg:", json.dumps(agg), flush=True)


if __name__ == "__main__":
    main()
