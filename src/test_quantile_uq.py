"""UQ (trend C): Pinball quantile regression + Conformal calibration (CQR).

Main model (multiscale + StageQuery + direct head, z-score protocol)
trained with PinballLoss(quantiles=(0.025, 0.5, 0.975)) on CS2_37/38.
Calibration cell CS2_36 (held out): conformity scores
  s_i = max(q_lo - y_i, y_i - q_hi)
  q_adj = (1-alpha)(1+1/n) empirical quantile of s  (alpha = 0.05)
Test cell CS2_35: raw vs CQR-adjusted coverage, interval widths, P50 MAE.

Model checkpoint saved to ../checkpoints/quantile_calce_seed42.pt.
Results saved to results/quantile_uq.json.
"""
import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, '.')
from gdn_model import build_gdn_model, PinballLoss
from make_figures import load_series

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH, EPOCHS, SEED = 64, 100, 42
EPS = 1e-6
QUANTILES = (0.025, 0.5, 0.975)
ALPHA = 0.05
TRAIN_CELLS = ["CS2_37", "CS2_38"]
CAL_CELL = "CS2_36"
TEST_CELL = "CS2_35"


def build_windows(caps, cells, lo, hi, W):
    X, Y = [], []
    for c in cells:
        seq = (caps[c] - lo) / (hi - lo + EPS)
        for i in range(W, len(seq)):
            X.append(seq[i - W:i, None])
            Y.append(seq[i])
    return np.stack(X).astype(np.float32), np.array(Y, dtype=np.float32)


def predict_seq(model, caps, cell, lo, hi, W):
    """De-normalized (lo_q, mid_q, hi_q) per window for one cell."""
    model.eval()
    tc = (caps[cell] - lo) / (hi - lo + EPS)
    qs = []
    with torch.no_grad():
        for i in range(W, len(tc)):
            win = tc[i - W:i, None]
            cin = torch.tensor(win, dtype=torch.float32).unsqueeze(0).to(DEV)
            wmean = float(win[:, 0].mean())
            wstd = float(win[:, 0].std()) + EPS
            qz = model(cin).cpu().numpy().squeeze()  # (3,)
            qs.append(qz * wstd + wmean)
    return np.array(qs), tc[W:]


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    caps, _, _, W, sps, eol_ah = load_series("calce")
    caps = {c: caps[c].astype(np.float32) for c in caps}

    all_tr = np.concatenate([caps[c] for c in TRAIN_CELLS])
    lo, hi = float(all_tr.min()), float(all_tr.max())

    model = build_gdn_model(
        multiscale=True, stage_query=True, input_dim=1, window_size=W,
        output_len=1, num_quantiles=3, readout="last",
    ).to(DEV)
    loss_fn = PinballLoss(quantiles=QUANTILES).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    X, Y = build_windows(caps, TRAIN_CELLS, lo, hi, W)
    N = len(X)
    print(f"training quantile-UQ on {TRAIN_CELLS} (N={N}) ...", flush=True)
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(N)
        for s in range(0, N, BATCH):
            idx = perm[s:s + BATCH]
            x = torch.tensor(X[idx]).to(DEV)
            y = torch.tensor(Y[idx]).to(DEV)
            opt.zero_grad()
            pred = model(x)                        # (B, 3, 1)
            wmean = x[:, :, 0].mean(dim=1)
            wstd = x[:, :, 0].std(dim=1) + EPS
            tgt_norm = (y - wmean) / wstd
            loss = loss_fn(pred, tgt_norm.unsqueeze(-1))
            loss.backward()
            opt.step()
        if ep % 25 == 0:
            print(f"    ep{ep} loss={loss.item():.4f}", flush=True)

    # checkpoint
    ckpt_path = "../checkpoints/quantile_calce_seed42.pt"
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "config": {"multiscale": True, "stage_query": True,
                           "input_dim": 1, "window_size": W,
                           "output_len": 1, "num_quantiles": 3,
                           "readout": "last",
                           "quantiles": list(QUANTILES)},
                "train_cells": TRAIN_CELLS, "cal_cell": CAL_CELL,
                "test_cell": TEST_CELL, "lo": lo, "hi": hi,
                "eol_ah": float(eol_ah), "sps": list(sps)},
               ckpt_path)
    print(f"saved {ckpt_path}", flush=True)

    # ---- calibration on held-out cell ----
    q_cal, y_cal = predict_seq(model, caps, CAL_CELL, lo, hi, W)
    lo_q, mid_q, hi_q = q_cal[:, 0], q_cal[:, 1], q_cal[:, 2]
    scores = np.maximum(lo_q - y_cal, y_cal - hi_q)
    n_cal = len(scores)
    q_level = (1 - ALPHA) * (1 + 1.0 / n_cal)
    q_adj = float(np.quantile(scores, min(q_level, 1.0)))
    print(f"  calibration ({CAL_CELL}, n={n_cal}): q_adj={q_adj:.4f} "
          f"(scores mean={scores.mean():.4f})", flush=True)

    # ---- test with raw and CQR intervals ----
    q_test, y_test = predict_seq(model, caps, TEST_CELL, lo, hi, W)
    lo_t, mid_t, hi_t = q_test[:, 0], q_test[:, 1], q_test[:, 2]

    mae50 = float(np.mean(np.abs(mid_t - y_test)))
    raw_in = (y_test >= lo_t) & (y_test <= hi_t)
    cqr_lo, cqr_hi = lo_t - q_adj, hi_t + q_adj
    cqr_in = (y_test >= cqr_lo) & (y_test <= cqr_hi)
    print(f"  raw: P50 MAE={mae50:.4f} coverage={raw_in.mean():.3f} "
          f"width={np.mean(hi_t - lo_t):.4f}", flush=True)
    print(f"  CQR: coverage={cqr_in.mean():.3f} "
          f"width={np.mean(cqr_hi - cqr_lo):.4f}", flush=True)

    per_sp = []
    for sp in sps:
        s = max(sp - W, 0)
        per_sp.append({
            "SP": sp,
            "raw_cov": round(float(raw_in[s:].mean()), 3),
            "cqr_cov": round(float(cqr_in[s:].mean()), 3),
            "cqr_width": round(float(np.mean(cqr_hi[s:] - cqr_lo[s:])), 4),
        })
        print(f"  SP{sp}: raw={raw_in[s:].mean():.3f} "
              f"CQR={cqr_in[s:].mean():.3f} "
              f"width={np.mean(cqr_hi[s:] - cqr_lo[s:]):.4f}", flush=True)

    out = {"seed": SEED, "alpha": ALPHA, "quantiles": list(QUANTILES),
           "train_cells": TRAIN_CELLS, "cal_cell": CAL_CELL,
           "test_cell": TEST_CELL, "q_adj": round(q_adj, 4),
           "P50_MAE": round(mae50, 4),
           "raw_coverage": round(float(raw_in.mean()), 3),
           "cqr_coverage": round(float(cqr_in.mean()), 3),
           "raw_width": round(float(np.mean(hi_t - lo_t)), 4),
           "cqr_width": round(float(np.mean(cqr_hi - cqr_lo)), 4),
           "per_sp": per_sp}
    os.makedirs("results", exist_ok=True)
    with open("results/quantile_uq.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved results/quantile_uq.json", flush=True)


if __name__ == "__main__":
    main()
