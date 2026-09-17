"""Our model on PoliMi-TUB, under Bellomo's protocol.

The decisive experiment.  Bellomo et al. 2024 report, for their autoregressive
model, RUL error at prediction cycles 50/150/300/450 of

    103 -> 81 -> 47 -> 21        (monotone convergence)

Ours freezes on CALCE (2/10 seeds cross a threshold at all).  Every comparison so
far has spanned TWO datasets -- CALCE's 4 LCO cells against their 6 NMC cells --
so "our model is broken" and "CALCE is too small a library" could not be told
apart.  This runs our model on their data, with their target, and reports the same
four numbers.

Their setup, matched here:
  * test cell held out, the other 5 train (one fold, not all six -- asked for)
  * SoH = Qdischarge / 2.5 Ah, EOL at 70%          (their Eq. 1 and stated threshold)
  * target = the SoH value itself, min-max scaled  (their "normalized to the 0-1
    range" -- i.e. ABS_TARGET, not our per-window z-score)
  * RUL error at a prediction cycle j = |predicted EOL - true EOL| in cycles

Two deliberate departures, both stated because they change the comparison:
  * W = 32, not 64.  Their first prediction is at cycle 50, so a 64-cycle window
    does not exist there.  Their own model takes the whole history; ours is
    windowed, and 32 is the largest power of two that fits.
  * the SoH series is INTERPOLATED from ~44 capacity tests per cell (see
    polimi_tub.soh_series) -- the archive has no per-cycle SoH column.

Writes only new paths: checkpoints/per_sp/polimi/ and results/polimi.json.

    cd src && D:/anaconda/envs/py312/python.exe run_polimi.py --test cell1 --seeds 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import train_per_sp as tps                                     # noqa: E402
from gdn_model import build_gdn_model                          # noqa: E402
from polimi_tub import EOL_SOH, QNOM, soh_series               # noqa: E402
from train_per_sp import BATCH, DEV, EPS, build_windows, train_one  # noqa: E402

W = 32
PRED_CYCLES = (50, 150, 300, 450)
AR_CAP = 900          # generously past any true EOL (max series length 840)


def first_below(series, thr):
    """Index (1-based cycle) of the first strictly-below crossing."""
    for i in range(len(series) - 1):
        if series[i] >= thr > series[i + 1]:
            return i + 1
    return -1


def ar_rollout(model, seq, t0, W, steps, abs_target=True):
    """Self-fed.  y = model(x) under an absolute target, y = z*std+mean under z-score."""
    win = [float(v) for v in seq[t0 - W:t0]]
    out = []
    with torch.no_grad():
        for _ in range(steps):
            x = np.asarray(win[-W:], dtype=np.float32)
            wm, ws = float(x.mean()), float(x.std()) + EPS
            z = float(model(torch.tensor(x[None, :, None], device=DEV)))
            y = z if abs_target else z * ws + wm
            out.append(y)
            win.append(y)
    return np.asarray(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="cell1")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--target", choices=["abs", "zscore"], default="abs",
                    help="abs matches Bellomo's 'normalized to the 0-1 range'; "
                         "zscore is our own protocol.  Running both separates "
                         "'the model cannot roll out' from 'the absolute target "
                         "drifts', which is the ambiguity the first run left.")
    args = ap.parse_args()
    abs_on = args.target == "abs"

    S = soh_series()
    test = args.test
    train_cells = [c for c in S if c != test]
    lo = min(float(S[c][1].min()) for c in train_cells)
    hi = max(float(S[c][1].max()) for c in train_cells)

    caps = {c: S[c][1] for c in S}
    print("=" * 96)
    print(f"  PoliMi-TUB  test={test}  train={len(train_cells)} cells  W={W}  "
          f"epochs={args.epochs}  ABS_TARGET")
    print(f"  SoH range from training cells: lo={lo:.4f} hi={hi:.4f}")
    print("=" * 96)

    X, Y = build_windows(caps, train_cells, lo, hi, W)
    spe = max(1, -(-len(X) // BATCH))
    tps.EPOCHS = args.epochs
    tps.PROG_EVERY = max(1, args.epochs // 8)
    tps.ABS_TARGET = abs_on                   # their "normalized to 0-1" target
    print(f"  {len(X)} windows from {len(train_cells)} cells "
          f"({spe} steps/epoch x {args.epochs})\n", flush=True)

    hdr = (f"  {'seed':>4}{'TrueEOL':>9}"
           + "".join(f"{'RULerr@' + str(j):>12}" for j in PRED_CYCLES)
           + f"{'TF MAE':>10}{'TF RMSE':>10}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    out = []
    os.makedirs(os.path.join("..", "checkpoints", "per_sp", "polimi"),
                exist_ok=True)
    cyc, soh = S[test]
    seq = (soh - lo) / (hi - lo + EPS)
    thr_n = (EOL_SOH - lo) / (hi - lo + EPS)
    true_eol = first_below(soh, EOL_SOH)
    for seed in args.seeds:
        t_wall = time.time()
        model = train_one(seed, X, Y, W)
        model.eval()

        # teacher-forced one-step error on the held-out cell
        idx = np.arange(W, len(seq))
        Xw = np.stack([seq[i - W:i] for i in idx]).astype(np.float32)[:, :, None]
        P = []
        with torch.no_grad():
            for k in range(0, len(Xw), 512):
                xb = torch.tensor(Xw[k:k + 512], device=DEV)
                z = model(xb).squeeze(-1)
                if abs_on:
                    P.append(z.cpu().numpy())
                else:
                    wm = xb[:, :, 0].mean(dim=1)
                    ws = xb[:, :, 0].std(dim=1) + EPS
                    P.append((z * ws + wm).cpu().numpy())
        pred = np.concatenate(P)
        mae = float(np.mean(np.abs(pred - seq[idx])))
        rmse = float(np.sqrt(np.mean((pred - seq[idx]) ** 2)))

        errs = []
        for j in PRED_CYCLES:
            if j < W or j >= true_eol:
                errs.append(None)
                continue
            full = ar_rollout(model, seq, j, W, AR_CAP, abs_on)
            k = first_below(full, thr_n)
            pred_eol = (j + k) if k > 0 else (j + AR_CAP)
            errs.append(abs(pred_eol - true_eol))
        print(f"  {seed:>4}{true_eol:>9}"
              + "".join(f"{(e if e is not None else -1):>12}" for e in errs)
              + f"{mae:>10.5f}{rmse:>10.5f}  [{time.time() - t_wall:.0f}s]",
              flush=True)
        out.append(dict(test=test, seed=seed, W=W, epochs=args.epochs,
                        target=args.target,
                        true_eol=true_eol, pred_cycles=list(PRED_CYCLES),
                        rul_err=errs, tf_mae=mae, tf_rmse=rmse))
        torch.save({"state_dict": model.state_dict(), "W": W, "seed": seed,
                    "lo": lo, "hi": hi, "test": test},
                   os.path.join("..", "checkpoints", "per_sp", "polimi",
                                f"{test}_seed{seed}.pt"))
        del model
        torch.cuda.empty_cache()

    p = "results/polimi.json"
    prev = json.load(open(p)) if os.path.exists(p) else []
    json.dump(prev + out, open(p, "w"), indent=2)
    print()
    print("  Bellomo et al. Table 2, their Autoregressive model:")
    print("      RULerr@50 = 103   @150 = 81   @300 = 47   @450 = 21"
          "   (monotone convergence)")
    print("  Ours freeze on CALCE.  If the four numbers above converge the same")
    print("  way, our model is fine and CALCE was the problem; if they do not,")
    print("  the fault is in our model or pipeline, not in the dataset size.")


if __name__ == "__main__":
    main()
