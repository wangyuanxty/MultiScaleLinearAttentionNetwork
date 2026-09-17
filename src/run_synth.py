"""Learning curve on fully synthetic cells: does the AR freeze depend on how many
independent degradation trajectories the model trains on?

The experiment the whole synthetic detour exists for.  Everything except the
number of training trajectories is held fixed:

  * one pool, generated once (synth_battery.make_pool), test cells carved out
    FIRST and never trained on at any N, for any seed
  * training sets are NESTED prefixes train[:N] -- N=1 subset of N=3 subset of
    ... -- so going up in N only ADDS cells.  Independent draws would change the
    data and the count together, which is the confound that voided the NASA
    augmentation run.
  * lo/hi come from all 115 training cells, a constant of the pool.  Taken from
    the arm's own prefix they would drift with N, changing the normalisation
    scale at the same time as the cell count.
  * COMPUTE IS MATCHED BY GRADIENT STEPS, not by epochs.  train_one runs
    ceil(n_windows/BATCH) steps per epoch, so 100 epochs on 100 cells is ~125x
    the optimisation of 100 epochs on 1 cell.  EPOCHS is patched per arm so the
    total step count is the same everywhere; the epoch count is reported.

Metrics follow the locked convention: MSE and AE, never a crossing rate (one bit,
and it hides bias).  A rollout that never crosses contributes AE = last cycle
reached - EOL, flagged ">=", rather than being dropped.

Writes only new paths: checkpoints/per_sp/synth/ and results/synth_curve.json.
Run from src/ -- these are the same relative paths train_per_sp.py uses.

    cd src && D:/anaconda/envs/py312/python.exe run_synth.py --ns 1 100 --seeds 1
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

import train_per_sp as tps                                  # noqa: E402
from gdn_model import build_gdn_model                       # noqa: E402
from synth_battery import first_crossing, make_pool         # noqa: E402
from train_per_sp import BATCH, DEV, EPS, build_windows, train_one  # noqa: E402

W = 64
LAUNCH_BEFORE = 50
AR_STEPS = 150          # rollout length, FIXED -- see eval_cell


def ar_rollout(model, seq, t0, W, steps):
    """Self-fed: after t0 nothing real is fed back in."""
    win = [float(v) for v in np.asarray(seq[t0 - W:t0], dtype=np.float64)]
    out = []
    with torch.no_grad():
        for _ in range(steps):
            x = np.asarray(win[-W:], dtype=np.float32)
            wm, ws = float(x.mean()), float(x.std()) + EPS
            z = float(model(torch.tensor(x[None, :, None], device=DEV)))
            y = z * ws + wm
            out.append(y)
            win.append(y)
    return np.asarray(out)


def eval_cell(model, seq, eol_cyc, thr, lo, hi, launch=LAUNCH_BEFORE,
              steps=AR_STEPS):
    """(TF MAE/RMSE/R2, AR MSE@launch, AR MSE, AR AE) for one test cell.

    The rollout length is FIXED, not "until the series ends".  A cell that
    crosses 80% while fading ~1%/cycle has only ~20 cycles of capacity left, so
    the series tail is short by physics (MIT's real batch2_cell5: 18 cycles) --
    and if the rollout stops where the DATA stops, a rollout that never crossed
    can only be bounded by that tail, which says almost nothing.  The rollout is
    self-fed and never needed the truth to continue, so it runs a fixed number of
    steps instead:

      * the CROSSING test uses all `steps`, so AE is comparable across cells of
        any tail length.  A rollout still not crossing at the cap is reported as
        exactly `steps` (a bound, and a uniform one).
      * MSE uses only the steps that overlap real values, and `n_true` says how
        many that was.  MSE@launch (launch -> EOL) always has its full window.

    TF is batched -- the same window-at-a-time computation eval_sp does, just
    stacked; on 5 cells x several hundred windows the python loop costs more than
    the forward passes.
    """
    s = (np.asarray(seq, dtype=np.float64) - lo) / (hi - lo + EPS)
    t = (thr - lo) / (hi - lo + EPS)

    idx = np.arange(W, len(s))
    Xw = np.stack([s[i - W:i] for i in idx]).astype(np.float32)[:, :, None]
    pred = []
    with torch.no_grad():
        for k in range(0, len(Xw), 512):
            xb = torch.tensor(Xw[k:k + 512], device=DEV)
            wm = xb[:, :, 0].mean(dim=1, keepdim=True)
            ws = xb[:, :, 0].std(dim=1, keepdim=True) + EPS
            pred.append((model(xb) * ws + wm).squeeze(-1).cpu().numpy())
    p = np.concatenate(pred)
    truth = s[idx]
    err = p - truth
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    r2 = float(1.0 - np.sum(err ** 2) / np.sum((truth - truth.mean()) ** 2))

    t0 = eol_cyc - launch
    full = ar_rollout(model, s, t0, W, steps)
    n_true = max(0, min(steps, len(s) - t0))
    tr = s[t0:t0 + n_true]
    n50 = min(launch, n_true)
    xi = first_crossing(full, t)
    if xi < 0:
        # lower bound = how far the rollout got PAST EOL, not the step count:
        # it launches at EOL - launch, so a 150-step rollout reaches EOL + 99.
        ae, crossed = float(steps - launch), False
    else:
        ae, crossed = float(abs(t0 + xi + 1 - eol_cyc)), True
    return dict(mae=mae, rmse=rmse, r2=r2,
                ar_mse50=float(np.mean((full[:n50] - tr[:n50]) ** 2)),
                ar_mse=float(np.mean((full[:n_true] - tr) ** 2)),
                n_true=n_true, ae=ae, crossed=crossed, tail=len(s) - eol_cyc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[1, 100],
                    help="training-trajectory counts, taken as nested prefixes")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--steps", type=int, default=10000,
                    help="TOTAL gradient steps, matched across every arm")
    ap.add_argument("--epochs", type=int, default=None,
                    help="fix the epoch count instead of matching gradient steps; "
                         "the two conventions answer different questions")
    ap.add_argument("--pool-seed", type=int, default=0)
    ap.add_argument("--tag", default="gate")
    args = ap.parse_args()

    train, test, tp, sp = make_pool(115, 5, args.pool_seed)
    tr_names = list(train)                      # generation order, never sorted
    te_names = list(test)
    all_tr = np.concatenate([train[c] for c in tr_names])
    lo, hi = float(all_tr.min()), float(all_tr.max())

    print("=" * 100)
    print(f"  SYNTHETIC LEARNING CURVE  [{args.tag}]  pool seed={args.pool_seed}  "
          f"{len(tr_names)} train + {len(te_names)} test  W={W}")
    print(f"  lo={lo:.4f} hi={hi:.4f} (from ALL {len(tr_names)} training cells, "
          f"constant across arms)   steps/arm={args.steps} (matched)")
    print(f"  N={args.ns}  seeds={args.seeds}")
    print("=" * 100)
    hdr = (f"  {'N':>5}{'seed':>5}{'win':>7}{'ep':>7}"
           f"{'TF MAE':>9}{'TF R2':>8}{'AR MSE@50':>12}{'AR MSE':>10}"
           f"{'AR AE':>8}{'crossed':>9}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    os.makedirs(os.path.join("..", "checkpoints", "per_sp", "synth"), exist_ok=True)
    out = []
    for n in args.ns:
        cells = tr_names[:n]
        X, Y = build_windows(train, cells, lo, hi, W)
        spe = max(1, -(-len(X) // BATCH))                  # steps per epoch
        # Two conventions, deliberately both available.  Matched STEPS isolates
        # the data effect at constant compute but leaves the small-N arm
        # memorising one trajectory for thousands of epochs.  Fixed EPOCHS is how
        # anyone actually trains, at the cost of giving large N proportionally
        # more optimisation.  A conclusion that holds under only one of them is
        # not a conclusion.
        epochs = (args.epochs if args.epochs is not None
                  else max(1, args.steps // spe))
        tps.EPOCHS = epochs
        tps.PROG_EVERY = max(1, epochs // 8)
        for seed in args.seeds:
            t_wall = time.time()
            model = train_one(seed, X, Y, W)
            model.eval()
            rows = [eval_cell(model, test[c], first_crossing(test[c], p.eol),
                              p.eol, lo, hi)
                    for c, p in zip(te_names, sp)]
            g = lambda f: float(np.mean([r[f] for r in rows]))   # noqa: E731
            ncross = sum(r["crossed"] for r in rows)
            print(f"  {n:>5}{seed:>5}{len(X):>7}{epochs:>7}"
                  f"{g('mae'):>9.5f}{g('r2'):>8.4f}{g('ar_mse50'):>12.3e}"
                  f"{g('ar_mse'):>10.3e}{g('ae'):>8.3f}{ncross:>5}/{len(rows):<3}"
                  f"  [{time.time() - t_wall:.0f}s]", flush=True)
            out.append(dict(tag=args.tag, N=n, seed=seed, n_win=len(X),
                            epochs=epochs, steps=spe * epochs,
                            mae=g("mae"), r2=g("r2"),
                            ar_mse50=g("ar_mse50"), ar_mse=g("ar_mse"),
                            ae=g("ae"), n_crossed=ncross, n_test=len(rows),
                            per_cell=rows))
            torch.save({"state_dict": model.state_dict(), "W": W, "N": n,
                        "seed": seed, "lo": lo, "hi": hi},
                       os.path.join("..", "checkpoints", "per_sp", "synth",
                                    f"{args.tag}_N{n}_seed{seed}.pt"))
            del model
            torch.cuda.empty_cache()
        print()

    out_path = "results/synth_curve.json"
    prev = json.load(open(out_path)) if os.path.exists(out_path) else []
    json.dump(prev + out, open(out_path, "w"), indent=2)
    print(f"  saved {len(out)} rows to {out_path}")
    print(f"  AE is the mean over the {len(te_names)} held-out cells.  A rollout")
    print(f"  that never crossed within {AR_STEPS} steps carries the bound")
    print(f"  {AR_STEPS} - {LAUNCH_BEFORE} = {AR_STEPS - LAUNCH_BEFORE} (how far it got past EOL),")
    print(f"  so 'crossed' is reported but is NOT the metric.")


if __name__ == "__main__":
    main()
