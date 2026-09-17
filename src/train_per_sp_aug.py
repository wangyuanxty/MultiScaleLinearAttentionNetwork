"""Does more training data fix the autoregressive freeze?  A noise-augmentation test.

The hypothesis under test: the rollout freezes because the per-SP protocol trains
on too few cells (CALCE 3, NASA 3, PANASONIC/TJU/GOTION 2).  The direct test --
MIT, the only library with a surplus -- was run already (8 -> 39 real training
cells, 4.9x) and moved nothing, but that is one point.  This is the other way to
get a large training set: synthesise cells.

NASA is the host because it is short (168 cycles against CALCE's 881), so a
30-cell run costs about what a CALCE 3-cell run costs.  Two properties of NASA
shape the protocol and are worth stating up front:

  * W = 30, not 64.  The self-consistency value differs accordingly
    (|Z_CRIT| = 1.79 at W=30, against 1.76 at W=64), and nothing measured on
    CALCE transfers numerically.  The result here is a NASA result.
  * there is NO per-SP split here.  NASA's sps are [50, 70, 90] but the rollout
    launches at EOL - 50 = cycle 74, so SP=90 would start it inside the model's
    own training data.  Rather than pick whichever SPs happen to fit, the test
    cell is held out ENTIRELY: training sees only the other cells, and the test
    cell is scored from its first valid window.  That also drops SP as a second
    variable, which is what this experiment wants anyway.

The augmentation adds i.i.d. Gaussian noise to each REAL training cell's capacity
series to make `n_aug` synthetic cells per real one, so 3 real + 27 synthetic =
30.  Each synthetic is a complete trajectory with its own windows and targets --
it is a new cell, not an extra window off an existing one.  The noise sigma is
the cell's OWN measured per-cycle noise times `mult`, so the strength is set by
the data rather than picked by hand.  The test cell is never noised.

Two things are deliberately held fixed so the comparison isolates cell count and
nothing else:

  * lo/hi come from the 3 REAL training cells in BOTH configurations.  Taking
    them from all 30 would move the normalisation scale at the same time.
  * the test cell's pre-SP windows are always built from the real series.

Control is the same script with --n-augs 0, so baseline and augmented are
produced by identical code.

Writes only new paths: checkpoints/per_sp/nasa_aug/ (created, unused) and
results/per_sp_nasa_aug.json.

    D:/anaconda/envs/py312/python.exe src/train_per_sp_aug.py \
        --n-augs 0 9 --sps 50 70 --seeds 1 2 3
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

from make_figures import load_series                       # noqa: E402
from train_per_sp import (DEV, EPS, build_windows, eval_sp,  # noqa: E402
                          train_one)

LAUNCH_BEFORE = 50      # launch this many cycles before the true EOL
RUN_AFTER = 50          # and roll out this many past it (series length caps it)
RESID_RADIUS = 9        # half-width of the local linear fit used to read noise


def cell_noise(s: np.ndarray, r: int = RESID_RADIUS) -> float:
    """Per-cycle noise of one capacity series, in Ah.

    Residual of a local linear fit: for each cycle, fit a line to the 2r+1
    points around it and take observed minus fitted.  A global polynomial would
    absorb the knee; a local line does not, so what is left is the measurement
    scatter rather than the degradation signal.
    """
    s = np.asarray(s, dtype=np.float64)
    x = np.arange(-r, r + 1)
    res = [float(s[i] - np.polyval(np.polyfit(x, s[i - r:i + r + 1], 1), 0))
           for i in range(r, len(s) - r)]
    return float(np.std(res)) if res else 0.0


def augment(caps, train_cells, n_aug, mult, seed=0):
    """The real cells plus `n_aug` noised copies of each, as a new cell dict.

    Copied, not in-place -- `caps` is left untouched.
    """
    rng = np.random.default_rng(seed)
    out, noise = {}, {}
    for c in train_cells:
        s = np.asarray(caps[c], dtype=np.float64)
        out[c] = s.copy()
        noise[c] = cell_noise(s)
        for k in range(n_aug):
            out[f"{c}_a{k + 1:02d}"] = s + rng.normal(
                0.0, mult * noise[c], size=s.shape)
    return out, noise


def first_crossing(series, thr) -> int:
    for i in range(len(series) - 1):
        if series[i] >= thr > series[i + 1]:
            return i
    return -1


def ar_rollout(model, seq, t0, W, steps):
    """Self-fed rollout with the plain z-score decode -- no correction of any kind.

    The whole point is to measure whether MORE DATA alone changes the freeze, so
    any decode-side fix (sigma from a training table, clamping z) would confound
    the answer.  This is the decode the model was trained for, verbatim.
    """
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


def run_one(tr_caps, tr_cells, test_caps, te, W, eol_ah, lo, hi, seed, tag):
    """Train one model and measure it both ways.  Returns a row dict.

    The test cell contributes NOTHING to training -- no per-SP split -- so the
    training set is exactly the training cells and the difference between
    configurations is exactly the cell count.  `tr_caps` holds the training
    cells (real + synthetic); `test_caps` holds the untouched real series, and
    is kept separate so a synthetic cell can never be scored by accident.
    """
    X, Y = build_windows(tr_caps, tr_cells, lo, hi, W)
    model = train_one(seed, X, Y, W)

    # sp=W: score from the test cell's first valid window, i.e. the whole series
    rows = eval_sp(model, test_caps, [te], lo, hi, W, W, eol_ah)
    g = lambda f: float(np.mean([r[f] for r in rows]))      # noqa: E731

    seq = (np.asarray(test_caps[te], dtype=np.float64) - lo) / (hi - lo + EPS)
    thr = (eol_ah - lo) / (hi - lo + EPS)
    eol = first_crossing(seq, thr) + 1
    t0 = eol - LAUNCH_BEFORE
    steps = min(LAUNCH_BEFORE + RUN_AFTER, len(seq) - t0)
    full = ar_rollout(model, seq, t0, W, steps)
    truth = seq[t0:t0 + len(full)]
    n50 = min(LAUNCH_BEFORE, len(full))
    xi = first_crossing(full, thr)
    pred_cyc = None if xi < 0 else t0 + xi + 1

    del model
    torch.cuda.empty_cache()
    return dict(tag=tag, seed=seed, n_train=len(tr_cells), n_win=len(X),
                MAE=g("MAE"), R2=g("R2"), AE_tf=g("AE"),
                ar_mse50=float(np.mean((full[:n50] - truth[:n50]) ** 2)),
                ar_mse=float(np.mean((full - truth) ** 2)),
                ar_steps=len(full), eol=eol,
                pred_cyc=pred_cyc,
                ar_ae=(None if pred_cyc is None else abs(pred_cyc - eol)),
                ar_final=float(full[-1]), ar_true_final=float(truth[-1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nasa")
    ap.add_argument("--n-augs", type=int, nargs="+", default=[0, 9],
                    help="synthetic copies per real training cell; 0 = control")
    ap.add_argument("--noise-mult", type=float, default=1.0,
                    help="noise sigma as a multiple of the cell's own noise")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args()

    caps, tr, te, W, _sps, eol_ah = load_series(args.dataset)
    test_cells = [te]
    real_tr = list(tr)

    # fixed for every configuration -- see the module docstring
    all_real = np.concatenate([np.asarray(caps[c]) for c in real_tr])
    lo, hi = float(all_real.min()), float(all_real.max())

    seq_len = len(np.asarray(caps[te]))
    print("=" * 104)
    print(f"  {args.dataset.upper()} noise augmentation  test={te}  W={W}"
          f"  eol={eol_ah} Ah  real train cells {len(real_tr)}")
    print(f"  lo={lo:.4f} hi={hi:.4f} (from the {len(real_tr)} REAL cells, "
          f"fixed across configs)   test series {seq_len} cycles")
    print("=" * 104)

    os.makedirs(os.path.join("..", "checkpoints", "per_sp", "nasa_aug"),
                exist_ok=True)
    out_path = "results/per_sp_nasa_aug.json"
    out = json.load(open(out_path)) if os.path.exists(out_path) else []
    done = {(r["tag"], r["seed"]) for r in out}

    for n_aug in args.n_augs:
        aug, noise = augment(caps, real_tr, n_aug, args.noise_mult)
        tr_cells = sorted(aug)
        tag = f"{len(tr_cells)}c"
        if n_aug:
            print(f"  n_aug={n_aug}: {len(tr_cells)} training cells "
                  f"({len(real_tr)} real + {n_aug * len(real_tr)} synthetic)"
                  f"   per-cell noise (Ah): "
                  + "  ".join(f"{c}:{noise[c]:.4f}" for c in real_tr))
            print(f"    added sigma = {args.noise_mult:.2f}x own noise"
                  f" -> total noise x{np.sqrt(1 + args.noise_mult ** 2):.2f}")
        else:
            print(f"  n_aug=0 (CONTROL): {len(tr_cells)} real training cells only")

        for seed in args.seeds:
            key = (tag, seed)
            if key in done:
                print(f"    {tag} seed{seed}: SKIP (in json)")
                continue
            t_wall = time.time()
            row = run_one(aug, tr_cells, caps, te, W, eol_ah, lo, hi, seed, tag)
            row["noise_mult"] = args.noise_mult
            row["noise"] = {c: noise[c] for c in real_tr}
            out.append(row)
            json.dump(out, open(out_path, "w"), indent=2)
            ar_ae = ("--" if row["ar_ae"] is None else f"{row['ar_ae']}")
            print(f"    {tag:>4} seed{seed}: "
                  f"TF MAE={row['MAE']:.4f} R2={row['R2']:.4f} | "
                  f"AR MSE@50={row['ar_mse50']:.2e} "
                  f"MSE={row['ar_mse']:.2e} AE={ar_ae:>4}"
                  f"  [{time.time() - t_wall:.0f}s]", flush=True)
        print()

    print("=" * 104)
    print(f"  {'cells':>5}{'win':>8}{'seeds':>7}{'TF MAE':>10}{'TF R2':>9}"
          f"{'AR MSE@50':>12}{'AR MSE':>11}{'AR AE':>9}{'crossed':>9}")
    print("-" * 104)
    for n_aug in args.n_augs:
        n_tr = len(real_tr) * (1 + n_aug)
        rs = [r for r in out if r["tag"] == f"{n_tr}c"]
        if not rs:
            continue
        crossed = [r["ar_ae"] for r in rs if r["ar_ae"] is not None]
        ae = "--" if not crossed else f"{np.mean(crossed):.0f}"
        print(f"  {n_tr:>5}{rs[0]['n_win']:>8}{len(rs):>7}"
              f"{np.mean([r['MAE'] for r in rs]):>10.4f}"
              f"{np.mean([r['R2'] for r in rs]):>9.4f}"
              f"{np.mean([r['ar_mse50'] for r in rs]):>12.2e}"
              f"{np.mean([r['ar_mse'] for r in rs]):>11.2e}"
              f"{ae:>9}{len(crossed):>5}/{len(rs):<3}")
    print()
    print(f"  Control and augmented runs are the same code path; only the cell")
    print(f"  count differs.  No per-SP split: the test cell is held out whole.")
    print(f"  TF = the per-step non-recursive pass over the test series; AR = a")
    print(f"  self-fed rollout launched {LAUNCH_BEFORE} cycles before EOL with")
    print(f"  the plain z-score decode, no correction.  A rollout that never")
    print(f"  crosses contributes no AR AE.")


if __name__ == "__main__":
    main()
