"""Autoregressive rollout probe for any per-SP checkpoint.

One protocol, applied to whatever checkpoint you point it at:

    launch at EOL - LAUNCH_BEFORE, feed only the model's own output after that,
    run to the end of the series.

Written because the existing CALCE numbers (ar_decode_variants.py) go through
tar.load_ckpt, which hard-codes the W=64 file name and therefore cannot see a
..._W128.pt checkpoint at all.  Measuring a longer window against the W=64
baseline needs the SAME code on both sides, or the comparison carries the
harness as a variable -- which is how the last three data experiments went wrong.

Everything the model needs (W, lo, hi, eol_ah, test cell) comes out of the
checkpoint, so nothing is guessed from the dataset defaults.

AE is ALWAYS defined.  A rollout that never crosses is not a missing value: it
is an error of at least (last cycle reached - EOL), and it is reported with a
">=" so the bound is visible rather than the row being dropped.

Read-only: loads a checkpoint, prints.  Writes nothing.

    D:/anaconda/envs/py312/python.exe src/ar_probe.py \
        --ckpt ../checkpoints/per_sp/calce/SP500_seed1_W128.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from gdn_model import build_gdn_model                 # noqa: E402
from make_figures import load_series                  # noqa: E402
from train_per_sp import DEV, EPS                     # noqa: E402

LAUNCH_BEFORE = 50
AR_STEPS = 150          # rollout length, FIXED -- independent of how long the series is


def first_crossing(series, thr) -> int:
    for i in range(len(series) - 1):
        if series[i] >= thr > series[i + 1]:
            return i
    return -1


def rollout(model, seq, t0, W, steps, mode="zscore"):
    """Self-fed: after t0 nothing real is ever fed back in.

    `mode` must match how the checkpoint was TRAINED (read it from the
    checkpoint's `tgt_mode`, or the `_abs`/`_anchor`/`_alast` filename suffix):
      zscore       y = z*std+mean    abs     y = z
      anchor       y = z + mean      anchor_last  y = z + y_last
    Getting it wrong is silent: applying the z-score rule to an absolute-capacity
    model multiplies a ~0.7 capacity by a ~0.01 window std, so the model's own
    contribution becomes ~1% of the output and the trajectory collapses onto the
    window mean -- every seed then reports the same crossing, which is the tell.
    See docs/ar_freeze_findings.md section 7.
    """
    win = [float(v) for v in np.asarray(seq[t0 - W:t0], dtype=np.float64)]
    out = []
    with torch.no_grad():
        for _ in range(steps):
            x = np.asarray(win[-W:], dtype=np.float32)
            xt = torch.tensor(x[None, :, None], device=DEV)
            # torch.std on the tensor the model is fed -- same operator and
            # dtype as the training target's scale.  numpy's .std() is the
            # BIASED estimator (0.78% smaller at W=64) and that error lands
            # directly in `y = z*std+mean`, which this rollout then amplifies.
            wm, ws = float(x.mean()), float(xt[:, :, 0].std(dim=1)) + EPS
            z = float(model(xt))
            if mode == "abs":
                y = z
            elif mode == "anchor":
                y = z + wm
            elif mode == "anchor_last":
                y = z + float(x[-1])      # anchor on the last value, not the mean
            else:
                y = z * ws + wm
            out.append(y)
            win.append(y)
    return np.asarray(out)


def measure(model, seq_raw, lo, hi, thr_ah, W, t0, steps, mode="zscore"):
    """One test cell, one ABSOLUTE launch cycle.  None if the cell never reaches EOL.

    t0 is an absolute cycle rather than a distance back from EOL, so the launch
    points can be the protocol's own SP values (300/400/500): no extra parameter
    to choose or defend, and one SP300 checkpoint is scored at all of them.  That
    is also how Bellomo et al. report their autoregressive error profile
    (prediction cycles 50/150/300/450) -- a curve, not a single number, which is
    what tells convergence apart from a lucky freeze.
    """
    seq = (np.asarray(seq_raw, dtype=np.float64) - lo) / (hi - lo + EPS)
    thr = (thr_ah - lo) / (hi - lo + EPS)
    eol = first_crossing(seq, thr) + 1
    if eol <= 0 or t0 < W or t0 >= len(seq):
        return None
    full = rollout(model, seq, t0, W, steps, mode)
    # The rollout is self-fed, so it can outrun the data -- and it should.  A
    # cell that crosses 80% while fading ~1%/cycle has only ~20 cycles of
    # capacity left, so stopping the rollout where the DATA stops would bound a
    # never-crossed rollout by a short tail (MIT: 18 cycles) and say almost
    # nothing.  MSE uses only the part that overlaps real values; the crossing
    # test uses the whole rollout.
    n_true = max(0, min(steps, len(seq) - t0))
    truth = seq[t0:t0 + n_true]
    n50 = min(50, n_true)
    xi = first_crossing(full, thr)
    if xi < 0:
        # how far the rollout got PAST EOL -- the lower bound on the error
        lb = (t0 + steps - 1) - eol
        ae_val, ae_lbl, crossed = float(lb), f">={lb}", False
    else:
        ae_val = float(abs(t0 + xi + 1 - eol))
        ae_lbl, crossed = f"{ae_val:.0f}", True
    return dict(eol=eol, t0=t0, n=len(full), n_true=n_true, slen=len(seq),
                crossed=crossed, ae=ae_val, ae_lbl=ae_lbl,
                mse50=float(np.mean((full[:n50] - truth[:n50]) ** 2)),
                mse=float(np.mean((full[:n_true] - truth) ** 2)) if n_true
                else float("nan"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="calce")
    ap.add_argument("--launch-at", type=int, nargs="+", default=None,
                    help="absolute cycle(s) to launch from -- normally the SP "
                         "values (300 400 500).  One SP300 checkpoint scored at "
                         "all of them gives the error PROFILE, which is what "
                         "separates convergence from a lucky freeze.")
    ap.add_argument("--launch-before", type=int, default=LAUNCH_BEFORE,
                    help="legacy: launch at EOL - this.  Used only when "
                         "--launch-at is absent.")
    ap.add_argument("--steps", type=int, default=AR_STEPS,
                    help="rollout length, FIXED regardless of series length")
    ap.add_argument("--mode", choices=["zscore", "abs", "anchor",
                                       "anchor_last"],
                    default=None,
                    help="must match how the checkpoint was trained: zscore "
                         "y=z*std+mean, abs y=z, anchor y=z+mean, "
                         "anchor_last y=z+y_last.  Defaults to the "
                         "checkpoint's own tgt_mode, which is the only safe "
                         "source -- a mismatch is silent and makes every seed "
                         "report the same crossing.")
    ap.add_argument("--quiet", action="store_true",
                    help="one line per (checkpoint, cell, launch point)")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location=DEV, weights_only=False)
    W = int(ck["W"])
    lo, hi = float(ck["lo"]), float(ck["hi"])
    eol_ah = float(ck["eol_ah"])
    # The checkpoint is the authority on how it was trained.  Older files predate
    # the tgt_mode field; fall back to the filename suffix rather than to a guess.
    mode = args.mode or ck.get("tgt_mode")
    if mode is None:
        base = os.path.basename(args.ckpt)
        mode = ("abs" if "_abs" in base else
                "anchor_last" if "_alast" in base else
                "anchor" if "_anchor" in base else "zscore")
        print(f"  # {base}: no tgt_mode in the checkpoint; inferred '{mode}' "
              f"from the filename")
    caps, _tr, _te, _W_ds, _sps, _eol = load_series(args.dataset)
    model = build_gdn_model(
        multiscale=True, stage_query=True, input_dim=1, window_size=W,
        output_len=1, readout="last").to(DEV)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    k = args.launch_before
    name = os.path.basename(args.ckpt)
    # EVERY test cell the checkpoint was trained to be scored on, not just the
    # first: MIT has two (batch2_cell5 and batch2_cell47), and taking [0]
    # silently halved every MIT measurement.
    cells = [c for c in ck["test_cells"] if c in caps]
    missing = [c for c in ck["test_cells"] if c not in caps]
    if missing:
        print(f"  # WARNING not in load_series('{args.dataset}'): {missing}")

    done = []
    for tc in cells:
        seq = (np.asarray(caps[tc], dtype=np.float64) - lo) / (hi - lo + EPS)
        eol = first_crossing(seq, (eol_ah - lo) / (hi - lo + EPS)) + 1
        if eol <= 0:
            print(f"  # {tc} never crosses {eol_ah} Ah -- skipped")
            continue
        pts = args.launch_at if args.launch_at else [eol - k]
        for t0 in pts:
            r = measure(model, caps[tc], lo, hi, eol_ah, W, t0, args.steps,
                        mode)
            if r is None:
                print(f"  # {tc} launch@{t0}: no usable window -- skipped")
                continue
            r["tc"] = tc
            done.append(r)
            rul0 = eol - t0                      # true remaining life at launch
            if args.quiet:
                print(f"  {name:<24}{tc:<14}{'@' + str(t0):>6}"
                      f"  MSE@50={r['mse50']:.3e}  MSE={r['mse']:.3e}"
                      f"  AE={r['ae_lbl']:>5}  AE/k={r['ae'] / rul0:.2f}"
                      f"  {'Y' if r['crossed'] else 'n'}")
            else:
                print(f"  ckpt      {name}"
                      + (f"  (ep {ck['ep']})" if "ep" in ck else "")
                      + f"   test={tc}   launch@{t0}")
                print(f"  protocol  W={W}  true EOL {eol}  remaining life {rul0};  "
                      f"rollout {r['n']} steps -> cycle {r['t0'] + r['n']}  "
                      f"(series {r['slen']}, {r['n_true']} steps have truth)")
                print(f"  MSE       {r['mse']:.3e}      MSE@50  {r['mse50']:.3e}")
                print(f"  AE        {r['ae_lbl']:<6}  AE/k = {r['ae'] / rul0:.2f}"
                      + ("" if r["crossed"] else "   (never crossed: lower bound)"))
    if not done:
        raise SystemExit("nothing could be scored")
    if len(done) > 1:
        print()
        print(f"  {'':<24}{'MEAN':<14}{'':>6}"
              f"  MSE@50={np.mean([r['mse50'] for r in done]):.3e}"
              f"  MSE={np.mean([r['mse'] for r in done]):.3e}"
              f"  AE={np.mean([r['ae'] for r in done]):.1f}"
              f"  crossed={sum(r['crossed'] for r in done)}/{len(done)}")




if __name__ == "__main__":
    main()
