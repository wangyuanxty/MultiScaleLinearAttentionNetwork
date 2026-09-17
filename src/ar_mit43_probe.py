"""Autoregressive probe for the MIT all-cells run, at any checkpoint.

The 39-cell run is long enough that it keeps an intermediate checkpoint every
PROG_SAVE epochs, so this measures the rollout at whatever epoch you point it
at -- the point being to see whether MORE TRAINING CELLS changes the
autoregressive behaviour without waiting for the run to finish.

Same measurement as the one used for the 8-cell baseline, so the two are
directly comparable:

  launch  100 cycles before the cell's true EOL crossing
  feed    only the model's own output from the launch onward
  report  trajectory MAE over the first 50 / 100 steps
          the window sigma at the end, as a multiple of the smallest sigma any
          REAL window of that cell has (scale-free, so it compares across
          datasets: ~0 means the window collapsed)
          the final value

Read-only: loads a checkpoint and prints.

    D:/anaconda/envs/py312/python.exe src/ar_mit43_probe.py \
        --ckpt ../checkpoints/per_sp/mit43/SP300_seed1_ep025.pt
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

import test_ar_rollout as tar          # noqa: E402
from gdn_model import build_gdn_model   # noqa: E402
from make_figures import load_series   # noqa: E402

EPS = 1e-6
TEST_CELL = "batch2_cell5"
EOL_AH = 0.86
LAUNCH_BEFORE_EOL = 100


def first_crossing(series, thr):
    for i in range(len(series) - 1):
        if series[i] >= thr > series[i + 1]:
            return i
    return -1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--launch-before-eol", type=int,
                    default=LAUNCH_BEFORE_EOL)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location=tar.DEV, weights_only=False)
    model = build_gdn_model(
        multiscale=True, stage_query=True, input_dim=1,
        window_size=ck["W"], output_len=1, readout="last").to(tar.DEV)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    caps, _tr, _te, _W, _sps, _eol = load_series("mit")
    lo, hi, W = ck["lo"], ck["hi"], ck["W"]
    seq = (np.asarray(caps[TEST_CELL], dtype=np.float32) - lo) / (hi - lo + EPS)
    thr = (EOL_AH - lo) / (hi - lo + EPS)
    eol = first_crossing(seq, thr) + 1
    if eol <= 0:
        raise SystemExit(f"{TEST_CELL} never crosses {EOL_AH} Ah")

    # smallest sigma any REAL window of this cell has -- the yardstick the
    # rollout's sigma has to be compared against
    sigma_floor = float(min(seq[i - W:i].std() for i in range(W, len(seq))))

    t0 = eol - args.launch_before_eol
    full = tar.rollout(model, seq, t0, W, len(seq) - t0)
    truth = seq[t0:t0 + len(full)]
    n = len(full)

    win = list(seq[t0 - W:t0])
    sigmas = []
    for j in range(n):
        sigmas.append(float(np.std(win[-W:])))
        win.append(full[j])

    def mae(k):
        return (float(np.mean(np.abs(full[:k] - truth[:k])))
                if n >= k else float("nan"))

    def mse(k):
        return (float(np.mean((full[:k] - truth[:k]) ** 2))
                if n >= k else float("nan"))

    # The threshold in this checkpoint's normalisation.  A rollout that never
    # goes under it has no measured AE -- but dropping it is worse than bounding
    # it.  "Never crossed" is not a missing value: it is an error of at least
    # (last cycle reached - EOL), and reporting it as "--" is precisely the
    # one-bit masking that the AE convention exists to avoid.  Non-crossers
    # therefore carry that lower bound, flagged ">=".
    xi = first_crossing(full, thr)
    pred_cyc = None if xi < 0 else t0 + xi + 1
    true_cyc = first_crossing(truth, thr)
    true_cyc = None if true_cyc < 0 else t0 + true_cyc + 1
    if pred_cyc is None:
        ae_val, ae_lbl = float(t0 + n - eol), f">={t0 + n - eol}"
    else:
        ae_val = float(abs(pred_cyc - eol))
        ae_lbl = f"{ae_val:.0f}"

    k = args.launch_before_eol
    print(f"  ckpt      {os.path.basename(args.ckpt)}"
          + (f"  (ep {ck['ep']})" if "ep" in ck else ""))
    print(f"  protocol  launch cycle {t0} = EOL - {k};  rollout {n} steps "
          f"-> cycle {t0 + n}  (W={W}, series {len(seq)})")
    print(f"  MSE       {mse(n):.3e}      MAE  {mae(n):.5f}")
    print(f"  MSE@{k:<4}   {mse(k):.3e}      MAE@{k:<3} {mae(k):.5f}")
    print(f"  AE        {ae_lbl:<6}  AE/k = {ae_val / k:.2f}"
          + ("" if pred_cyc else "   (never crossed: lower bound)"))
    print(f"  sigma end {sigmas[-1]:.5f}   floor {sigma_floor:.5f}"
          f"   ratio {sigmas[-1] / sigma_floor:.1f}"
          f"   ({'COLLAPSED' if sigmas[-1] < sigma_floor else 'in range'})")
    print(f"  final     {float(full[-1]):.4f}"
          f"   (truth at end {float(truth[-1]):.4f})")
    print()
    print("  Same test cell / same seed / same protocol against the 8-cell")
    print("  split: --ckpt ../checkpoints/per_sp/mit/SP300_seed1.pt")


if __name__ == "__main__":
    main()
