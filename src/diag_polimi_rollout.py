"""Print one PoliMi autoregressive rollout, step by step, next to the truth.

A single RUL error cannot tell "the model is biased" from "the decode is wrong":
both show up as a big number.  The trajectory can.  This dumps the input window,
the model's raw output, and the decoded value for the first steps, then samples
the rest -- so a smooth over-decline, a constant (the freeze), and a jump to a
nonsensical value are all distinguishable at a glance.

Read-only: loads a checkpoint and a series, prints.  Writes nothing.

    cd src && D:/anaconda/envs/py312/python.exe diag_polimi_rollout.py \
        --ckpt ../checkpoints/per_sp/polimi/cell1_seed1.pt --launch 50
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

from gdn_model import build_gdn_model               # noqa: E402
from polimi_tub import EOL_SOH, QNOM, soh_series    # noqa: E402
from train_per_sp import DEV, EPS                   # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--launch", type=int, default=50)
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location=DEV, weights_only=False)
    W, lo, hi = int(ck["W"]), float(ck["lo"]), float(ck["hi"])
    test = ck["test"]
    model = build_gdn_model(multiscale=True, stage_query=True, input_dim=1,
                            window_size=W, output_len=1, readout="last").to(DEV)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    S = soh_series()
    soh = S[test][1]
    seq = (soh - lo) / (hi - lo + EPS)
    thr = (EOL_SOH - lo) / (hi - lo + EPS)

    print("=" * 92)
    print(f"  {os.path.basename(args.ckpt)}   test={test}   W={W}")
    print(f"  lo={lo:.4f} hi={hi:.4f}   EOL(SoH {EOL_SOH}) -> normalised {thr:.4f}")
    print(f"  series length {len(soh)} cycles;  true SoH at launch "
          f"{soh[args.launch - 1]:.4f} (normalised {seq[args.launch - 1]:.4f})")
    print("=" * 92)

    win = [float(v) for v in seq[args.launch - W:args.launch]]
    print(f"  launch window (cycles {args.launch - W}..{args.launch - 1})")
    print(f"    min {min(win):.5f}  max {max(win):.4f}  mean {np.mean(win):.5f}"
          f"  std {np.std(win):.6f}")
    print()

    print(f"  {'step':>5}{'cycle':>7}{'raw z':>10}{'window mean':>13}"
          f"{'window std':>12}{'decoded':>10}{'truth':>10}")
    out = []
    with torch.no_grad():
        for i in range(args.steps):
            x = np.asarray(win[-W:], dtype=np.float32)
            wm, ws = float(x.mean()), float(x.std()) + EPS
            z = float(model(torch.tensor(x[None, :, None], device=DEV)))
            y = z                       # absolute target: decoded == raw output
            out.append(y)
            win.append(y)
            if i < 6 or i % 25 == 0:
                cyc = args.launch + i
                t = seq[cyc] if cyc < len(seq) else float("nan")
                print(f"  {i:>5}{cyc:>7}{z:>10.5f}{wm:>13.5f}{ws:>12.6f}"
                      f"{y:>10.5f}{t:>10.5f}")
    full = np.asarray(out)
    print()
    print(f"  rollout: min {full.min():.4f}  max {full.max():.4f}  "
          f"final {full[-1]:.4f}   (threshold {thr:.4f})")
    xs = np.where(full < thr)[0]
    print(f"  first step below threshold: "
          + (f"{xs[0]}  -> cycle {args.launch + xs[0]}" if len(xs) else "never"))
    print(f"  per-step change: mean {np.mean(np.diff(full)):+.6f}  "
          f"first 10 mean {np.mean(np.diff(full[:11])):+.6f}  "
          f"last 10 mean {np.mean(np.diff(full[-11:])):+.6f}")
    print(f"  truth over the same span: "
          f"{seq[args.launch]:.4f} -> "
          f"{seq[min(args.launch + args.steps, len(seq) - 1)]:.4f}")


if __name__ == "__main__":
    main()
