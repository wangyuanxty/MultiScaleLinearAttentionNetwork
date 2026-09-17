"""Where do the rollouts that never crossed actually end up?

"Did not cross" and "froze" are different claims.  A rollout that ends at 0.29
with a threshold at 0.30 merely fell short; one that ends at 0.55 has not moved
at all.  The profile table collapses both into the same word, so this prints the
raw end state next to the threshold and the launch value.

Read-only: loads checkpoints and series, prints.  Writes nothing.

    cd src && D:/anaconda/envs/py312/python.exe diag_noncross.py --launch 300
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

import ar_probe as ap                                    # noqa: E402
from gdn_model import build_gdn_model                    # noqa: E402
from make_figures import load_series                     # noqa: E402
from train_per_sp import DEV, EPS                        # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--launch", type=int, default=300)
    p.add_argument("--steps", type=int, default=650)
    p.add_argument("--datasets", nargs="+", default=["calce", "panasonic"])
    args = p.parse_args()

    for ds in args.datasets:
        print("=" * 100)
        print("  %s   launch@%d   %d steps" % (ds.upper(), args.launch, args.steps))
        print("=" * 100)
        print("  %-14s %9s %9s %9s %9s %8s  %s"
              % ("ckpt", "launch", "thr", "final", "moved", "crossed", "verdict"))
        for seed in range(1, 11):
            path = os.path.join("..", "checkpoints", "per_sp", ds,
                                "SP300_seed%d.pt" % seed)
            if not os.path.exists(path):
                continue
            ck = torch.load(path, map_location=DEV, weights_only=False)
            W, lo, hi = int(ck["W"]), float(ck["lo"]), float(ck["hi"])
            eol_ah = float(ck["eol_ah"])
            caps, _tr, te, _W, _sps, _e = load_series(ds)
            seq = (np.asarray(caps[te], dtype=np.float64) - lo) / (hi - lo + EPS)
            thr = (eol_ah - lo) / (hi - lo + EPS)
            model = build_gdn_model(
                multiscale=True, stage_query=True, input_dim=1, window_size=W,
                output_len=1, readout="last").to(DEV)
            model.load_state_dict(ck["state_dict"])
            model.eval()
            full = ap.rollout(model, seq, args.launch, W, args.steps, "zscore")
            launched_at = float(seq[args.launch - 1])
            moved = float(full[-1] - launched_at)
            crossed = bool((full < thr).any())
            if crossed:
                verdict = "crossed"
            elif full[-1] < thr + 0.05:
                verdict = "fell just short"
            elif abs(moved) < 0.02:
                verdict = "FROZEN (barely moved)"
            else:
                verdict = "declining, too slow"
            print("  %-14s %9.4f %9.4f %9.4f %+9.4f %8s  %s"
                  % ("SP300_seed%d" % seed, launched_at, thr,
                     float(full[-1]), moved, crossed, verdict))
            del model
            torch.cuda.empty_cache()
        print()


if __name__ == "__main__":
    main()
