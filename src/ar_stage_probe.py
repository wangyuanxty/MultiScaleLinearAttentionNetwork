"""Does the model know what stage of degradation the cell is in?

Motivation.  At inference the model sees exactly one thing: a W-long window of
normalised capacity.  Nothing else -- no cycle index, no RUL feature, no level
token.  So the stage is only reachable through the values themselves.  But in
REAL data the stage and the window's SHAPE are almost the same variable:

    high level (early life)  -> flat window
    low  level (knee)        -> steep window

so a model can score well while reading only the shape and treating the level as
decoration.  The autoregressive rollout breaks that coupling -- it manufactures
low level with a flat window, a combination that does not occur in training --
which would explain the freeze without the model being "wrong" about anything.

This script separates the two readings with a direct manipulation.  The window's
level and its spread are moved INDEPENDENTLY of each other:

  A1  level sweep, shape fixed   add a constant to a real window -- sigma and the
                                 demeaned profile are untouched, only the level
                                 moves.  Flat response => the model reads shape.
  A2  spread sweep, level fixed  demean a real window, rescale, re-add the same
                                 mean -- only sigma moves.  This is the control;
                                 without it, "flat response" in A1 is ambiguous.

  B   the stage->sigma manifold  the sig-train table (level -> typical spread of
                                 a REAL window at that level) plus the test
                                 cell's own real (level, sigma) path, so the
                                 coupling the rollout violates is visible.

Read-only: loads a checkpoint, prints.  Writes nothing.

    D:/anaconda/envs/py312/python.exe src/ar_stage_probe.py
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

import test_ar_rollout as tar                              # noqa: E402
from ar_decode_variants import EPS, sigma_table            # noqa: E402
from make_figures import load_series                       # noqa: E402

EOL = 640
DELTAS = (-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20)
SCALES = (0.25, 0.5, 1.0, 2.0)
# (label, index of the window's last cycle) -- early life, mid life, the knee
ANCHORS = (("early", 128), ("mid", 400), ("knee", EOL))


def model_z(model, win: np.ndarray) -> float:
    """The model's z for one window.  Nothing else is fed to it."""
    x = np.asarray(win, dtype=np.float32)
    with torch.no_grad():
        return float(model(torch.tensor(x[None, :, None], device=tar.DEV)))


def probe_a1(model, seq, W):
    """Level sweep at fixed shape: z(window + delta)."""
    print("  A1. level sweep, SHAPE HELD FIXED  (z as the whole window shifts)")
    print(f"      {'anchor':<8}{'level':>8}{'sigma':>9}"
          + "".join(f"{d:>+9.2f}" for d in DELTAS) + f"{'dz/dlev':>10}")
    for label, end in ANCHORS:
        base = seq[end - W:end].astype(np.float64)
        lev, sig = float(base.mean()), float(base.std())
        zs = [model_z(model, base + d) for d in DELTAS]
        # least-squares slope of z against the level actually fed
        lv = np.asarray(DELTAS) + lev
        slope = float(np.polyfit(lv, zs, 1)[0])
        print(f"      {label:<8}{lev:>8.3f}{sig:>9.5f}"
              + "".join(f"{z:>+9.3f}" for z in zs) + f"{slope:>10.2f}")


def probe_a2(model, seq, W):
    """Spread sweep at fixed level: reshape the window, put the mean back."""
    print()
    print("  A2. spread sweep, LEVEL HELD FIXED  (control for A1)")
    print(f"      {'anchor':<8}{'level':>8}{'sigma0':>9}"
          + "".join(f"{s:>9.2f}x" for s in SCALES) + f"{'dz/dsig':>10}")
    for label, end in ANCHORS:
        base = seq[end - W:end].astype(np.float64)
        lev, sig = float(base.mean()), float(base.std())
        centred = base - base.mean()
        zs = [model_z(model, lev + centred * s) for s in SCALES]
        sg = np.asarray(SCALES) * sig
        slope = float(np.polyfit(sg, zs, 1)[0])
        print(f"      {label:<8}{lev:>8.3f}{sig:>9.5f}"
              + "".join(f"{z:>+9.3f}" for z in zs) + f"{slope:>10.2f}")
    print("      dz/dlev in A1 against dz/dsig here is the whole answer: if the")
    print("      level slope is ~0 while the sigma slope is not, the model reads")
    print("      shape and the level is decoration.")


def probe_b(caps32, ck, te, W):
    """The stage -> sigma manifold, and where the test cell actually lives."""
    centres, tabs = sigma_table(caps32, ck["train_cells"], ck["lo"], ck["hi"], W)
    print()
    print("  B. the stage -> sigma manifold")
    print("      sig-train table, from the TRAINING cells' own windows:")
    print(f"      {'level':>8}{'sigma':>10}   bar")
    lo_s, hi_s = float(tabs.min()), float(tabs.max())
    for c, t in zip(centres, tabs):
        bar = "#" * max(1, int(round(30 * (t - lo_s) / (hi_s - lo_s + EPS))))
        print(f"      {c:>8.3f}{t:>10.5f}   {bar}")
    mono = bool(np.all(np.diff(tabs) > 0))
    print(f"      monotone increasing in level?  {mono}"
          f"   (sigma min {lo_s:.5f} at level "
          f"{centres[int(np.argmin(tabs))]:.3f},"
          f" max {hi_s:.5f} at {centres[int(np.argmax(tabs))]:.3f})")

    # the test cell's OWN real windows -- the path the rollout is supposed to
    # stay on and does not
    s = (np.asarray(caps32[te], dtype=np.float64) - ck["lo"]) / (
        ck["hi"] - ck["lo"] + EPS)
    rows = [(float(s[i - W:i].mean()), float(s[i - W:i].std()))
            for i in range(W, len(s))]
    means = np.asarray([r[0] for r in rows])
    sigs = np.asarray([r[1] for r in rows])
    print()
    print(f"      the test cell's real (level, sigma) path, {te}:")
    print(f"      {'level band':>14}{'n':>6}{'sigma mean':>12}{'sigma min':>11}")
    edges = np.linspace(means.min(), means.max(), 6)
    for j in range(5):
        m = (means >= edges[j]) & (means <= edges[j + 1])
        if not m.any():
            continue
        print(f"      {edges[j]:>6.3f}-{edges[j+1]:<7.3f}{int(m.sum()):>6}"
              f"{sigs[m].mean():>12.5f}{sigs[m].min():>11.5f}")
    print("      narrow sigma band per level = shape and stage are near-")
    print("      interchangeable in real data, which is what lets the rollout")
    print("      produce a combination the model never saw.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="calce")
    ap.add_argument("--sp", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    caps, _tr, te, W, _sps, _eol = load_series(args.dataset)
    caps32 = {c: caps[c].astype(np.float32) for c in caps}
    model, ck = tar.load_ckpt(args.dataset, args.sp, args.seed)
    seq = (np.asarray(caps32[te], dtype=np.float64) - ck["lo"]) / (
        ck["hi"] - ck["lo"] + EPS)

    print("=" * 88)
    print(f"  stage probe: does the model use the level, or only the shape?")
    print(f"  {args.dataset.upper()} {te} SP{args.sp} seed{args.seed}  W={W}"
          f"  input = {W} normalised capacities, nothing else")
    print("=" * 88)
    probe_a1(model, seq, W)
    probe_a2(model, seq, W)
    probe_b(caps32, ck, te, W)


if __name__ == "__main__":
    main()
