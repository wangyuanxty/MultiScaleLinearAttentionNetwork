"""Long W loses slope -- is it curvature, or capacity regeneration?

delta = (W+1)/2 * (1/A - r/sigma) is read off the launch window alone.  Two
different causes can move it as W grows, and they call for opposite conclusions:

  * CONVEX fade -- the window spans a stretch that is not a line, so the fitted
    slope understates the rate at the launch point and the spread about that
    line is large.  A quadratic absorbs it, so this is a statement about the
    window being long, and a shorter one would not have the problem.

  * CAPACITY REGENERATION -- transient recovery spikes.  They add variance with
    no systematic slope change, and a quadratic does NOT absorb them.  This is a
    statement about the cell, present at any W.

Fitting degree 1 and degree 2 to the SAME window separates them: if the
residual collapses when the quadratic term is added, the spread was curvature.

Also reports the window's trend share, (r*A)^2 / sigma^2, which is what the
delta argument actually turns on -- and both std conventions side by side, since
A must match whichever estimator the decode uses (torch.std is ddof=1).

No model, no checkpoint involved; the window is taken from the measured series.

    python src/diag_regen_vs_curvature.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from make_figures import load_series                          # noqa: E402

WS = (32, 64, 128, 256, 512)
K = 50                      # launch at EOL - K, matching the delta tables

# exactly the (dataset, launch, W) rows the theory doc quotes, so every delta in
# it can be recomputed rather than carried over
SPECS = (
    ("tju",    50,  64), ("tju",   100,  64), ("tju",    50,  32),
    ("tju",    50,  96), ("tju",    50, 128),
    ("mit",    50,  64), ("mit",   100,  64),
    ("calce",  50,  64), ("calce", 100,  64), ("calce",  50, 128),
    ("calce",  50, 256), ("calce",  50, 512),
    ("gotion", 50,  30), ("gotion", 100, 30), ("gotion", 50,  64),
    ("gotion", 50, 128), ("gotion", 50, 256), ("gotion", 50, 512),
)


def first_cross(cap: np.ndarray, thr: float) -> int:
    """First cycle at which capacity drops below the EOL threshold."""
    for i in range(len(cap) - 1):
        if cap[i] >= thr > cap[i + 1]:
            return i + 1
    return -1


def fit(win: np.ndarray):
    """Window fits: the two slopes, and the spread about each.

    `r` values are returned POSITIVE for a declining series -- polyfit gives
    the slope with respect to the cycle index, which is negative for capacity
    fade, and the delta formula wants the fade rate.

    r_local is the quadratic's slope AT THE LAUNCH POINT (the end of the
    window), i.e. the rate the cell is actually fading at there.  That is the
    column that separates the two candidate causes of an r_lin drop:

        convex fade  ->  r_lin falls with W, r_local holds  (window averages
                         over a flatter past)
        regeneration ->  both fall together                 (the line is being
                         fitted to spike structure, not to a trend)
    """
    x = np.arange(len(win), dtype=np.float64)
    p1 = np.polyfit(x, win, 1)
    p2 = np.polyfit(x, win, 2)
    r1 = win - np.polyval(p1, x)
    r2 = win - np.polyval(p2, x)
    r_local = float(-(2.0 * p2[0] * (len(win) - 1) + p2[1]))
    return (float(-p1[0]), r_local, float(win.std(ddof=1)),
            float(r1.std(ddof=1)), float(r2.std(ddof=1)))


def main() -> None:
    print("=" * 104)
    print("  Every (dataset, launch, W) the theory doc quotes, at full precision.")
    print("  delta = (W+1)/2 * (1/A - r/sigma).  A1 is the UNBIASED sd of a ramp")
    print("  (sqrt(W(W+1)/12), what torch.std returns and therefore what the")
    print("  decode uses); A0 is the biased one (sqrt((W^2-1)/12)) the doc used.")
    print("=" * 104)
    print("  %-7s %-8s %5s %13s %13s %9s %9s %9s %9s %9s"
          % ("ds", "launch", "W", "r", "sd", "r/sigma", "z_req", "z_crit",
             "delta_A1", "delta_A0"))
    cache = {}
    for ds, k, W in SPECS:
        if ds not in cache:
            caps, _tc, test_cell, _w, _s, eol_ah = load_series(ds)
            seq = np.asarray(caps[test_cell], dtype=np.float64).ravel()
            cache[ds] = (seq, first_cross(seq, eol_ah), test_cell)
        seq, eol, test_cell = cache[ds]
        t = eol - k
        if W >= t:
            continue
        r, _rloc, sd, _rl, _rq = fit(seq[t - W:t])
        A1 = np.sqrt(W * (W + 1) / 12.0)
        A0 = np.sqrt((W * W - 1) / 12.0)
        zreq = -(W + 1) / 2.0 * (r / sd)
        zc1, zc0 = -(W + 1) / (2 * A1), -(W + 1) / (2 * A0)
        print("  %-7s EOL-%-4d %5d %13.6e %13.6e %9.6f %+9.5f %+9.5f %+9.5f"
              " %+9.5f" % (ds, k, W, r, sd, r / sd, zreq, zc1,
                           zreq - zc1, zreq - zc0), flush=True)
    print(flush=True)

    for ds in ("calce", "tju", "gotion"):
        caps, train_cells, test_cell, _W_ds, _sps, eol_ah = load_series(ds)
        seq = np.asarray(caps[test_cell], dtype=np.float64).ravel()
        eol = first_cross(seq, eol_ah)
        if eol <= 0:
            print(f"{ds}: test cell never crosses EOL; skipped", flush=True)
            continue
        t = eol - K
        print("=" * 96)
        print(f"  {ds}  test={test_cell}  EOL={eol}  launch=EOL-{K}={t}"
              f"  (series {len(seq)})")
        print("=" * 96)
        print("  %5s %10s %10s %8s %10s %10s %8s %8s %9s"
              % ("W", "r_lin", "r_local", "loc/lin", "sd_total", "sd_resid",
                 "quad/lin", "trend%", "delta"))
        for W in WS:
            if W >= t:
                print("  %5d  skipped (W >= launch index)" % W)
                continue
            win = seq[t - W:t]
            r, rloc, sd, rl, rq = fit(win)
            A1 = np.sqrt(W * (W + 1) / 12.0)        # ddof=1 -- the decode's
            rA = r * A1
            # delta with ddof=1 throughout, i.e. matching the decode's torch.std
            d1 = (W + 1) / 2.0 * (1.0 / A1 - r / sd)
            # the value the current theory doc quotes: z_req at ddof=1 but
            # z_crit at ddof=0 -- kept only to show the size of the correction
            A0 = np.sqrt((W * W - 1) / 12.0)
            d_doc = (W + 1) / 2.0 * (1.0 / A0 - r / sd)
            trend = 100.0 * rA * rA / (sd * sd) if sd > 0 else float("nan")
            print("  %5d %10.3e %10.3e %8.3f %10.3e %10.3e %8.3f %8.1f %9.4f"
                  % (W, r, rloc, (rloc / r if r else float("nan")), sd, rl,
                     (rq / rl if rl > 0 else float("nan")), trend, d1),
                  flush=True)
            print("  %5s %10s %10s %8s %10s %10s %8s %8s %9.4f"
                  % ("", "", "", "", "", "", "", "doc-mix", d_doc), flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
