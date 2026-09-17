"""PoliMi-TUB loader -- the one published dataset whose autoregressive results we can compare against.

Bellomo et al. 2024 (Batteries 10(8):292) train on 5 of the 6 LG 18650HE4 cells and
predict the 6th, reporting RUL error at prediction cycles 50/150/300/450:

    Autoregressive  103 -> 81 -> 47 -> 21   (monotone convergence)

Ours freezes on CALCE (2/10 seeds cross at all).  Running our model on THEIR data
under THEIR protocol is the only way to separate "our model is broken" from "CALCE's
4 cells are too few" -- every other comparison so far spans two datasets at once.

Their setup, taken from the paper:
  SoH(t) = Qmax(t) / Qnom, Qnom = 2.5 Ah          (Eq. 1)
  EOL threshold = 70% SoH                          ("we consider the RUL threshold to be 70%")
  data below the threshold removed, first point below kept
  6-fold: one cell held out, the other 5 for train+val (80/20)
  target normalised to the 0-1 range               <- i.e. ABSOLUTE, not per-window z-score

Which capacity.  The dataset has ~700 charge CSVs and only ~45 discharge CSVs per
cell (a full capacity test every 20-30 cycles).  So the per-cycle series has to come
from CHARGE throughput -- and that is consistent with their own numbers: their AR has
RUL error 21 at cycle 450, i.e. an EOL near cycle 471, which matches the charge-based
crossing (430-494) and not the discharge-based one (535-640).

Read-only.  Writes nothing.

    D:/anaconda/envs/py312/python.exe src/polimi_tub.py --check
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np

ROOT = r"D:\research\degradation_prognostics\polimi_tub\extracted"
QNOM = 2.5              # Ah, nominal -- Bellomo Eq. 1
EOL_SOH = 0.70          # their RUL threshold
HOLD = 3                # cycles the series must stay below EOL to count (charge
                        # throughput is noisy; one bad cycle is not end of life)


def load_cells(root: str = ROOT, kind: str = "chg"):
    """{cell: (cycles, capacity_Ah)} from the per-cycle CSVs, sorted by cycle."""
    out = {}
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if not os.path.isdir(p):
            continue
        per = {}
        for f in glob.glob(os.path.join(p, f"Cycle*_{kind}.csv")):
            m = re.match(r"Cycle(\d+)_" + kind + r"\.csv$", os.path.basename(f))
            if not m:
                continue
            try:
                with open(f) as fh:
                    head = fh.readline().strip().split(",")
                    col = head.index("capacity_Ah")
                    mx = max(float(ln.split(",")[col])
                             for ln in fh if ln.strip())
            except Exception:
                continue
            per[int(m.group(1))] = mx
        if per:
            ks = np.array(sorted(per), dtype=np.int64)
            out[d] = (ks, np.array([per[k] for k in ks], dtype=np.float64))
    return out


def eol_cycle(cycles, caps, soh_thr=EOL_SOH, hold=HOLD):
    """First cycle from which the cell stays below the EOL threshold."""
    under = caps < QNOM * soh_thr
    for i in range(len(under) - hold + 1):
        if under[i:i + hold].all():
            return int(cycles[i])
    return -1


def soh_series(root: str = ROOT):
    """{cell: (cycles 1..N, SoH in [0,1])} -- per cycle.

    RECONSTRUCTION, not the original series.  The archive holds only ~44 discharge
    (capacity-test) CSVs per cell, spaced ~21 cycles, but Bellomo's Figure 7 plots
    smooth per-cycle SoH curves across 0-750 cycles starting at 100%.  So the
    dataset's own SoH is per-cycle and those 44 points are its anchors; we
    reconstruct the curve by linear interpolation between them.  Recorded here so
    this is never mistaken for raw data.

    The charge-throughput series is NOT used: measured on the same cells it starts
    at 1.75 Ah = exactly the 70% threshold and swings between 1.0 and 2.35 Ah cycle
    to cycle, because the UDDS/US06/LA92/Mixed driving cycles differ in depth of
    discharge.  It tracks the duty cycle, not the ageing.
    """
    raw = load_cells(root, kind="dchg")
    out = {}
    for c, (ks, q) in raw.items():
        n = int(ks.max())
        grid = np.arange(1, n + 1)
        out[c] = (grid, np.interp(grid, ks, q) / QNOM)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--kind", default="chg", choices=["chg", "dchg"])
    args = ap.parse_args()

    cells = load_cells(kind=args.kind)
    print("=" * 92)
    print(f"  PoliMi-TUB  ({args.kind} throughput)   Qnom={QNOM} Ah   "
          f"EOL = {EOL_SOH:.0%} SoH = {QNOM * EOL_SOH:.3f} Ah")
    print("=" * 92)
    print("  %-8s %6s %6s %9s %9s %9s %9s   %s"
          % ("cell", "n", "first", "cap@1st", "cap_max", "cap_last", "EOL_cyc",
             "SoH at end"))
    for c, (ks, q) in cells.items():
        eol = eol_cycle(ks, q)
        print("  %-8s %6d %6d %9.4f %9.4f %9.4f %9s   %.3f"
              % (c, len(ks), ks[0], q[0], q.max(), q[-1],
                 eol if eol > 0 else "none", q[-1] / QNOM))
    print()
    print("  Bellomo's AR has RUL error 21 at cycle 450 -> their EOL is near 471.")
    print("  The charge-based EOL above should land in that neighbourhood; the")
    print("  discharge-based one (a capacity test every 20-30 cycles) does not.")


if __name__ == "__main__":
    main()
