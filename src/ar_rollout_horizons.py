"""Per-horizon error of our own model's AR rollout (CALCE, seed 1).

test_ar_rollout.py reports the rollout MAE over the whole span.  The
reviewer item that motivated this measurement asks for multi-step error at
k = 25/50/100, so this re-uses that script's loader/rollout verbatim
(imported, not copied) and adds the per-horizon numbers.

Run with py312:
    python src/ar_rollout_horizons.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_ar_rollout import (  # noqa: E402
    EPS, first_crossing, load_caps, load_ckpt, rollout,
)

HORIZONS = (25, 50, 100)


def main() -> None:
    caps, test_cell, sps = load_caps("calce")
    print(f"=== calce  test={test_cell}  seed=1 (ours, via test_ar_rollout) ===")
    print(f"  {'SP':>5}" + "".join(f" {'MAE@'+str(k):>10}" for k in HORIZONS)
          + f" {'MAE all':>9} {'R2':>9}")
    for sp in sps:
        model, ck = load_ckpt("calce", sp, 1)
        lo, hi, W, eol_ah = ck["lo"], ck["hi"], ck["W"], ck["eol_ah"]
        seq = (caps[test_cell] - lo) / (hi - lo + EPS)
        thr = (eol_ah - lo) / (hi - lo + EPS)

        pv = rollout(model, seq, sp, W, len(seq) - sp)
        tv = seq[sp:sp + len(pv)]
        n = min(len(pv), len(tv))
        pv, tv = pv[:n], tv[:n]
        r2 = float(1 - np.sum((tv - pv) ** 2)
                   / (np.sum((tv - tv.mean()) ** 2) + EPS))
        cells = "".join(f" {np.mean(np.abs(pv[:k] - tv[:k])):>10.4f}"
                        for k in HORIZONS if n >= k)
        print(f"  {sp:>5}{cells} {np.mean(np.abs(pv - tv)):>9.4f} {r2:>9.4f}"
              f"   (true crosses at {first_crossing(tv, thr) + 1})")


if __name__ == "__main__":
    main()
