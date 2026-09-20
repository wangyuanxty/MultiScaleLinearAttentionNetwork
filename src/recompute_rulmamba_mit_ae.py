"""Recompute RUL-Mamba's MIT metrics from its saved Prediction.npy arrays.

Their own log reports RUL_pred=1 for every MIT repeat, so AE=308/208/108 is
the sentinel value of a crossing that was never detected, not a measurement.
The per-repeat prediction arrays are on disk, so AE is recomputed here under
the paper's convention: per-dataset EOL threshold (MIT = 1.074 Ah x 80% =
0.8592 Ah, the same 0.86 the dataset table lists) and the first-crossing
index rule used by eval_multiseed.true_rul.

MAE and R2 are also recomputed so the log's numbers can be checked rather
than trusted; both the raw-Ah and the min-max-normalized variants are
printed, and whichever matches the log identifies the convention in use.

Reads only. Writes nothing.

Usage: cd src && python recompute_rulmamba_mit_ae.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_multiseed import true_rul          # noqa: E402

BASE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "reference_repos", "ref_rul_mamba", "Outputs", "MIT", "Univariable", "RULMamba",
)
SPS = (200, 300, 400)
REPEATS = range(1, 11)
EOL_AH = 1.074 * 0.8          # 0.8592 Ah, the paper's MIT EOL threshold
TH = 0.86


def metrics(pred, actual, lo, hi):
    """MAE/R2 in normalized capacity plus AE under the crossing rule."""
    pn = (pred - lo) / (hi - lo)
    an = (actual - lo) / (hi - lo)
    mae = float(np.mean(np.abs(pn - an)))
    r2 = 1 - np.sum((an - pn) ** 2) / np.sum((an - an.mean()) ** 2)
    ae = abs(true_rul(an, (TH - lo) / (hi - lo)) - true_rul(pn, (TH - lo) / (hi - lo)))
    return mae, float(r2), ae


def main() -> None:
    rd = np.load(os.path.join(BASE, "Real_Data.npy")).ravel()
    lo_all, hi_all = float(rd.min()), float(rd.max())
    print(f"MIT real-data range: {lo_all:.5f} .. {hi_all:.5f}   EOL threshold {TH} Ah "
          f"(= 1.074 x 0.8 = {EOL_AH:.4f})\n")

    log = {200: (0.0011, 0.9994), 300: (0.0016, 0.9991), 400: (0.0022, 0.9979)}
    print(f"{'SP':>5} {'metric':>9} | {'global-norm':>20} | {'per-seg-norm':>20} | {'log':>16}")
    for sp in SPS:
        g, s = [], []
        for rep in REPEATS:
            d = os.path.join(BASE, f"Repeat_{rep}", f"Start_Point_{sp}")
            P = np.load(os.path.join(d, "Prediction.npy")).ravel()
            A = np.load(os.path.join(d, "Actual.npy")).ravel()
            g.append(metrics(P, A, lo_all, hi_all))
            s.append(metrics(P, A, float(min(P.min(), A.min())), float(max(P.max(), A.max()))))
        g, s = np.array(g), np.array(s)
        lg = log[sp]
        print(f"{sp:>5} {'MAE':>9} | {g[:,0].mean():>20.5f} | {s[:,0].mean():>20.5f} | {lg[0]:>16.4f}")
        print(f"{sp:>5} {'R2':>9} | {g[:,1].mean():>20.4f} | {s[:,1].mean():>20.4f} | {lg[1]:>16.4f}")
        print(f"{sp:>5} {'AE':>9} | {g[:,2].mean():>20.2f} | {s[:,2].mean():>20.2f} | "
              f"{'308/208/108 <- broken':>16}")


if __name__ == "__main__":
    main()
