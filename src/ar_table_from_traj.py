"""Recompute the AR rollout table directly from the saved trajectories.

Why from trajectories: the PANASONIC run wrote all three models to the same
JSON path in turn, so the metric file only kept the last writer. The `.npz`
trajectories were written per model and are intact, and every metric in the
table is a function of the predicted curve and the true curve, so they can be
recovered exactly.

Conventions, identical for both datasets
----------------------------------------
launch      k cycles before the true EOL crossing
MAE         trajectory error over the first k steps after the launch, in
            normalised capacity (paper convention), reported mean +- std
            over the 10 seeds
AE          predicted_crossing_cycle - true_EOL, signed, in cycles.  A run
            whose predicted curve never falls below the threshold has no AE:
            it is excluded from the AE mean and counted in the parentheses.

    python src/ar_table_from_traj.py
"""
import os

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(SRC, "results")

KS = (10, 20, 25, 30, 40, 50, 75, 100)
MODELS = ("ours", "pf", "rm")

# dataset -> threshold, true EOL, normalisation range, trajectory files
DS = {
    "CALCE": dict(thr=0.77, eol=640,
                  lo=0.1421641707420349, hi=1.134451150894165,
                  files=["ar_traj_calce.npz"]),
    "PANASONIC": dict(thr=2.12, eol=588, lo=1.7591, hi=2.7784,
                      files=[f"_ar_traj_panasonic_{m}.npz" for m in MODELS]),
}


def first_crossing(series, thr):
    """First downward crossing index within `series`, or -1."""
    for i in range(len(series) - 1):
        if series[i] >= thr > series[i + 1]:
            return i
    return -1


def run_one(ds_name, cfg):
    arrays = {}
    for fn in cfg["files"]:
        path = os.path.join(RES, fn)
        if not os.path.exists(path):
            print(f"  [{ds_name}] missing {fn}")
            continue
        with np.load(path) as z:
            for k in z.files:
                arrays.setdefault(k, z[k])

    if "true" not in arrays or "cycles" not in arrays:
        print(f"  [{ds_name}] no true/cycles -- skipping")
        return None

    true = arrays["true"].astype(float)
    cycles = arrays["cycles"].astype(int)
    span = cfg["hi"] - cfg["lo"]
    thr, eol = cfg["thr"], cfg["eol"]

    out = {}
    for m in MODELS:
        for k in KS:
            maes, aes, signed, n_never = [], [], [], 0
            for s in range(1, 11):
                key = f"pred_{m}_k{k}_s{s}"
                if key not in arrays:
                    continue
                p = arrays[key].astype(float)
                valid = np.where(~np.isnan(p))[0]
                if len(valid) == 0:
                    continue
                i0 = int(valid[0])
                seg, tseg = p[i0:i0 + k], true[i0:i0 + k]
                n = min(len(seg), len(tseg))
                if n == 0:
                    continue
                maes.append(float(np.mean(np.abs(seg[:n] - tseg[:n])) / span))

                xi = first_crossing(p[i0:], thr)
                if xi < 0:
                    n_never += 1
                else:
                    # AE = |TRUL - PRUL| (04_experiments.tex:47).  The signed
                    # value is kept separately as a diagnostic -- it says
                    # whether the model calls EOL early or late -- but it is
                    # not the reported metric.
                    d = int(cycles[i0 + xi]) - eol
                    aes.append(abs(d))
                    signed.append(d)
            out[(m, k)] = dict(
                mae_mean=float(np.mean(maes)) if maes else float("nan"),
                mae_std=float(np.std(maes, ddof=1)) if len(maes) > 1 else 0.0,
                n_mae=len(maes),
                ae_mean=float(np.mean(aes)) if aes else float("nan"),
                ae_std=float(np.std(aes, ddof=1)) if len(aes) > 1 else 0.0,
                n_cross=len(aes), n_never=n_never,
                signed=sorted(signed),
            )
    return out


def fmt_ae(r):
    """|AE| mean +- std, with the crossing count and the sign direction.

    AE is the absolute RUL error; the direction marker is the diagnostic that
    says whether the model called EOL early (negative) or late (positive).
    """
    tot = r["n_cross"] + r["n_never"]
    if r["n_cross"] == 0:
        return f"--  (0/{tot})"
    sg = r.get("signed") or []
    if sg and all(v > 0 for v in sg):
        d = "late"
    elif sg and all(v < 0 for v in sg):
        d = "early"
    else:
        d = "mixed"
    if r["n_cross"] == 1:
        return f"{r['ae_mean']:.0f} {d} (1/{tot})"
    return f"{r['ae_mean']:.0f}+/-{r['ae_std']:.0f} {d} ({r['n_cross']}/{tot})"


def main():
    for name, cfg in DS.items():
        res = run_one(name, cfg)
        if not res:
            continue
        print()
        print("=" * 94)
        print(f"  {name}   (true EOL = {cfg['eol']}, threshold {cfg['thr']} Ah)")
        print("=" * 94)
        hdr = (f"{'k':>4} | {'ours MAE':>16} {'PF MAE':>16} {'RM MAE':>16} | "
               f"{'ours AE':>18} {'PF AE':>18} {'RM AE':>18}")
        print(hdr)
        print("-" * len(hdr))
        for k in KS:
            cells = []
            for m in MODELS:
                r = res.get((m, k))
                if r is None or np.isnan(r["mae_mean"]):
                    cells.append(("--", "--"))
                else:
                    cells.append((f"{r['mae_mean']:.5f}+/-{r['mae_std']:.5f}",
                                  fmt_ae(r)))
            print(f"{k:>4} | {cells[0][0]:>16} {cells[1][0]:>16} "
                  f"{cells[2][0]:>16} | {cells[0][1]:>18} {cells[1][1]:>18} "
                  f"{cells[2][1]:>18}")
        print()
        print("  MAE normalised capacity (mean +/- std over seeds) | "
              "AE = pred crossing - true EOL, cycles (crossed/total)")


if __name__ == "__main__":
    main()
