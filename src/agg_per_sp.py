"""Aggregate per-SP 10-seed protocol results into paper-style mean +/- std.

Reads results/per_sp_train.json (output of train_per_sp.py) and writes
results/per_sp_summary.json. Aggregation matches Section 4 main-results
convention: per SEED, average metrics across the test cells (each (SP, seed)
row is one test cell), then mean +/- sample std (ddof=1) over the 10 seeds;
AE reported as mean over seeds (as in the paper's Table A caption).

Usage:  python src/agg_per_sp.py [--datasets calce nasa panasonic tju]
Prints a markdown table; also prints the current Table A (old protocol)
numbers when --compare is given.
"""
import argparse
import json
import numpy as np

METE = ["TRUL", "PRUL", "AE", "MAE", "RMSE", "R2"]
METER = ["MAE", "RMSE", "R2"]

# Old-protocol Table A values (tab:tableA text of 2026-08-15) for comparison.
OLD_TABLE_A = {
    "calce": {300: (339, 0.0066, 0.0148, 0.9950, 2.7),
              400: (239, 0.0075, 0.0162, 0.9940, 2.7),
              500: (139, 0.0083, 0.0179, 0.9916, 2.7)},
    "nasa": {50: (73, 0.0078, 0.0139, 0.9914, 2.0),
             70: (53, 0.0087, 0.0152, 0.9812, 2.0),
             90: (33, 0.0075, 0.0117, 0.9799, 2.0)},
    "panasonic": {300: (287, 0.0037, 0.0098, 0.9976, 1.0),
                  500: (87, 0.0041, 0.0113, 0.9933, 1.0)},
    "tju": {200: (577, 0.0011, 0.0018, 0.9999, 0.3),
            300: (477, 0.0012, 0.0019, 0.9999, 0.3),
            400: (377, 0.0012, 0.0020, 0.9998, 0.3)},
}


def load():
    with open("src/results/per_sp_train.json", encoding="utf-8") as f:
        return json.load(f)


def aggregate(ds, sp, out):
    seeds = out[ds][str(sp)]
    per_seed = {}
    for skey, rows in seeds.items():
        if not rows:
            continue
        per_seed[int(skey)] = {
            m: float(np.mean([r[m] for r in rows])) for m in METE}
    assert len(per_seed) == 10, f"{ds} SP{sp}: {len(per_seed)} seeds"
    res = {"n_seeds": 10,
           "TRUL": float(np.mean([v["TRUL"] for v in per_seed.values()]))}
    for m in METER:
        vals = np.array([v[m] for v in per_seed.values()])
        res[m] = float(vals.mean())
        res[m + "_std"] = float(vals.std(ddof=1))
    res["AE"] = float(np.mean([v["AE"] for v in per_seed.values()]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["calce", "nasa", "panasonic", "tju"])
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    out = load()
    summary = {}
    for ds in args.datasets:
        summary[ds] = {}
        for s, sp in sorted(((int(s), s) for s in out[ds])):
            res = aggregate(ds, sp, out)
            summary[ds][str(s)] = res
            old = OLD_TABLE_A.get(ds, {}).get(s)
            cmp_str = ""
            if args.compare and old:
                cmp_str = (f" | OLD: TRUL={old[0]} MAE={old[1]:.4f} "
                           f"RMSE={old[2]:.4f} R2={old[3]:.4f} AE={old[4]}")
            print(f"| {ds} | {sp} | {res['TRUL']:.1f} | "
                  f"{res['MAE']:.4f}$\\pm${res['MAE_std']:.4f} | "
                  f"{res['RMSE']:.4f}$\\pm${res['RMSE_std']:.4f} | "
                  f"{res['R2']:.4f}$\\pm${res['R2_std']:.4f} | "
                  f"{res['AE']:.1f}{cmp_str}")
    with open("src/results/per_sp_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nwritten: src/results/per_sp_summary.json")


if __name__ == "__main__":
    main()
