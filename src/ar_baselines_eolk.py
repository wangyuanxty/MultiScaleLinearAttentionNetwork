"""PatchFormer and RUL-Mamba in the EOL-k launch matrix.

The AR section compares models at a launch point stated as `EOL - k`, so that
the horizon is explicit and identical across datasets and models.  The existing
baseline harness (ar_rollout_baselines.py) launches at SP instead, because there
the SP *is* the protocol.  This re-launches the SAME checkpoints at `EOL - k`
without touching that file -- so its SP-launch numbers stay exactly as the paper
recorded them.

Nothing about the baselines is re-implemented.  `load_pf` / `load_rm`,
`build_frames`, `teacher_forced` and `rollout` are imported verbatim:

  * `build_frames(series, sp)` still splits the training data at the SP -- that
    is what the checkpoint was trained for, and it must not move.
  * `rollout(..., sp=t0, ...)` only uses its `sp` argument to locate the launch
    index in the test series, so passing `t0 = EOL - k` relaunches there.
  * the scaler (`sc_m`, `sc_s`) and the min--max come from the teacher-forced
    pass at the same SP, unchanged.

Each baseline is still decoded in its OWN space -- PF with `official_calce()`
and PyTorch-Forecasting's decode, RM with `ours_calce()` and its own normalizer
-- so the conventions stay self-consistent per model, and the comparison across
models is a comparison of models, not of decoders.

Read-only w.r.t. checkpoints.  Writes only results/ar_baselines_eolk.json.

    conda run -n patchformer python src/ar_baselines_eolk.py --model pf
    conda run -n patchformer python src/ar_baselines_eolk.py --model rm
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import ar_rollout_baselines as B                          # noqa: E402

KS = (50, 100, 150, 200)
STEPS = 1000
OUT = os.path.join(SRC, "results", "ar_baselines_eolk.json")


def true_eol_cycle(cap_ah: np.ndarray, thr_ah: float) -> int:
    """First cycle at which capacity drops below the EOL threshold."""
    for i in range(len(cap_ah) - 1):
        if cap_ah[i] >= thr_ah > cap_ah[i + 1]:
            return i + 1
    return -1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["pf", "rm"])
    ap.add_argument("--sps", type=int, nargs="+", default=[300, 400, 500])
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--ks", type=int, nargs="+", default=list(KS))
    ap.add_argument("--steps", type=int, default=STEPS)
    args = ap.parse_args()

    name = args.model
    series = B.official_calce() if name == "pf" else B.ours_calce()
    loader = B.load_pf if name == "pf" else B.load_rm

    cap_cyc = series[B.TEST_CELL]
    eol = true_eol_cycle(cap_cyc[:, 1], B.EOL_AH)
    if eol <= 0:
        raise SystemExit("test cell never crosses the EOL threshold")
    print("=" * 104)
    print("  %s   test=%s   EOL=%d   rated=%.2f  EOL_AH=%.2f  SEQL=%d"
          % (name.upper(), B.TEST_CELL, eol, B.RATED, B.EOL_AH, B.SEQL))
    print("=" * 104, flush=True)

    rows = []
    for sp in args.sps:
        try:
            model, _ckpt = loader(sp, args.run)
        except (FileNotFoundError, IndexError, OSError) as e:
            print("  SP%d: NO CHECKPOINT (%s)" % (sp, e), flush=True)
            continue
        df_train, df_test, df_all, minv, maxv = B.build_frames(series, sp)
        i_sp = int(np.where(df_all["Cycle"].values == sp)[0][0])
        _yt, _yp, _man, _off, diag = B.teacher_forced(
            model, df_all, minv, maxv, sp, i_sp, name)
        sc_m, sc_s = diag["scaler_mean"], diag["scaler_scale"]
        thr = (B.EOL_AH / B.RATED - minv) / (maxv - minv)
        tv_all = df_all["Capacity"].values

        print("  --- SP%d (checkpoint)   launch points EOL-%s ---"
              % (sp, "/".join(str(k) for k in args.ks)), flush=True)
        print("  %-8s %7s %9s %9s %9s %8s %6s %11s %11s"
              % ("launch", "t0", "final", "AE", "AE/k", "crossed",
                 "steps", "MSE@50", "MSE"), flush=True)
        for k in args.ks:
            t0 = eol - k
            if t0 <= B.SEQL or t0 >= len(tv_all):
                print("  EOL-%-5d skipped (t0=%d outside series)" % (k, t0))
                continue
            try:
                preds, i_launch = B.rollout(model, df_all, minv, maxv, t0,
                                            sc_m, sc_s, max_steps=args.steps,
                                            name=name)
            except ValueError as e:
                print("  EOL-%-5d skipped (%s)" % (k, e))
                continue
            tv = tv_all[i_launch:i_launch + len(preds)]
            m = min(len(preds), len(tv))
            pv, tv = preds[:m], tv[:m]

            c_true = B.first_crossing(tv, thr)
            c_pred = B.first_crossing(pv, thr)
            if c_pred < 0:
                ae_val, ae_lbl = None, "none"
            else:
                # the crossing index is into the rollout, whose step 0 is cycle
                # t0 (1-based), so step i is cycle t0 + i + 1
                pred_cycle = t0 + c_pred + 1
                true_cycle = (t0 + c_true + 1) if c_true >= 0 else eol
                ae_val = float(abs(pred_cycle - true_cycle))
                ae_lbl = "%.0f" % ae_val
            n50 = min(50, m)
            mse50 = (float(np.mean((pv[:n50] - tv[:n50]) ** 2))
                     if n50 else float("nan"))
            mse = float(np.mean((pv - tv) ** 2)) if m else float("nan")
            print("  %-8s %7d %9.4f %9s %9s %8s %6d %11.3e %11.3e"
                  % ("EOL-%d" % k, t0, float(pv[-1]), ae_lbl,
                     ("%.2f" % (ae_val / k)) if ae_val is not None else "-",
                     "Y" if c_pred >= 0 else "n", len(pv), mse50, mse),
                  flush=True)
            rows.append(dict(sp=sp, run=args.run, k=k, t0=int(t0), eol=int(eol),
                             final=float(pv[-1]), ae=ae_val, ae_lbl=ae_lbl,
                             crossed=bool(c_pred >= 0), n=len(pv),
                             mse50=mse50, mse=mse))
        print(flush=True)

    prev = {}
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as fh:
            prev = json.load(fh)
    prev.setdefault(name, {})["run%d" % args.run] = rows
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(prev, fh, indent=2)
    print("  -> %s  (%s: %d rows)" % (OUT, name, len(rows)))


if __name__ == "__main__":
    main()
