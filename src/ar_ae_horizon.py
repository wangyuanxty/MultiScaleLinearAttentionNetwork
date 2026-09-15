"""AE versus prediction horizon on CALCE -- how far ahead can each model
call the end of life?

Table A measures the *teacher-forced* trajectory: every input window is
sliced out of the ground-truth capacity series, so the reported AE is where
a one-step predictor's output happens to cross the threshold, not a
forecast.  This script measures the operational quantity instead: from a
starting point T the model is fed only its own predictions, and the error
that matters is the error in the *crossing cycle* -- i.e. how much warning
the model can give.

Protocol (per T, per checkpoint run):

  1. warm up on the W TRUE cycles immediately before T;
  2. roll forward one step at a time, feeding only the model's own output,
     until the true EOL cycle (the rollout is additionally continued to the
     end of the true series so that a *late* crossing can be told apart from
     a frozen trajectory -- reported separately, never as the primary AE);
  3. record MAE / R2 over [T, true_EOL], the predicted crossing cycle, and
     AE = |true crossing cycle - predicted crossing cycle|.

Model selection: the nearest per-SP checkpoint whose SP is <= T.  Only
SP300 / SP400 / SP500 exist, so T = 300/400/500 use SP300/400/500 and
T = 540/580/600 all use SP500.

The decode is the delicate part and is NOT re-implemented here:

  * ours      -- per-window z-score denormalisation, verbatim from
                 src/test_ar_rollout.py (rollout) and
                 src/eval_per_sp_existing.py (batched teacher-forced pass);
  * baselines -- pytorch_forecasting EncoderNormalizer over a
                 StandardScaler'd known real, verbatim from
                 src/ar_rollout_baselines.py (scale_series / decode /
                 net_out / build_frames / teacher_forced).

Run with the two environments the sources require:

    D:/anaconda/envs/py312/python.exe    src/ar_ae_horizon.py --model ours
    python src/ar_ae_horizon.py --model pf --out ...   # patchformer env
    python src/ar_ae_horizon.py --model rm --out ...   # patchformer env
    python src/ar_ae_horizon.py --merge
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

EPS = 1e-6
EOL_AH = 0.77
T_STARTS = (300, 400, 500, 540, 580, 600)
CKPT_SPS = (300, 400, 500)
DEFAULT_RUNS = (1, 2, 3, 4, 5)

PARTIAL = {m: os.path.join(SRC, "results", f"_ar_ae_horizon_{m}.json")
           for m in ("ours", "pf", "rm")}
MERGED = os.path.join(SRC, "results", "ar_ae_horizon.json")


def sp_for(t: int) -> int:
    """Nearest per-SP checkpoint whose SP is <= T."""
    return max(s for s in CKPT_SPS if s <= t)


def first_crossing(series: np.ndarray, thr: float) -> int:
    """First downward crossing index, or -1.  Same convention as
    test_ar_rollout.first_crossing / ar_rollout_baselines.first_crossing."""
    for i in range(len(series) - 1):
        if series[i] >= thr > series[i + 1]:
            return i
    return -1


def crossing_cycle(series: np.ndarray, thr: float, t: int):
    """Cycle number (1-based, absolute) of the first downward crossing of a
    rollout that started at cycle `t`, or None if it never crosses."""
    i = first_crossing(series, thr)
    return None if i < 0 else int(t + i + 1)


# ── metrics ─────────────────────────────────────────────────────────────
def window_metrics(pred_ah: np.ndarray, true_ah: np.ndarray):
    mae = float(np.mean(np.abs(pred_ah - true_ah)))
    denom = float(np.sum((true_ah - true_ah.mean()) ** 2))
    r2 = float(1.0 - np.sum((true_ah - pred_ah) ** 2) / (denom + EPS))
    return mae, r2


def collapse(vals):
    """mean/std over the runs that produced a number; None-safe."""
    a = np.asarray([v for v in vals if v is not None], dtype=float)
    if a.size == 0:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": float(a.mean()),
            "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "n": int(a.size)}


# ── ours (py312) ────────────────────────────────────────────────────────
def tf_ours(model, seq, t, w):
    """Batched teacher-forced one-step pass over [t, end].

    Identical math to eval_per_sp_existing.eval_sp_batched (windows stacked
    out of the ground-truth series, per-window z-score denormalisation), so
    this reproduces Table A's MAE when t == SP.
    """
    import torch

    dev = next(model.parameters()).device
    idx = np.arange(t, len(seq))
    x = np.stack([seq[i - w:i, None] for i in idx]).astype(np.float32)
    wmean = x[:, :, 0].mean(axis=1)
    wstd = x[:, :, 0].std(axis=1) + EPS
    with torch.no_grad():
        pred = model(torch.tensor(x, device=dev)).squeeze(-1).cpu().numpy()
    return pred * wstd + wmean


def run_ours(ts, runs, steps_cap):
    from test_ar_rollout import load_caps, load_ckpt, rollout as ar_rollout

    caps, test_cell, _ = load_caps("calce")
    out = {"series": test_cell, "points": {}, "verification": {},
           "first_step": {}}

    # -- verification: teacher-forced MAE at SP300 over seeds -------------
    sp = 300
    mae_seeds = []
    for seed in range(1, 11):
        try:
            model, ck = load_ckpt("calce", sp, seed)
        except FileNotFoundError:
            continue
        lo, hi, w = ck["lo"], ck["hi"], ck["W"]
        seq = (caps[test_cell] - lo) / (hi - lo + EPS)
        pred = tf_ours(model, seq, sp, w)
        mae_seeds.append(float(np.mean(np.abs(pred - seq[sp:]))))
        del model
    out["verification"] = {
        "sp": sp, "space": "checkpoint min-max (Table A)",
        "mae_seed1": mae_seeds[0] if mae_seeds else None,
        "mae_mean_10seed": float(np.mean(mae_seeds)) if mae_seeds else None,
        "expected": "seed1 ~0.0073, 10-seed mean ~0.0066",
        "per_seed_mae": mae_seeds,
    }

    for t in ts:
        sp = sp_for(t)
        per_run = []
        for seed in runs:
            model, ck = load_ckpt("calce", sp, seed)
            lo, hi, w, eol_ah = ck["lo"], ck["hi"], ck["W"], ck["eol_ah"]
            seq = (caps[test_cell] - lo) / (hi - lo + EPS)
            thr = (eol_ah - lo) / (hi - lo + EPS)

            tf = tf_ours(model, seq, t, w)          # teacher-forced, [t, end]
            n = len(seq) - t
            steps = steps_cap if steps_cap else n
            steps = min(steps, n)
            pv = ar_rollout(model, seq, t, w, steps)  # AR, [t, t+steps)
            del model

            seg = seq[t:t + steps]
            true_ah = seg * (hi - lo) + lo
            pred_ah = np.asarray(pv) * (hi - lo) + lo
            tf_ah = tf * (hi - lo) + lo

            # true EOL -- a property of the series, identical for every run
            true_rel = first_crossing(seg, thr)
            assert true_rel >= 0, f"true series never crosses from T={t}"
            true_cycle = t + true_rel + 1
            n_proto = true_rel + 2          # include the sample after the crossing

            # protocol window: rollout stopped at the true EOL cycle
            mae, r2 = window_metrics(pred_ah[:n_proto], true_ah[:n_proto])
            proto_pred = crossing_cycle(pv[:n_proto], thr, t)
            proto_ae = (abs(true_cycle - proto_pred)
                        if proto_pred is not None else None)
            # diagnostic only: same rollout continued to the end of the series
            full_pred = crossing_cycle(pv, thr, t)
            full_ae = (abs(true_cycle - full_pred)
                       if full_pred is not None else None)

            per_run.append({
                "seed": seed, "ckpt_sp": sp, "checkpoint": f"SP{sp}_seed{seed}",
                "horizon": int(true_rel + 1),
                "mae_ah": mae, "r2": r2,
                "true_crossing": int(true_cycle),
                "pred_crossing": proto_pred,
                "pred_crossing_full": full_pred,
                "ae": proto_ae, "ae_full": full_ae,
                "first_step_dev": float(abs(pv[0] - tf[0])),
                "first_step_rollout_ah": float(pred_ah[0]),
                "first_step_tf_ah": float(tf_ah[0]),
                "end_window_std": (float(np.std(pv[-w:]))
                                   if len(pv) >= w else None),
            })
            out["first_step"][f"T{t}_seed{seed}"] = per_run[-1]["first_step_dev"]
            print(f"  ours T={t} (SP{sp}) seed{seed}: horizon={true_rel + 1} "
                  f"MAE={mae:.4f} R2={r2:.4f} true={true_cycle} "
                  f"pred={proto_pred} AE={proto_ae} "
                  f"(full pred={full_pred} AE_full={full_ae}) "
                  f"step1dev={per_run[-1]['first_step_dev']:.2e}", flush=True)

        out["points"][str(t)] = _point(t, sp_for(t), per_run)
    return out


# ── baselines (patchformer env) ─────────────────────────────────────────
def rollout_baseline(a, model, df_all, minv, maxv, t, sc_m, sc_s, name,
                     steps_cap):
    """ar_rollout_baselines.rollout, started at cycle `t` instead of SP.

    Same warm-up (the SEQL true cycles before the start point), same
    per-window EncoderNormalizer decode, same one-cycle slide.
    """
    cap_mm = df_all["Capacity"].values.astype(np.float64)
    cyc = df_all["Cycle"].values
    i_t = int(np.where(cyc == t)[0][0])
    window = list(cap_mm[i_t - a.SEQL:i_t])
    if len(window) < a.SEQL:
        raise ValueError(f"T={t}: only {len(window)} warm-up cycles")

    n = len(cap_mm) - i_t
    steps = min(steps_cap, n) if steps_cap else n
    preds = []
    for _ in range(steps):
        x = np.asarray(window[-a.SEQL:], dtype=np.float64)
        x_sc = a.scale_series(x, sc_m, sc_s).astype(np.float32)
        z = a.net_out(model, x_sc.reshape(1, a.SEQL, 1), name)
        p_rated = a.decode(z, x_sc.reshape(1, a.SEQL), minv, maxv,
                           sc_m, sc_s)[0]
        p_mm = (p_rated - minv) / (maxv - minv)
        preds.append(p_mm)
        window.append(p_mm)
    return np.asarray(preds), i_t


def run_baseline(tag, ts, runs, steps_cap):
    import ar_rollout_baselines as a
    import torch

    series = a.official_calce() if tag == "pf" else a.ours_calce()
    loader = a.load_pf if tag == "pf" else a.load_rm
    out = {"series": a.TEST_CELL, "points": {}, "verification": {},
           "first_step": {}}

    # -- verification: teacher-forced MAE / R2 at SP300, runs 1..5 ---------
    ver = []
    for run in runs:
        model, ckpt = loader(300, run)
        os.chdir(ROOT)
        df_train, df_test, df_all, minv, maxv = a.build_frames(series, 300)
        i_sp = int(np.where(df_all["Cycle"].values == 300)[0][0])
        y_true, y_pred, man, off, diag = a.teacher_forced(
            model, df_all, minv, maxv, 300, i_sp, tag)
        mae = float(np.mean(np.abs(y_true - y_pred)))
        r2 = float(1 - np.sum((y_true - y_pred) ** 2)
                   / (np.sum((y_true - y_true.mean()) ** 2) + a.EPS))
        ver.append({"run": run, "mae_ah": mae, "r2": r2,
                    "decode_err": diag["max_decode_err"],
                    "x_is_true_capacity": diag["x_is_true_capacity"],
                    "scaler_mean": diag["scaler_mean"],
                    "scaler_scale": diag["scaler_scale"]})
        print(f"  [verify] {tag} SP300 run{run}: MAE={mae:.4f} R2={r2:.4f} "
              f"decode_err={diag['max_decode_err']:.2e}", flush=True)
        del model
    out["verification"] = {
        "sp": 300, "space": "Ah", "per_run": ver,
        "expected": ("~0.0058 / R2~0.9962" if tag == "pf"
                     else "~0.0175 / R2~0.9799"),
    }

    for t in ts:
        sp = sp_for(t)
        per_run = []
        for run in runs:
            model, ckpt = loader(sp, run)
            os.chdir(ROOT)
            df_train, df_test, df_all, minv, maxv = a.build_frames(series, sp)
            i_sp = int(np.where(df_all["Cycle"].values == sp)[0][0])
            # teacher-forced trajectory over [sp, end] -- gives the scaler and
            # the ground-truth first step at T as well
            y_true, y_pred, man, off, diag = a.teacher_forced(
                model, df_all, minv, maxv, sp, i_sp, tag)
            sc_m, sc_s = diag["scaler_mean"], diag["scaler_scale"]

            tf_ah_all = y_pred * a.RATED          # official predict(), Ah
            cap_ah = df_all["Capacity"].values * (maxv - minv) * a.RATED \
                + minv * a.RATED
            i_t = int(np.where(df_all["Cycle"].values == t)[0][0])

            pv, _ = rollout_baseline(a, model, df_all, minv, maxv, t,
                                     sc_m, sc_s, tag, steps_cap)
            del model

            pv_ah = pv * (maxv - minv) * a.RATED + minv * a.RATED
            seg_ah = cap_ah[i_t:i_t + len(pv)]
            steps = len(pv)

            thr = (a.EOL_AH / a.RATED - minv) / (maxv - minv)
            seg_mm = df_all["Capacity"].values[i_t:i_t + steps]
            true_rel = a.first_crossing(seg_mm, thr)
            assert true_rel >= 0, f"true series never crosses from T={t}"
            true_cycle = t + true_rel + 1
            n_proto = true_rel + 2

            mae, r2 = window_metrics(pv_ah[:n_proto], seg_ah[:n_proto])
            proto_pred = crossing_cycle(pv[:n_proto], thr, t)
            proto_ae = (abs(true_cycle - proto_pred)
                        if proto_pred is not None else None)
            full_pred = crossing_cycle(pv, thr, t)
            full_ae = (abs(true_cycle - full_pred)
                       if full_pred is not None else None)

            # first rollout step vs the teacher-forced step at T.  The
            # teacher-forced trajectory starts at `sp`; T sits at offset
            # T - sp inside the ground-truth-windowed pass.
            k = t - sp
            first_tf_ah = float(tf_ah_all[k]) if k < len(tf_ah_all) else None
            first_dev = (abs(float(pv_ah[0]) - first_tf_ah)
                         if first_tf_ah is not None else None)

            per_run.append({
                "run": run, "ckpt_sp": sp,
                "checkpoint": os.path.basename(ckpt),
                "horizon": int(true_rel + 1),
                "mae_ah": mae, "r2": r2,
                "true_crossing": int(true_cycle),
                "pred_crossing": proto_pred,
                "pred_crossing_full": full_pred,
                "ae": proto_ae, "ae_full": full_ae,
                "first_step_dev": first_dev,
                "first_step_rollout_ah": float(pv_ah[0]),
                "first_step_tf_ah": first_tf_ah,
                "end_window_std": (float(np.std(pv[-a.SEQL:]))
                                   if len(pv) >= a.SEQL else None),
            })
            out["first_step"][f"T{t}_run{run}"] = first_dev
            print(f"  {tag} T={t} (SP{sp}) run{run}: horizon={true_rel + 1} "
                  f"MAE={mae:.4f} R2={r2:.4f} true={true_cycle} "
                  f"pred={proto_pred} AE={proto_ae} "
                  f"(full pred={full_pred} AE_full={full_ae}) "
                  f"step1dev={first_dev}", flush=True)

        out["points"][str(t)] = _point(t, sp_for(t), per_run)

    import torch as _t
    _t.cuda.empty_cache()
    return out


# ── aggregation ─────────────────────────────────────────────────────────
def _point(t, sp, per_run):
    return {
        "T": int(t), "ckpt_sp": int(sp),
        "horizon": int(per_run[0]["horizon"]),
        "n_runs": len(per_run),
        "n_crossed": int(sum(r["ae"] is not None for r in per_run)),
        "n_crossed_full": int(sum(r["ae_full"] is not None for r in per_run)),
        "per_run": per_run,
        "mae_ah": collapse([r["mae_ah"] for r in per_run]),
        "r2": collapse([r["r2"] for r in per_run]),
        "pred_crossing": collapse([r["pred_crossing"] for r in per_run]),
        "ae": collapse([r["ae"] for r in per_run]),
        "pred_crossing_full": collapse([r["pred_crossing_full"]
                                        for r in per_run]),
        "ae_full": collapse([r["ae_full"] for r in per_run]),
        "first_step_dev_max": float(np.max([r["first_step_dev"]
                                            for r in per_run
                                            if r["first_step_dev"] is not None])),
    }


def merge():
    merged = {"protocol": {
        "dataset": "CALCE", "test_cell": "CS2_35",
        "train_cells": ["CS2_36", "CS2_37", "CS2_38"],
        "W": 64, "eol_ah": EOL_AH,
        "normalization": "global min-max from train cells",
        "model_selection": "nearest per-SP checkpoint with SP <= T",
        "runs": "seeds/runs 1..5", "curves": {}}}
    for tag in ("ours", "pf", "rm"):
        with open(PARTIAL[tag], encoding="utf-8") as f:
            merged["curves"][tag] = json.load(f)
    with open(MERGED, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    print(f"merged -> {MERGED}")
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ours", choices=["ours", "pf", "rm",
                                                        "merge"])
    ap.add_argument("--Ts", type=int, nargs="+", default=list(T_STARTS))
    ap.add_argument("--runs", type=int, nargs="+", default=list(DEFAULT_RUNS))
    ap.add_argument("--steps-cap", type=int, default=0,
                    help="0 = roll to the end of the true series")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.model == "merge":
        merge()
        return

    print(f"=== {args.model}  Ts={args.Ts}  runs={args.runs} ===")
    if args.model == "ours":
        res = run_ours(args.Ts, args.runs, args.steps_cap)
    else:
        res = run_baseline(args.model, args.Ts, args.runs, args.steps_cap)

    res["model"] = args.model
    path = args.out or PARTIAL[args.model]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
