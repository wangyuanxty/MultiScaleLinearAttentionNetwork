"""k-horizon AR (self-fed) table on CALCE CS2_35: how far before end of life
does each model's own rollout still call the crossing?

Protocol (the same shape as src/ar_ktable_panasonic.py, retargeted to CALCE):

  * true_EOL = 640 = CS2_35's first downward crossing of 0.77 Ah under the
    repo/paper convention (the crossing cycle is the last cycle AT/above the
    threshold; the first sub-threshold cycle is 641).  Both series in play --
    our `load_series` series and PatchFormer's official one -- cross at the
    same cycle, asserted below.
  * for each k: launch at cycle T = true_EOL - k, warm up on the W = 64 TRUE
    cycles before the launch (seq[T-W:T]) and then feed only the model's own
    predictions; the k-step MAE window is therefore cycles T+1..T+k, which
    ends exactly on the true EOL cycle.
  * MAE = trajectory MAE over those k steps, per run; mean +- std over the 10
    runs (ddof=1).
  * The rollout does NOT stop at k: it runs on to the end of the true series
    so the predicted crossing is the model's real call, not one truncated by
    the horizon.  AE = predicted_crossing_cycle - true_EOL, signed (negative =
    called it early, positive = late), the same convention on both sides, so a
    perfect forecast scores 0.  A run that never crosses by the end of the
    series contributes no AE and is counted in the footnote -- never imputed.
  * checkpoint = the nearest per-SP checkpoint with SP <= T.  Only SP
    300/400/500 exist, so k = 340/240/140 would land exactly on SP300/400/500
    (the protocol-internal points, `--ks 140 240 340`; not part of the
    reported grid) and every other k borrows a checkpoint from another
    launch point.  `sp_exact` / `protocol_internal` record which.

Environments (the two families need different torch builds):

    D:/anaconda/envs/py312/python.exe       src/ar_ktable_calce.py --model ours
    D:/anaconda/envs/patchformer/python.exe src/ar_ktable_calce.py --model pf
    D:/anaconda/envs/patchformer/python.exe src/ar_ktable_calce.py --model rm
    python src/ar_ktable_calce.py --model report

Machinery is IMPORTED, not rewritten (CLAUDE.md):

  ours      -- src/test_ar_rollout.py : load_ckpt / rollout (per-window
               z-score decode), + src/eval_per_sp_existing.eval_sp_batched for
               the teacher-forced check against Table A's CALCE row.
  baselines -- src/ar_rollout_baselines.py : build_frames / teacher_forced /
               rollout / net_out / decode / scale_series.

One phase detail worth stating because the two families count from different
places: `test_ar_rollout.rollout(model, seq, T, ...)` predicts the value at
index T (= cycle T+1), while `ar_rollout_baselines.rollout(..., start, ...)`
predicts the value AT cycle `start`.  The baselines are therefore launched at
`start = T + 1` so that all three models forecast the same cycles with the
same warm-up (the W true cycles ending at the launch cycle).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

EPS = 1e-6
TEST_CELL = "CS2_35"
TRAIN_CELLS = ("CS2_36", "CS2_37", "CS2_38")
RATED = 1.1
EOL_AH = 0.77
W_DEFAULT = 64
CKPT_SPS = (300, 400, 500)
KS = (25, 50, 75, 100)                 # the reported table grid
SUPP_KS = ()                           # superseded grids, kept runnable
LEGACY_KS = ()
ALL_KS = tuple(sorted(set(KS) | set(SUPP_KS) | set(LEGACY_KS)))
# Marker only (pass --ks 140 240 340 to get them): T = true_EOL - k lands
# exactly on SP500/SP400/SP300, i.e. the checkpoint trained for that launch.
PROTOCOL_INTERNAL_KS = frozenset({140, 240, 340})
RUNS = tuple(range(1, 11))
STD_DDOF = 1
# Same window, same model, same decode -> equality to float32 roundoff.  Seen
# values are ~1e-7; 1e-5 catches a warm-up / denormalisation error (those show
# up at 1e-2 and above), not float noise.
FIRST_STEP_TOL = 1e-5
REF_TF_REL_TOL = 0.02

OUT_JSON = os.path.join(SRC, "results", "ar_ktable_calce.json")
TRAJ_NPZ = os.path.join(SRC, "results", "ar_traj_calce.npz")

# Teacher-forced reference each family must still reproduce (paper Table A,
# CALCE CS2_35), quoted in the space each reference lives in.
REF_TF = {
    "ours": {"mae": 0.0066, "r2": 0.9951, "space": "common (checkpoint min-max)",
             "stat": "10-seed mean", "seed1_mae": 0.0073},
    "pf": {"mae": 0.0058, "r2": 0.9962, "space": "Ah", "stat": "run 1"},
    "rm": {"mae": 0.0175, "r2": 0.9799, "space": "Ah", "stat": "run 1"},
}


# ── shared helpers ──────────────────────────────────────────────────────
def stats(vals) -> dict:
    a = np.asarray([v for v in vals if v is not None], dtype=np.float64)
    if a.size == 0:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": float(a.mean()),
            "std": float(a.std(ddof=STD_DDOF)) if a.size > 1 else 0.0,
            "n": int(a.size)}


def first_crossing(series: np.ndarray, thr: float) -> int:
    """First downward crossing index, or -1 (test_ar_rollout convention)."""
    series = np.asarray(series, dtype=np.float64)
    for i in range(len(series) - 1):
        if series[i] >= thr > series[i + 1]:
            return i
    return -1


def crossing_cycle(series: np.ndarray, thr: float, first_cycle: int):
    """Cycle (1-based, absolute) of the LAST value at/above `thr` before the
    first downward crossing, for a rollout whose element 0 is the value at
    cycle `first_cycle`; None if the rollout never goes under the threshold.

    That is the repo's convention everywhere else (Table A's AE,
    ar_ae_horizon, ar_rollout_baselines.first_crossing + 1): the crossing
    cycle is the last healthy cycle, not the first dead one (which is +1).
    """
    i = first_crossing(series, thr)
    return None if i < 0 else int(first_cycle + i)


# When set, every launch point uses this SP's checkpoint instead of the
# nearest one with SP <= T.  Used to check whether the conclusions depend on
# which per-SP model is deployed; SP300 is valid for every launch point here
# (its cutoff, cycle 300, precedes all of them).
FORCE_SP: int | None = None


def sp_for(t: int) -> tuple[int, bool]:
    """(checkpoint SP, exact match).  Nearest SP <= T; if T precedes every
    trained SP the smallest is used (never happens on this grid)."""
    if FORCE_SP is not None:
        return FORCE_SP, FORCE_SP == t
    lower = [s for s in CKPT_SPS if s <= t]
    if lower:
        return max(lower), max(lower) == t
    return min(CKPT_SPS), False


def calce_series() -> tuple[dict, int]:
    """Canonical `load_series` CALCE capacities (raw Ah) + the true EOL cycle."""
    from make_figures import load_series
    caps, train_cells, test_cell, w, sps, eol = load_series("calce")
    caps = {c: np.asarray(v, dtype=np.float64) for c, v in caps.items()}
    assert test_cell == TEST_CELL, f"test cell {test_cell} != {TEST_CELL}"
    assert w == W_DEFAULT, f"W={w} != {W_DEFAULT}"
    i = first_crossing(caps[TEST_CELL], EOL_AH)
    assert i >= 0, f"{TEST_CELL} never crosses {EOL_AH} Ah"
    return caps, int(i + 1)


def horizon_points(ks, true_eol: int) -> dict:
    """k -> {T, checkpoint, sp_exact} for the requested grid.

    `protocol_internal` marks k = true_EOL - SP for a trained SP: only there
    is the model the one that was actually trained for that launch point.
    """
    out = {}
    for k in ks:
        t = true_eol - k
        sp, exact = sp_for(t)
        assert t - W_DEFAULT >= 0, f"k={k}: T={t} has no {W_DEFAULT}-cycle warm-up"
        out[str(int(k))] = {"k": int(k), "T": int(t), "sp": int(sp),
                            "sp_exact": bool(exact),
                            "protocol_internal": bool(int(k) in PROTOCOL_INTERNAL_KS),
                            "first_predicted_cycle": int(t + 1)}
    return out


def summarize(records: list[dict], true_eol: int) -> dict:
    """mean +- std over the runs of one model x k."""
    ok = [r for r in records if r.get("ok")]
    crossed = [r for r in ok if r["rollout"]["crossed"]]
    return {
        "n_runs_requested": len(records),
        "n_runs_ok": len(ok),
        "n_runs_failed": len(records) - len(ok),
        "mae_common": stats([r["rollout"]["mae_common"] for r in ok]),
        "mae_ah": stats([r["rollout"]["mae_ah"] for r in ok]),
        "mae_mah": stats([r["rollout"]["mae_ah"] * 1000.0 for r in ok]),
        "crossing": {
            "true_eol": int(true_eol),
            "n_cross": len(crossed),
            "n_never_cross": len(ok) - len(crossed),
            "pred_crossing_per_run": [r["rollout"]["pred_crossing_cycle"]
                                      for r in ok],
            "ae_signed_per_run": [r["rollout"]["ae"] for r in ok],
            "ae_values_sorted": sorted(int(r["rollout"]["ae"])
                                       for r in crossed),
            "ae_signed": stats([r["rollout"]["ae"] for r in crossed]),
            "ae_abs": stats([abs(r["rollout"]["ae"]) for r in crossed]),
            "ae_abs_mean_incl_never_as_never": None,
        },
        "pred_rise_frac": stats([r["rollout"]["pred_rise_frac"] for r in ok]),
        "true_rise_frac": stats([r["rollout"]["true_rise_frac"] for r in ok]),
        "end_window_std_common": stats([r["rollout"]["end_window_std_common"]
                                        for r in ok]),
        "last_step_ah": stats([r["rollout"]["last_step_ah"] for r in ok]),
        "first_step_dev_max": (float(max(r["first_step_dev"] for r in ok))
                               if ok else None),
        "first_step_all_within_tol": (bool(all(r["first_step_within_tol"]
                                               for r in ok)) if ok else None),
        "runs": records,
    }


def teacher_forced_verdict(tag: str, mae: dict, r2: dict, per_run: list,
                           n_runs: int) -> dict:
    """The gate: the teacher-forced pass must still reproduce Table A."""
    ref = REF_TF[tag]
    if ref["stat"] == "run 1" or n_runs < 10:
        # a partial run (fewer than 10 seeds) can only be compared to the
        # single-run reference
        stat = "run 1" if ref["stat"] == "run 1" else "seed 1 (partial run)"
        want = ref["mae"] if ref["stat"] == "run 1" else ref.get("seed1_mae")
        got = mae["seed1"]
        r2_got = r2["seed1"]
    else:
        stat, want, got, r2_got = ref["stat"], ref["mae"], mae["mean"], r2["mean"]
    ok = bool(got is not None and want is not None
              and abs(got - want) / want <= REF_TF_REL_TOL)
    print(f"  [verify] {tag} teacher-forced SP300 (ref = {stat}, "
          f"{ref['space']}): MAE={got:.6f} (ref {want}) "
          f"R2={r2_got:.4f} (ref {ref['r2']}) -> "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    return {"sp": 300, "space": ref["space"], "stat": stat,
            "mae": mae, "r2": r2, "reference": ref, "pass": ok,
            "n_runs": n_runs, "per_run": per_run}


def save_trajs(tag: str, trajs: dict, cycles: np.ndarray,
               true_ah: np.ndarray) -> None:
    """Merge this model's rollouts into the shared trajectory npz.

    One axis for every model: cycles 1..len(paper series), the paper's raw
    CS2_35 capacity in Ah in `true`, and one `pred_<tag>_k<k>_s<seed>` array
    per rollout, aligned to that axis and NaN outside the rollout's range (the
    arrays carry the FULL rollout, not just the k-step MAE window).
    """
    existing: dict = {}
    if os.path.exists(TRAJ_NPZ):
        with np.load(TRAJ_NPZ) as z:
            existing = {k: z[k] for k in z.files}
    existing["cycles"] = cycles
    existing["true"] = true_ah
    for key, (start_cycle, vals) in trajs.items():
        arr = np.full(len(cycles), np.nan)
        i0 = start_cycle - 1
        n = min(len(vals), len(cycles) - i0)
        arr[i0:i0 + n] = np.asarray(vals, dtype=np.float64)[:n]
        existing[f"pred_{tag}_{key}"] = arr
    np.savez_compressed(TRAJ_NPZ, **existing)
    print(f"  trajectories merged into {TRAJ_NPZ} "
          f"({len(trajs)} new arrays, {len(existing)} total)", flush=True)


# ── ours (env py312) ────────────────────────────────────────────────────
def run_ours(ks, runs) -> dict:
    import torch

    import test_ar_rollout as tar
    from eval_per_sp_existing import eval_sp_batched

    caps, true_eol = calce_series()
    caps32 = {c: caps[c].astype(np.float32) for c in caps}
    points = horizon_points(ks, true_eol)
    print(f"  load_series calce: true EOL cycle = {true_eol} (last cycle at/"
          f"above {EOL_AH} Ah); grid Ks={sorted(int(k) for k in points)}",
          flush=True)

    # -- verification: teacher-forced trajectory at SP300 (Table A) ---------
    ver = []
    for seed in runs:
        try:
            model, ck = tar.load_ckpt("calce", 300, seed)
        except (FileNotFoundError, OSError) as e:
            ver.append({"run": seed, "ok": False, "error": repr(e)})
            continue
        row = eval_sp_batched(model, caps32, [TEST_CELL], ck["lo"], ck["hi"],
                              ck["W"], 300, ck["eol_ah"])[0]
        ver.append({"run": seed, "ok": True, "sp": 300,
                    "mae_common": float(row["MAE"]), "r2": float(row["R2"]),
                    "ae": int(row["AE"]), "trul": int(row["TRUL"]),
                    "prul": int(row["PRUL"])})
        del model
        print(f"  [verify] ours SP300 seed{seed:>2}: MAE={row['MAE']:.6f} "
              f"R2={row['R2']:.4f} AE={row['AE']}", flush=True)
    good = [v for v in ver if v.get("ok")]
    verification = teacher_forced_verdict(
        "ours", {"mean": stats([v["mae_common"] for v in good])["mean"],
                 "seed1": good[0]["mae_common"] if good else None},
        {"mean": stats([v["r2"] for v in good])["mean"],
         "seed1": good[0]["r2"] if good else None}, ver, len(runs))

    # -- one model load per (seed, SP); several k share a checkpoint --------
    by_sp: dict[int, list[str]] = {}
    for key, p in points.items():
        by_sp.setdefault(p["sp"], []).append(key)

    records = {key: [] for key in points}
    trajs: dict = {}
    for seed in runs:
        for sp in sorted(by_sp):
            try:
                model, ck = tar.load_ckpt("calce", sp, seed)
            except (FileNotFoundError, OSError) as e:
                for key in by_sp[sp]:
                    records[key].append({"run": seed, "ok": False,
                                         "ckpt": f"SP{sp}_seed{seed}.pt",
                                         "error": repr(e)})
                continue
            lo, hi, w, eol_ah = ck["lo"], ck["hi"], ck["W"], ck["eol_ah"]
            # float32 sequence, exactly as ar_ae_horizon / ar_ktable_panasonic
            # build it (the AR rollout is chaotic, so the ulp matters)
            seq = (caps32[TEST_CELL] - lo) / (hi - lo + EPS)
            thr_c = (eol_ah - lo) / (hi - lo + EPS)
            for key in by_sp[sp]:
                p = points[key]
                k, T = p["k"], p["T"]
                t0 = time.time()
                # first rollout step must equal the teacher-forced one-step
                # prediction for the SAME warm-up window
                x0 = np.asarray(seq[T - w:T].reshape(1, w, 1), dtype=np.float32)
                wm, ws = float(x0[:, :, 0].mean()), float(x0[:, :, 0].std()) + EPS
                with torch.no_grad():
                    tf_first = (float(model(torch.tensor(x0, device=tar.DEV))
                                      .item()) * ws + wm)
                # ONE rollout per run, run past k until the series ends
                full = tar.rollout(model, seq, T, w, len(seq) - T)
                pv_c, tv_c = full[:k], seq[T:T + k]
                pv_ah = pv_c * (hi - lo) + lo
                tv_ah = tv_c * (hi - lo) + lo
                trajs[f"k{k}_s{seed}"] = (T + 1,
                                          full.astype(np.float64) * (hi - lo) + lo)
                cyc = crossing_cycle(full, thr_c, T + 1)
                dp, dt = np.diff(pv_c), np.diff(tv_c)
                dev = float(abs(pv_c[0] - tf_first))
                records[key].append({
                    "run": seed, "ok": True, "ckpt": f"SP{sp}_seed{seed}.pt",
                    "sp": int(sp), "sp_exact": p["sp_exact"],
                    "first_step_dev": dev,
                    "first_step_dev_ah": dev * (hi - lo),
                    "first_step_within_tol": bool(dev <= FIRST_STEP_TOL),
                    "teacher_forced_first_step": tf_first,
                    "rollout": {
                        "n_steps_mae_window": int(len(pv_c)),
                        "n_steps_full": int(len(full)),
                        "mae_common": float(np.mean(np.abs(pv_c - tv_c))),
                        "mae_ah": float(np.mean(np.abs(pv_ah - tv_ah))),
                        "pred_crossing_cycle": cyc,
                        "crossed": cyc is not None,
                        "ae": None if cyc is None else int(cyc - true_eol),
                        "pred_rise_frac": (float(np.mean(dp > 0))
                                           if len(dp) else None),
                        "true_rise_frac": (float(np.mean(dt > 0))
                                           if len(dt) else None),
                        "end_window_std_common": (float(np.std(pv_c[-w:]))
                                                  if len(pv_c) >= w else None),
                        "last_step_ah": float(pv_ah[-1]) if len(pv_ah) else None,
                    },
                })
                r = records[key][-1]["rollout"]
                print(f"  ours seed{seed:>2} SP{sp} k={k:>3} (T={T}): "
                      f"MAE={r['mae_ah'] * 1000:.2f}mAh cross="
                      f"{r['pred_crossing_cycle']} AE={r['ae']} "
                      f"d1={dev:.1e} [{time.time() - t0:.0f}s]", flush=True)
            del model
            torch.cuda.empty_cache()
    cycles = np.arange(1, len(caps[TEST_CELL]) + 1, dtype=np.int64)
    save_trajs("ours", trajs, cycles, caps[TEST_CELL])
    out = {"verification": verification, "points": {}}
    for key, recs in records.items():
        out["points"][key] = {**points[key], "summary": summarize(recs,
                                                                  true_eol)}
    return out


# ── baselines (env patchformer) ─────────────────────────────────────────
def run_baseline(tag, ks, runs) -> dict:
    import ar_rollout_baselines as arb

    caps, true_eol = calce_series()
    points = horizon_points(ks, true_eol)
    # each baseline sees the series its own training recipe built
    series = arb.official_calce() if tag == "pf" else arb.ours_calce()
    other = "ours_calce" if tag == "pf" else "official_calce"
    print(f"  {tag}: series = {len(series[TEST_CELL])} rows "
          f"({'official_calce' if tag == 'pf' else other} is the other one); "
          f"true EOL cycle = {true_eol}", flush=True)
    i_pf = first_crossing(series[TEST_CELL][:, 1], EOL_AH)
    assert i_pf + 1 == true_eol, (f"{tag} series crosses at cycle {i_pf + 1} "
                                  f"but load_series crosses at {true_eol}")

    LO, HI = common_ref()
    loader = arb.load_pf if tag == "pf" else arb.load_rm

    def load(sp, run):
        model, ckpt = loader(sp, run)
        os.chdir(ROOT)
        return model, ckpt

    # -- verification: teacher-forced trajectory at SP300 (Table A) ---------
    ver = []
    for run in runs:
        model, ckpt = load(300, run)
        df_tr, df_te, df_all, minv, maxv = arb.build_frames(series, 300)
        i_sp = int(np.where(df_all["Cycle"].values == 300)[0][0])
        y_true, y_pred, man, off, diag = arb.teacher_forced(
            model, df_all, minv, maxv, 300, i_sp, tag)
        ver.append({
            "run": run, "ok": True, "sp": 300,
            "mae_ah": float(np.mean(np.abs(y_true - y_pred))),
            "r2": float(1 - np.sum((y_true - y_pred) ** 2)
                        / (np.sum((y_true - y_true.mean()) ** 2) + EPS)),
            "decode_err": float(diag["max_decode_err"]),
            "x_is_true_capacity": bool(diag["x_is_true_capacity"])})
        del model
        print(f"  [verify] {tag} SP300 run{run}: MAE={ver[-1]['mae_ah']:.6f}Ah "
              f"R2={ver[-1]['r2']:.4f} decode_err="
              f"{ver[-1]['decode_err']:.2e}", flush=True)
    verification = teacher_forced_verdict(
        tag, {"mean": stats([v["mae_ah"] for v in ver])["mean"],
              "seed1": ver[0]["mae_ah"]},
        {"mean": stats([v["r2"] for v in ver])["mean"],
         "seed1": ver[0]["r2"]}, ver, len(runs))

    # -- one model load per (run, SP); several k share a checkpoint ---------
    by_sp: dict[int, list[str]] = {}
    for key, p in points.items():
        by_sp.setdefault(p["sp"], []).append(key)

    records = {key: [] for key in points}
    trajs: dict = {}
    for run in runs:
        for sp in sorted(by_sp):
            t0 = time.time()
            try:
                model, ckpt = load(sp, run)
            except (FileNotFoundError, IndexError, OSError) as e:
                for key in by_sp[sp]:
                    records[key].append({"run": run, "ok": False, "ckpt": "",
                                         "error": repr(e)})
                continue
            # frames depend only on the checkpoint SP -> build once
            df_tr, df_te, df_all, minv, maxv = arb.build_frames(series, sp)
            i_sp = int(np.where(df_all["Cycle"].values == sp)[0][0])
            cap_mm = df_all["Capacity"].values.astype(np.float64)
            thr_mm = (EOL_AH / RATED - minv) / (maxv - minv)
            try:
                _, _, man, _, diag = arb.teacher_forced(
                    model, df_all, minv, maxv, sp, i_sp, tag)
            except (RuntimeError, ValueError, IndexError) as e:
                for key in by_sp[sp]:
                    records[key].append({"run": run, "ok": False, "ckpt": ckpt,
                                         "error": repr(e)})
                del model
                continue
            sc_m, sc_s = diag["scaler_mean"], diag["scaler_scale"]

            def to_ah(q_mm):
                return np.asarray(q_mm) * (maxv - minv) * RATED + minv * RATED

            for key in by_sp[sp]:
                p = points[key]
                k, T = p["k"], p["T"]
                start = T + 1           # first predicted cycle (see docstring)
                i_start = int(np.where(df_all["Cycle"].values == start)[0][0])
                try:
                    full, _ = arb.rollout(model, df_all, minv, maxv, start,
                                          sc_m, sc_s,
                                          max_steps=len(cap_mm) - i_start,
                                          name=tag)
                except (RuntimeError, ValueError, IndexError) as e:
                    records[key].append({"run": run, "ok": False, "ckpt": ckpt,
                                         "error": repr(e)})
                    continue
                full = np.asarray(full)
                pv_ah = to_ah(full[:k])
                tv_ah = to_ah(cap_mm[i_start:i_start + k])
                if not np.all(np.isfinite(pv_ah)):
                    records[key].append({"run": run, "ok": False, "ckpt": ckpt,
                                         "error": "non-finite rollout"})
                    continue
                pv_c = (pv_ah - LO) / (HI - LO)
                tv_c = (tv_ah - LO) / (HI - LO)
                trajs[f"k{k}_s{run}"] = (start, to_ah(full))
                cyc = crossing_cycle(full, thr_mm, start)
                dp, dt = np.diff(full[:k]), np.diff(cap_mm[i_start:i_start + k])
                # teacher-forced first step: the TF row whose target cycle is
                # `start` (the TF trajectory starts at cycle sp)
                j = start - sp
                tf_first = float(man[j])
                dev = float(abs(pv_ah[0] - tf_first))
                records[key].append({
                    "run": run, "ok": True, "ckpt": ckpt, "sp": int(sp),
                    "sp_exact": p["sp_exact"],
                    "first_step_dev": dev, "first_step_dev_ah": dev,
                    "first_step_within_tol": bool(dev <= FIRST_STEP_TOL),
                    "teacher_forced_first_step": tf_first,
                    "tf_row_index": int(j),
                    "max_decode_err": float(diag["max_decode_err"]),
                    "rollout": {
                        "n_steps_mae_window": int(len(pv_ah)),
                        "n_steps_full": int(len(full)),
                        "mae_common": float(np.mean(np.abs(pv_c - tv_c))),
                        "mae_ah": float(np.mean(np.abs(pv_ah - tv_ah))),
                        "pred_crossing_cycle": cyc,
                        "crossed": cyc is not None,
                        "ae": None if cyc is None else int(cyc - true_eol),
                        "pred_rise_frac": (float(np.mean(dp > 0))
                                           if len(dp) else None),
                        "true_rise_frac": (float(np.mean(dt > 0))
                                           if len(dt) else None),
                        "end_window_std_common": float(np.std(pv_c[-W_DEFAULT:])
                                                       if len(pv_c) >= W_DEFAULT
                                                       else np.nan),
                        "last_step_ah": float(pv_ah[-1]) if len(pv_ah) else None,
                    },
                })
                r = records[key][-1]["rollout"]
                print(f"  {tag} run{run:>2} SP{sp} k={k:>3} (T={T}): "
                      f"MAE={r['mae_ah'] * 1000:.2f}mAh cross="
                      f"{r['pred_crossing_cycle']} AE={r['ae']} d1={dev:.1e} "
                      f"[{time.time() - t0:.0f}s]", flush=True)
            del model
    cycles = np.arange(1, len(caps[TEST_CELL]) + 1, dtype=np.int64)
    save_trajs(tag, trajs, cycles, caps[TEST_CELL])
    out = {"verification": verification, "points": {}}
    for key, recs in records.items():
        out["points"][key] = {**points[key], "summary": summarize(recs,
                                                                  true_eol)}
    return out


def common_ref() -> tuple[float, float]:
    """(LO, HI) of the comparison space: our model's train-cell min--max.

    Read straight off a checkpoint so the value is environment-independent
    (the baselines pass runs under a different torch build).
    """
    path = os.path.join(SRC, "results", "ar_ktable_calce.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        cs = doc.get("common_space")
        if cs:
            return float(cs["lo"]), float(cs["hi"])
    import torch
    ck = torch.load(os.path.join(ROOT, "checkpoints", "per_sp", "calce",
                                 "SP300_seed1.pt"),
                    map_location="cpu", weights_only=False)
    return float(ck["lo"]), float(ck["hi"])


# ── report ──────────────────────────────────────────────────────────────
def _fmt(st: dict, nd: int = 4) -> str:
    return ("-" if st.get("mean") is None
            else f"{st['mean']:.{nd}f}+-{st['std']:.{nd}f}")


def report(ks=KS) -> None:
    with open(OUT_JSON, encoding="utf-8") as f:
        doc = json.load(f)
    keys = [str(k) for k in ks if str(k) in doc["models"]["ours"]["points"]]
    print(f"common space lo={doc['common_space']['lo']:.6f} "
          f"hi={doc['common_space']['hi']:.6f}")
    print("\n[teacher-forced verification, SP300]")
    for tag, sec in doc["models"].items():
        v = sec["verification"]
        print(f"  {tag:>4}: MAE={_fmt({'mean': v['mae']['seed1'] if v['stat'] == 'run 1' else v['mae']['mean'], 'std': 0.0}, 6)} "
              f"(ref {v['reference']['mae']} {v['reference']['space']}, "
              f"{v['stat']}) pass={v['pass']}")
    print("\n[MAE over the k launch steps, mean +- std over the 10 runs]"
          "\n  (* = protocol-internal: T == SP, the model trained for it)")
    print(f"  {'k':>4} " + " ".join(f"{t:>21}" for t in ("ours", "PF", "RM")))
    for key in keys:
        cells = []
        for tag in ("ours", "pf", "rm"):
            s = doc["models"][tag]["points"][key]["summary"]["mae_common"]
            cells.append(_fmt(s))
        star = "*" if doc["models"]["ours"]["points"][key]["protocol_internal"] else " "
        print(f"  {key:>3}{star}" + " ".join(f"{c:>21}" for c in cells))
    print("\n[AE = predicted crossing - true EOL (signed cycles), "
          "crossers only; (n/10) = runs that crossed]")
    print(f"  {'k':>4} " + " ".join(f"{t:>21}" for t in ("ours", "PF", "RM")))
    for key in keys:
        cells = []
        for tag in ("ours", "pf", "rm"):
            c = doc["models"][tag]["points"][key]["summary"]["crossing"]
            cells.append(f"{_fmt(c['ae_signed'], 2)} ({c['n_cross']}/10)")
        star = "*" if doc["models"]["ours"]["points"][key]["protocol_internal"] else " "
        print(f"  {key:>3}{star}" + " ".join(f"{c:>21}" for c in cells))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["ours", "pf", "rm", "report"])
    ap.add_argument("--ks", type=int, nargs="+", default=list(ALL_KS))
    ap.add_argument("--runs", type=int, nargs="+", default=list(RUNS))
    ap.add_argument("--force-sp", type=int, default=None,
                    help="use this SP's checkpoint for every launch point "
                         "instead of the nearest SP <= T")
    args = ap.parse_args()

    global FORCE_SP
    FORCE_SP = args.force_sp

    if args.model == "report":
        report()
        return

    print(f"=== {args.model}  Ks={args.ks}  runs={args.runs} ===", flush=True)
    if args.model == "ours":
        import torch
        from make_figures import load_series
        caps, *_ = load_series("calce")
        import test_ar_rollout as tar
        ck = torch.load(os.path.join(ROOT, "checkpoints", "per_sp", "calce",
                                     "SP300_seed1.pt"), map_location="cpu",
                        weights_only=False)
        LO, HI = float(ck["lo"]), float(ck["hi"])
        sec = run_ours(args.ks, args.runs)
    else:
        LO, HI = common_ref()
        sec = run_baseline(args.model, args.ks, args.runs)

    doc = {}
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON, encoding="utf-8") as f:
            doc = json.load(f)
    doc.update({
        "protocol": {
            "dataset": "CALCE", "test_cell": TEST_CELL,
            "train_cells": list(TRAIN_CELLS), "W": W_DEFAULT,
            "eol_ah": EOL_AH, "rated_ah": RATED,
            "warmup": "the W true cycles before the launch cycle",
            "launch": "T = true_EOL - k; first predicted cycle = T + 1",
            "horizon": "k steps, ending on the true EOL cycle",
            "mae_space": "common = our checkpoint's train-cell min--max "
                         "(Ah = common*(HI-LO) + LO)",
            "ae": "predicted_crossing_cycle - true_EOL, signed; the rollout "
                  "runs to the end of the true series, not just k steps",
            "crossing_convention": "last cycle at/above the threshold (first "
                                   "sub-threshold cycle is +1)",
            "reported_ks": list(KS),
            "protocol_internal_ks_available": sorted(PROTOCOL_INTERNAL_KS),
            "runs": args.runs, "std_ddof": STD_DDOF,
            "first_step_tol": FIRST_STEP_TOL,
            "baseline_phase": "baselines launched at T+1 (their rollout "
                              "predicts the value AT the start cycle)",
            "numerics": "the AR rollout is chaotic: a float64-built input "
                        "window instead of the canonical float32 one moves the "
                        "long-horizon crossing by ~10 cycles.  These numbers "
                        "use the float32 path of ar_ae_horizon / "
                        "ar_ktable_panasonic and reproduce exactly.",
        },
        "common_space": {"lo": LO, "hi": HI,
                         "src": "checkpoints/per_sp/calce/SP300_seed1.pt"},
        "reference_teacher_forced": REF_TF,
        "notes": [
            "k = 140/240/340 (T == SP500/400/300) are the protocol-internal "
            "launch points -- the only k where the checkpoint is the one "
            "trained for that launch.  k = 140 rides along here as an "
            "end-to-end check against src/results/ar_rollout_10seed.json "
            "(its SP500 rows are the same rollout); the reported table is "
            "k = 25/50/75/100.",
            "Launch points 614/589/564/539 all lie after SP500's training "
            "cutoff (cycle 500), and the nearest SP <= launch is SP500 for "
            "every k, so the whole table is one model per method.  No "
            "EVALUATED cycle was seen in training; the W = 64 warm-up window "
            "can still reach back past the cutoff (k = 100 covers cycles "
            "476-539), exactly as the teacher-forced protocol does.",
            "MAE@k is a prefix of ONE full-length rollout per run, not a "
            "separate k-step run.",
            "A run whose rollout never crosses 0.77 Ah by the end of the "
            "series contributes no AE; it is counted separately, never "
            "imputed.",
            "AE is signed (predicted crossing cycle - true EOL cycle, 640): "
            "negative = called it early.",
            "Raw rollouts: src/results/ar_traj_calce.npz holds `cycles` "
            "(1..881), `true` (the paper's CS2_35 series in Ah) and one "
            "`pred_<model>_k<k>_s<seed>` array per rollout -- the FULL rollout "
            "in Ah, aligned to `cycles`, NaN outside its range.  PatchFormer's "
            "rollout is produced on the upstream official CALCE series, which "
            "differs slightly from the series in `true` (see notes in "
            "src/ar_rollout_baselines.py).",
        ],
    })
    doc.setdefault("models", {})[args.model] = sec
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    print(f"\nsaved {OUT_JSON}")


if __name__ == "__main__":
    main()
