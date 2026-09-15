"""k-horizon AR (self-fed) table on PANASONIC Cell03: how far before end of
life does each model's own rollout still call the crossing?

Protocol — same shape as src/ar_ae_horizon.py (CALCE), retargeted to
PANASONIC and to a doubling k grid:

  * true_EOL = 588 = Cell03's first downward crossing of 2.12 Ah (the paper's
    convention: the crossing cycle is the last cycle AT/above the threshold;
    the first sub-threshold cycle is 589).  Cell03 regenerates after that --
    it climbs back to 2.167 Ah and crosses down a second time at 646 -- so
    "true EOL" is the first crossing, as in every other script in this repo.
  * for each k, start at cycle T = true_EOL - k; warm up on the W = 30 TRUE
    cycles T-29..T; then roll exactly k steps feeding only the model's own
    predictions, so the horizon covers cycles T+1..T+k and always ends on the
    true EOL cycle.
  * MAE = trajectory MAE over the first k steps after the launch, per run;
    mean +- std over the 10 runs (ddof=1).  (Those k steps are the prefix of
    the rollout below -- one rollout per run, not two.)
  * AE  = predicted crossing cycle - true EOL, SIGNED, in cycles (negative =
    called early, positive = late).  The rollout is NOT stopped at k: it runs
    on until the predicted series first goes under the threshold (or the true
    series ends).  A run that never crosses has no AE: it is counted in the
    footnote and left out of the mean, never imputed.
  * k grid (reported table): 25 / 50 / 75 / 100, i.e. launch 25-100 cycles
    before EOL.  Other k values that were run along the way (the doubling
    grid 10/20/40/80/160/320 and the protocol-internal k = true_EOL - SP =
    88/188/288) stay in the JSON as extra points, never as table rows.
  * checkpoint = the nearest per-SP checkpoint with SP <= T.  Only SP
    300/400/500 exist.  For k = 320 (T = 268) no SP <= T exists; the nearest
    available checkpoint, SP300, is used and flagged (`sp_fallback`).

Machinery is IMPORTED, not rewritten (see CLAUDE.md / ar_rollout_panasonic):

  ours      -- src/test_ar_rollout.py : load_ckpt / rollout (per-window
               z-score decode), + src/eval_per_sp_existing.eval_sp_batched for
               the teacher-forced check against Table A's PANASONIC row.
  baselines -- src/ar_rollout_baselines.py : build_frames / teacher_forced /
               rollout / net_out / decode / scale_series, with its module
               globals rebound for PANASONIC by ar_rollout_panasonic.

Two envs, one output file each side:

    D:/anaconda/envs/py312/python.exe       src/ar_ktable_panasonic.py --model ours
    D:/anaconda/envs/patchformer/python.exe src/ar_ktable_panasonic.py --model pf
    D:/anaconda/envs/patchformer/python.exe src/ar_ktable_panasonic.py --model rm
    python src/ar_ktable_panasonic.py --model report
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
TEST_CELL = "Cell03"
TRAIN_CELLS = ("Cell01", "Cell02")
W = 30
RATED = 3.03                       # baselines' frame unit (capacity / rated)
EOL_AH = 2.12
EOL_AH_TRAINER = RATED * 0.7       # 2.121 -- the trainers' own threshold
CKPT_SPS = (300, 400, 500)
KS_TABLE = (25, 50, 75, 100)           # the four reported table rows
KS = KS_TABLE                          # grid a fresh invocation computes
RUNS = tuple(range(1, 11))
STD_DDOF = 1
# Same window, same model, same decode -> equality to float32 roundoff.  Seen
# values are ~1e-7 (baselines) / exactly 0 (ours); 1e-5 catches a warm-up or
# denormalisation error (those show up at 1e-2 and above), not float noise.
FIRST_STEP_TOL = 1e-5
REF_TF_REL_TOL = 0.02

OUT_JSON = os.path.join(SRC, "results", "ar_ktable_panasonic.json")
TRAJ_JSON = os.path.join(SRC, "results", "ar_traj_panasonic.npz")
TRAJ_PART = {t: os.path.join(SRC, "results", f"_ar_traj_panasonic_{t}.npz")
             for t in ("ours", "pf", "rm")}

# Teacher-forced reference each family must still reproduce (paper Table A,
# PANASONIC, 10-run mean), quoted in the space each reference lives in.
REF_TF = {
    "ours": {"mae": 0.0037, "r2": 0.9976, "space": "common (train-cell min-max)"},
    "pf": {"mae": 0.0033, "r2": 0.9977, "space": "Ah"},
    "rm": {"mae": 0.0174, "r2": 0.9725, "space": "Ah"},
}


# ── shared helpers ──────────────────────────────────────────────────────
def stats(vals) -> dict:
    a = np.asarray([v for v in vals if v is not None], dtype=np.float64)
    if a.size == 0:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": float(a.mean()),
            "std": float(a.std(ddof=STD_DDOF)) if a.size > 1 else 0.0,
            "n": int(a.size)}


# When set, every launch point uses this SP's checkpoint instead of the
# nearest one with SP <= T.  Needed because PANASONIC's true EOL (588) is
# early enough that k=100 launches at 488, before SP500's cutoff of 500 --
# the per-row rule would then mix SP500 and SP400 within one table.  Forcing
# SP400 for all four rows keeps the table on a single model, and every launch
# point (488..563) is still strictly after SP400's cutoff.
FORCE_SP: int | None = None


def sp_for(t: int) -> tuple[int, bool, bool]:
    """(checkpoint SP, exact match, fallback used).  Nearest SP <= T; if T
    precedes every trained SP, the nearest available (smallest) one is used
    and flagged."""
    if FORCE_SP is not None:
        return FORCE_SP, FORCE_SP == t, False
    lower = [s for s in CKPT_SPS if s <= t]
    if lower:
        sp = max(lower)
        return sp, sp == t, False
    return min(CKPT_SPS), False, True


def first_crossing(series: np.ndarray, thr: float) -> int:
    """First downward crossing index, or -1 (test_ar_rollout convention)."""
    series = np.asarray(series, dtype=np.float64)
    for i in range(len(series) - 1):
        if series[i] >= thr > series[i + 1]:
            return i
    return -1


def first_below_cycle(series: np.ndarray, thr: float, start_cycle: int
                      ) -> int | None:
    """Cycle of the first value under `thr` in `series`, whose element 0 sits
    at `start_cycle`; None if it never goes under.

    `series` must be [true value at T, pred(T+1), ..., pred(T+k)], so the scan
    can fire on the LAST horizon step -- which is exactly where the true
    crossing sits (T + k = 588).  That is the point of this table.
    """
    i = first_crossing(series, thr)
    return None if i < 0 else int(start_cycle + i + 1)


def panasonic_series_ah() -> tuple[dict, float]:
    """Raw Ah series per cell (off the shared cache) + the threshold."""
    from ar_rollout_panasonic import panasonic_caps
    caps = panasonic_caps()
    return {c: np.asarray(v, dtype=np.float64) for c, v in caps.items()}, EOL_AH


def true_first_below() -> int:
    """First cycle of Cell03 under 2.12 Ah (589 = true_EOL 588 + 1)."""
    caps, thr = panasonic_series_ah()
    i = first_crossing(caps[TEST_CELL], thr)
    assert i >= 0, "Cell03 never crosses 2.12 Ah"
    return int(i + 2)


def proto_ks() -> tuple[int, ...]:
    """k = true_EOL - SP: the launch offsets that land exactly on a per-SP
    checkpoint's own SP, i.e. the only rows where the model used is the one
    trained for that launch point (every other k borrows another SP's model).
    Computed from this dataset's series, not assumed."""
    eol = true_first_below() - 1
    return tuple(sorted(eol - sp for sp in CKPT_SPS))


def full_grid() -> tuple[int, ...]:
    """The reported table grid plus any extra k an invocation asks for."""
    return tuple(sorted(set(KS_TABLE) | set(proto_ks())))


def horizon_points(ks) -> dict:
    """k -> {T, checkpoint, true crossing cycle} for the requested grid."""
    caps, _ = panasonic_series_ah()
    below = true_first_below()
    out = {}
    for k in ks:
        t = below - 1 - k                # true_EOL - k
        sp, exact, fb = sp_for(t)
        assert t - W >= 1 and t + k <= len(caps[TEST_CELL]), \
            f"k={k}: T={t} does not fit the {len(caps[TEST_CELL])}-cycle series"
        out[str(int(k))] = {"k": int(k), "T": int(t), "sp": int(sp),
                            "sp_exact": bool(exact), "sp_fallback": bool(fb),
                            "true_crossing_cycle_first_below": int(below),
                            "true_eol_cycle": int(below - 1)}
    return out


def rollout_metrics(pred_c, true_c, pred_ah, true_ah, thr_c, prev_true_c, T,
                    true_below) -> dict:
    """MAE + shape diagnostics for ONE run over one k-step launch offset.

    `prev_true_c` is the true value at cycle T (the last warm-up cycle): the
    crossing scan starts there so a crossing on the launch's final step --
    where the true EOL sits by construction -- is detectable at all.
    """
    n = min(len(pred_c), len(true_c))
    pred_c, true_c = np.asarray(pred_c[:n]), np.asarray(true_c[:n])
    pred_ah, true_ah = np.asarray(pred_ah[:n]), np.asarray(true_ah[:n])

    cross_c = first_below_cycle(np.concatenate([[prev_true_c], pred_c]),
                                thr_c, T)
    dp, dt = np.diff(pred_c), np.diff(true_c)
    return {
        "n_steps": int(n),
        "mae_common": float(np.mean(np.abs(pred_c - true_c))),
        "mae_ah": float(np.mean(np.abs(pred_ah - true_ah))),
        # diagnostic only: does the curve already cross within the k steps?
        "pred_crossing_within_horizon": cross_c,
        "crossed_within_horizon": cross_c is not None,
        "pred_rise_frac": float(np.mean(dp > 0)) if len(dp) else None,
        "true_rise_frac": float(np.mean(dt > 0)) if len(dt) else None,
        "first_step_ah": float(pred_ah[0]) if n else None,
        "last_step_ah": float(pred_ah[-1]) if n else None,
        "true_last_step_ah": float(true_ah[-1]) if n else None,
        "end_window_std_common": (float(np.std(pred_c[-W:]))
                                  if n >= W else None),
    }


def crossing_and_ae(pred_full, prev_true_c, thr_c, T, true_below) -> dict:
    """The reported AE: the rollout runs past k until it crosses, and
    AE = predicted crossing cycle - true EOL, signed."""
    fb = first_below_cycle(np.concatenate([[prev_true_c], pred_full]), thr_c, T)
    return {"pred_crossing_cycle": fb,
            "crossed": fb is not None,
            "ae": None if fb is None else int(fb - true_below)}



def traj_pad(pred_ah: np.ndarray, T: int, n_series: int) -> np.ndarray:
    """A rollout's capacity curve laid on the shared cycle axis.

    The rollout starts at cycle T+1, so the first T entries (cycles 1..T) are
    NaN and the rest are the predictions; every rollout runs to the end of the
    true series, so all curves share one axis.
    """
    out = np.full(n_series, np.nan, dtype=np.float64)
    a = np.asarray(pred_ah, dtype=np.float64)
    out[T:T + len(a)] = a
    return out


def traj_save(tag: str, store: dict) -> None:
    np.savez_compressed(TRAJ_PART[tag], **store)
    print(f"  trajectories -> {TRAJ_PART[tag]} ({len(store)} arrays)")


def traj_merge() -> None:
    data = {}
    for t, path in TRAJ_PART.items():
        if not os.path.exists(path):
            continue
        with np.load(path) as z:
            data.update({k: z[k] for k in z.files})
    if not data:
        raise SystemExit("no partial trajectory files found")
    np.savez_compressed(TRAJ_JSON, **data)
    print(f"merged {len(data)} arrays -> {TRAJ_JSON}")


def summarize(records: list[dict]) -> dict:
    """mean +- std over the runs of one model x k (AE = signed cycles)."""
    ok = [r for r in records if r.get("ok")]
    crossed = [r for r in ok if r["rollout"]["crossed"]]
    within = [r for r in ok if r["rollout"]["crossed_within_horizon"]]
    return {
        "n_runs_requested": len(records),
        "n_runs_ok": len(ok),
        "n_runs_failed": len(records) - len(ok),
        "mae_common": stats([r["rollout"]["mae_common"] for r in ok]),
        "mae_ah": stats([r["rollout"]["mae_ah"] for r in ok]),
        "mae_mah": stats([r["rollout"]["mae_ah"] * 1000.0 for r in ok]),
        # the reported AE: rollout continued past k until it crosses
        "crossing": {
            "n_cross": len(crossed),
            "n_never_cross": len(ok) - len(crossed),
            "pred_crossing_per_run": [r["rollout"]["pred_crossing_cycle"]
                                      for r in ok],
            "ae_values_sorted": sorted(int(r["rollout"]["ae"]) for r in crossed),
            "ae_signed": stats([r["rollout"]["ae"] for r in crossed]),
        },
        # diagnostic: did the curve already cross inside the k launch window?
        "crossing_within_horizon": {
            "n_cross": len(within),
            "n_never_cross": len(ok) - len(within),
            "pred_crossing_per_run": [
                r["rollout"]["pred_crossing_within_horizon"] for r in ok],
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


def teacher_forced_verdict(tag: str, mae: dict, r2: dict, per_run: list) -> dict:
    ref = REF_TF[tag]
    ok = bool(mae["mean"] is not None
              and abs(mae["mean"] - ref["mae"]) / ref["mae"] <= REF_TF_REL_TOL)
    print(f"  [verify] {tag} teacher-forced SP300: MAE={mae['mean']:.6f} "
          f"(ref {ref['mae']} {ref['space']}) R2={r2['mean']:.4f} "
          f"(ref {ref['r2']}) -> {'PASS' if ok else 'FAIL'}", flush=True)
    return {"sp": 300, "space": ref["space"], "mae": mae, "r2": r2,
            "reference": ref, "pass": ok, "per_run": per_run}


# ── ours (env py312) ────────────────────────────────────────────────────
def run_ours(ks, runs) -> dict:
    import torch

    import test_ar_rollout as tar
    from ar_rollout_panasonic import panasonic_caps
    from eval_per_sp_existing import eval_sp_batched

    caps32 = {c: v.astype(np.float32) for c, v in panasonic_caps().items()}
    caps64, _ = panasonic_series_ah()
    below = true_first_below()
    points = horizon_points(ks)

    # -- verification: teacher-forced at SP300 (Table A's PANASONIC row) -----
    ver = []
    for seed in runs:
        try:
            model, ck = tar.load_ckpt("panasonic", 300, seed)
        except (FileNotFoundError, OSError) as e:
            ver.append({"run": seed, "ok": False, "error": repr(e)})
            continue
        row = eval_sp_batched(model, caps32, [TEST_CELL], ck["lo"], ck["hi"],
                              ck["W"], 300, ck["eol_ah"])[0]
        ver.append({"run": seed, "ok": True, "sp": 300,
                    "mae_common": float(row["MAE"]), "r2": float(row["R2"]),
                    "ae": int(row["AE"])})
        del model
    print(f"  verification seeds done: {sum(v.get('ok', False) for v in ver)}"
          f"/{len(ver)}", flush=True)
    verification = teacher_forced_verdict(
        "ours", stats([v["mae_common"] for v in ver if v.get("ok")]),
        stats([v["r2"] for v in ver if v.get("ok")]), ver)

    # -- one model load per (seed, SP); several k share a checkpoint ---------
    by_sp: dict[int, list[str]] = {}
    for key, p in points.items():
        by_sp.setdefault(p["sp"], []).append(key)
    assert all(int(key) == below - 1 - p["T"] for key, p in points.items())

    records = {key: [] for key in points}
    n_series = len(caps64[TEST_CELL])
    traj = {"cycles": np.arange(1, n_series + 1, dtype=np.int64),
            "true": caps64[TEST_CELL].copy()}
    for seed in runs:
        for sp in sorted(by_sp):
            try:
                model, ck = tar.load_ckpt("panasonic", sp, seed)
            except (FileNotFoundError, OSError) as e:
                for key in by_sp[sp]:
                    records[key].append({"run": seed, "ok": False,
                                         "ckpt": f"SP{sp}_seed{seed}.pt",
                                         "error": repr(e)})
                continue
            lo, hi, w, eol_ah = ck["lo"], ck["hi"], ck["W"], ck["eol_ah"]
            seq = (caps32[TEST_CELL] - lo) / (hi - lo + EPS)
            thr_c = (eol_ah - lo) / (hi - lo + EPS)
            for key in by_sp[sp]:
                p = points[key]
                k, T = p["k"], p["T"]
                t0 = time.time()
                # teacher-forced first step: the SAME warm-up window, one step
                x0 = seq[T - w:T].reshape(1, w, 1)
                wm, ws = float(x0.mean()), float(x0.std()) + EPS
                with torch.no_grad():
                    tf_first = (float(model(torch.tensor(x0, device=tar.DEV))
                                      .item()) * ws + wm)
                # ONE rollout per run: it runs on past k until the predicted
                # series crosses, and its first k steps are the MAE window.
                full = tar.rollout(model, seq, T, w, len(seq) - T)
                pv, pv_ah = full[:k], (full * (hi - lo) + lo)[:k]
                tv_c = seq[T:T + k]
                tv_ah = caps64[TEST_CELL][T:T + k]
                met = rollout_metrics(pv, tv_c, pv_ah, tv_ah, thr_c,
                                      float(seq[T - 1]), T, below)
                met.update(crossing_and_ae(full, float(seq[T - 1]), thr_c, T,
                                           below))
                dev = float(abs(pv[0] - tf_first))
                traj[f"pred_ours_k{k}_s{seed}"] = traj_pad(
                    full * (hi - lo) + lo, T, n_series)
                records[key].append({
                    "run": seed, "ok": True, "ckpt": f"SP{sp}_seed{seed}.pt",
                    "sp": int(sp), "sp_exact": p["sp_exact"],
                    "sp_fallback": p["sp_fallback"],
                    "first_step_dev": dev, "first_step_dev_ah": dev * (hi - lo),
                    "first_step_within_tol": bool(dev <= FIRST_STEP_TOL),
                    "teacher_forced_first_step": tf_first,
                    "rollout": met,
                })
                print(f"  ours seed{seed:>2} SP{sp} k={k:>3} (T={T}): "
                      f"MAE={met['mae_ah'] * 1000:.2f}mAh "
                      f"AE={met['ae']} (cross {met['pred_crossing_cycle']}, "
                      f"by EOL: "
                      f"{met['pred_crossing_within_horizon']}) "
                      f"d1={dev:.1e} [{time.time() - t0:.0f}s]", flush=True)
            del model
    out = {"verification": verification, "points": {}}
    for key, recs in records.items():
        out["points"][key] = {**points[key], "summary": summarize(recs)}
    traj_save("ours", traj)
    return out


# ── baselines (env patchformer) ─────────────────────────────────────────
def run_baseline(tag, ks, runs) -> dict:
    import ar_rollout_baselines as arb
    import ar_rollout_panasonic as arp

    arp.patch_baseline_globals()          # PANASONIC constants for arb
    series = arp.panasonic_series()
    LO, HI = arp.common_ref()
    below = true_first_below()
    points = horizon_points(ks)
    loader = arp.load_pf if tag == "pf" else arp.load_rm

    def load(sp, run):
        model, ckpt = loader(sp, run)
        os.chdir(ROOT)
        return model, ckpt

    # -- verification: teacher-forced at SP300 (Table A's PANASONIC row) -----
    ver = []
    for run in runs:
        model, ckpt = load(300, run)
        _, _, df_all, i_sp, minv, maxv = frame(arb, series, 300)
        y_true, y_pred, _, _, diag = arb.teacher_forced(
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
        tag, stats([v["mae_ah"] for v in ver]),
        stats([v["r2"] for v in ver]), ver)

    records = {key: [] for key in points}
    true_ah = np.asarray(series[TEST_CELL][:, 1], dtype=np.float64)
    traj = {"cycles": np.arange(1, len(true_ah) + 1, dtype=np.int64),
            "true": true_ah}
    for key, p in points.items():
        k, T, sp = p["k"], p["T"], p["sp"]
        start = T + 1                      # first predicted cycle
        _, _, df_all, i_start, minv, maxv = frame(arb, series, start)
        cap_mm = df_all["Capacity"].values.astype(np.float64)
        i_T = int(np.where(df_all["Cycle"].values == T)[0][0])

        def to_ah(q_mm):
            return np.asarray(q_mm) * (maxv - minv) * RATED + minv * RATED

        for run in runs:
            t0 = time.time()
            try:
                model, ckpt = load(sp, run)
            except (FileNotFoundError, IndexError, OSError) as e:
                records[key].append({"run": run, "ok": False, "ckpt": "",
                                     "error": repr(e)})
                continue
            try:
                _, _, man, _, diag = arb.teacher_forced(
                    model, df_all, minv, maxv, start, i_start, tag)
                sc_m, sc_s = diag["scaler_mean"], diag["scaler_scale"]
                full, _ = arb.rollout(model, df_all, minv, maxv, start,
                                      sc_m, sc_s,
                                      max_steps=len(cap_mm) - i_start,
                                      name=tag)
            except (RuntimeError, ValueError, IndexError) as e:
                records[key].append({"run": run, "ok": False, "ckpt": ckpt,
                                     "error": repr(e)})
                del model
                continue
            del model

            # Crossing + MAE are both read in ONE space -- our train-cell
            # min--max, the space the paper's MAE column lives in (thr_c is
            # the same physical 2.12 Ah seen through that map).  Mixing the
            # two spaces (common-space predictions vs the baseline's own
            # mm-space threshold) silently moves the threshold: it happened to
            # differ by only 1e-4 here, but detection in Ah against the mm
            # threshold never fires at all.
            thr_c = (EOL_AH - LO) / (HI - LO)
            prev_c = float((to_ah(cap_mm)[i_T] - LO) / (HI - LO))
            pv_ah = to_ah(np.asarray(full)[:k])
            tv_ah = to_ah(cap_mm[i_start:i_start + k])
            if not np.all(np.isfinite(pv_ah)):
                records[key].append({"run": run, "ok": False, "ckpt": ckpt,
                                     "error": "non-finite rollout"})
                continue
            met = rollout_metrics((pv_ah - LO) / (HI - LO),
                                  (tv_ah - LO) / (HI - LO), pv_ah, tv_ah,
                                  thr_c, prev_c, T, below)
            met.update(crossing_and_ae((to_ah(full) - LO) / (HI - LO), prev_c,
                                       thr_c, T, below))
            met["max_decode_err"] = float(diag["max_decode_err"])
            traj[f"pred_{tag}_k{k}_s{run}"] = traj_pad(to_ah(full), T,
                                                       len(true_ah))
            dev = float(abs(pv_ah[0] - man[0]))
            records[key].append({
                "run": run, "ok": True, "ckpt": ckpt, "sp": int(sp),
                "sp_exact": p["sp_exact"], "sp_fallback": p["sp_fallback"],
                "first_step_dev": dev, "first_step_dev_ah": dev,
                "first_step_within_tol": bool(dev <= FIRST_STEP_TOL),
                "teacher_forced_first_step": float(man[0]),
                "rollout": met,
            })
            print(f"  {tag} run{run:>2} SP{sp} k={k:>3} (T={T}): "
                  f"MAE={met['mae_ah'] * 1000:.2f}mAh "
                  f"AE={met['ae']} (cross {met['pred_crossing_cycle']}, "
                  f"by EOL: {met['pred_crossing_within_horizon']}) "
                  f"d1={dev:.1e} [{time.time() - t0:.0f}s]", flush=True)
    out = {"verification": verification, "points": {}}
    for key, recs in records.items():
        out["points"][key] = {**points[key], "summary": summarize(recs)}
    traj_save(tag, traj)
    return out


def frame(arb, series, sp):
    df_train, df_test, df_all, minv, maxv = arb.build_frames(series, sp)
    i_sp = int(np.where(df_all["Cycle"].values == sp)[0][0])
    return df_train, df_test, df_all, i_sp, minv, maxv


# ── persistence / report ────────────────────────────────────────────────
def build_protocol(doc: dict) -> dict:
    caps, _ = panasonic_series_ah()
    eol = true_first_below() - 1
    all_ks = sorted({int(k) for m in doc.get("models", {}).values()
                     for k in m.get("points", {})})
    return {
        "dataset": "PANASONIC", "test_cell": TEST_CELL,
        "train_cells": list(TRAIN_CELLS),
        "checkpoints": {
            "ours": "checkpoints/per_sp/panasonic/SP{sp}_seed{s}.pt, s=1..10",
            "pf": "reference_repos/ref_patchformer/"
                  "results_PANASONIC_RUL_prediction_sl_30/Cell03/PatchFormer/"
                  "SP{sp}/run{r}/checkpoints/*.ckpt, r=1..10",
            "rm": "reference_repos/ref_rul_mamba/Outputs/PANASONIC/"
                  "Univariable/RULMamba/Repeat_{r}/Start_Point_{sp}/"
                  "Checkpoints/*.ckpt, r=1..10",
        },
        "warmup_W": W, "rated_ah": RATED, "eol_ah": EOL_AH,
        "eol_ah_trainer_threshold": EOL_AH_TRAINER,
        "true_eol_cycle": eol, "true_first_below_cycle": eol + 1,
        "second_crossing_cycle": 646,
        "series_len_test_cell": int(len(caps[TEST_CELL])),
        "ks": list(KS_TABLE),
        "ks_protocol_internal": list(proto_ks()),
        "ks_protocol_internal_note": "true_EOL - SP for SP in 300/400/500: the "
                                     "launch offsets whose checkpoint is the "
                                     "one trained for exactly that point "
                                     "(strict comparison, not table rows).",
        "ks_computed": all_ks, "n_runs": len(RUNS),
        "launch_vs_training_cutoff": {
            "all_launches_after_their_checkpoints_SP": True,
            "detail": "launch T = 563/538/513 for k = 25/50/75 all fall after "
                      "500, so those rows use SP500 (the nearest SP <= T); "
                      "k = 100 launches at T = 488 < 500 and therefore uses "
                      "SP400, not SP500.  Every T is still strictly greater "
                      "than the SP of the checkpoint it uses (563/538/513 > "
                      "500, 488 > 400), so no rollout is evaluated on data "
                      "its model saw in training.",
        },
        "start": "T = true_EOL - k; warm-up = the W true cycles before T "
                 "(cycles T-W+1..T); the first predicted cycle is T+1, so the "
                 "horizon covers T+1..T+k and ends on the true EOL cycle.",
        "model_selection": "nearest per-SP checkpoint with SP <= T; only "
                           "SP 300/400/500 exist.  For T < 300 (k = 320, "
                           "T = 268) no SP <= T exists and the nearest "
                           "available one (SP300) is used, flagged "
                           "sp_fallback.",
        "metric_space": "MAE reported in mAh (= Ah x 1000); the same numbers "
                        "are stored in Ah and in the common (our train-cell "
                        "min-max) space, common = (Ah - LO)/(HI - LO).",
        "ae_convention": "AE = predicted crossing cycle - true EOL (588), "
                         "signed cycles: negative = the model calls EOL "
                         "early, positive = late.  A crossing is the first "
                         "cycle at/under 2.12 Ah; Cell03's first "
                         "sub-threshold cycle is 589 (the 1-cycle offset "
                         "between 'first below' and 'last at/above' the "
                         "threshold cancels because true and predicted are "
                         "read the same way).  The rollout is NOT stopped at "
                         "k: it continues until the prediction goes under the "
                         "threshold or the true series ends.  A run that "
                         "never crosses is excluded from the AE mean and "
                         "counted in the footnote, never imputed.",
        "std_ddof": STD_DDOF, "first_step_tol": FIRST_STEP_TOL,
        "points": horizon_points(all_ks),
    }


def write_doc(doc: dict) -> None:
    doc["protocol"] = build_protocol(doc)
    doc["reference_teacher_forced"] = REF_TF
    doc["notes"] = [
        "CELL03 REGENERATES: it crosses 2.12 Ah at 588, climbs back to "
        "2.167 Ah over the next ~35 cycles, and crosses down again at 646.  "
        "true_EOL is the first crossing, as everywhere else in this repo; a "
        "model that only ever predicts a smooth decline cannot represent the "
        "recovery, and one that freezes above the threshold never reports a "
        "crossing at all.",
        "MAE at k comes from ONE rollout per run, not from a run of its own: "
        "the rollout runs on until the prediction crosses, and its first k "
        "steps are the MAE window.  Each k launches from its own T with its "
        "own checkpoint.",
        "crossing_within_horizon is a DIAGNOSTIC (does the prediction cross "
        "inside the k-step launch window, i.e. by the true EOL?); the "
        "reported AE comes from the continued rollout.",
        "end_window_std_common diagnoses how the rollout ends: ~1e-7 means "
        "the per-window normaliser has collapsed and the trajectory is "
        "frozen, huge means the decode feedback diverged.",
        "The teacher-forced first step of the rollout is compared with the "
        "rollout's own first step (same warm-up window) as the protocol "
        "check; max deviation is reported per k.",
        "RAW TRAJECTORIES: src/results/ar_traj_panasonic.npz holds cycles "
        "(1-based integer cycle index, shared 1..924 axis), true (Cell03 "
        "capacity in Ah) and pred_{model}_k{k}_s{seed} (the model's "
        "self-fed rollout in Ah, from its first predicted cycle to the end "
        "of the true series, NaN-padded before the launch's first predicted "
        "cycle).  Capacities are in Ah, not normalised.",
    ]
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)


def report() -> None:
    with open(OUT_JSON, encoding="utf-8") as f:
        doc = json.load(f)
    pr = doc["protocol"]
    print(f"PANASONIC {pr['test_cell']}: true EOL cycle {pr['true_eol_cycle']}"
          f" (first sub-threshold cycle {pr['true_first_below_cycle']}, "
          f"second crossing {pr['second_crossing_cycle']}); W="
          f"{pr['warmup_W']}, threshold {pr['eol_ah']} Ah, "
          f"{pr['n_runs']} runs")
    for m in ("ours", "pf", "rm"):
        v = doc["models"].get(m, {}).get("verification")
        if not v:
            print(f"  [verify] {m}: missing")
            continue
        print(f"  [verify] {m}: teacher-forced SP300 "
              f"MAE={v['mae']['mean']:.6f} ({v['space']}, ref "
              f"{v['reference']['mae']}) R2={v['r2']['mean']:.4f} (ref "
              f"{v['reference']['r2']}) -> {'PASS' if v['pass'] else 'FAIL'}")

    def cell(key, m):
        s = ((doc["models"].get(m, {}).get("points", {}).get(key) or {})
             .get("summary"))
        if not s:
            return "-"
        return s

    proto = {str(x) for x in pr["ks_protocol_internal"]}
    table_ks = [str(x) for x in pr["ks"]]
    extra_ks = [k for k in sorted(int(k) for k in pr["ks_computed"])
                if k not in set(pr["ks"])]
    print(f"\ntable grid k={pr['ks']}"
          + (f"; also computed (not table rows): {extra_ks}" if extra_ks else "")
          + f"; protocol-internal k={pr['ks_protocol_internal']} "
            f"(= true_EOL - SP, marked ‡ where present)")
    print("\n| k | ours MAE | PF MAE | RM MAE | ours AE | PF AE | RM AE |")
    print("|---|----------|--------|--------|---------|-------|-------|")
    for key in table_ks:
        row = []
        for m in ("ours", "pf", "rm"):
            s = cell(key, m)
            if s == "-":
                row += ["-", "-"]
                continue
            mae = s["mae_mah"]
            row.append("n/a" if mae["mean"] is None
                       else f"{mae['mean']:.2f} ± {mae['std']:.2f}")
            ae = s["crossing"]["ae_signed"]
            row.append("n/a" if ae["mean"] is None
                       else f"{ae['mean']:+.1f} ± {ae['std']:.1f}")
        p = pr["points"].get(key, {})
        mark = (" ‡" if key in proto else
                (" ^" if p.get("sp_fallback") else ""))
        print(f"| {key}{mark} | " + " | ".join(row) + " |")
        print(f"    <!-- T={p.get('T')} ckpt=SP{p.get('sp')}"
              f"{' exact' if p.get('sp_exact') else ''} -->")
    if extra_ks:
        print("\n[extra k values computed along the way (informational, not "
              "part of the table)]")
        for key in [str(x) for x in extra_ks]:
            p = pr["points"].get(key, {})
            for m in ("ours", "pf", "rm"):
                s = cell(key, m)
                if s == "-":
                    continue
                mae, ae = s["mae_mah"], s["crossing"]["ae_signed"]
                print(f"  k={key:>3}{'‡' if key in proto else ' '} (T="
                      f"{p.get('T')}, SP{p.get('sp')}) {m:>4}: MAE "
                      f"{mae['mean']:.2f} ± {mae['std']:.2f} mAh; AE "
                      + ("n/a" if ae["mean"] is None
                         else f"{ae['mean']:+.1f} ± {ae['std']:.1f}")
                      + f"  ({s['crossing']['n_cross']}/"
                        f"{s['crossing']['n_cross'] + s['crossing']['n_never_cross']}"
                        f" crossed)")
    ks = [str(x) for x in sorted(int(k) for k in pr["ks_computed"])]
    print("\nfootnote a — seeds that NEVER crossed 2.12 Ah by the end of the "
          "true series (left out of the AE mean):")
    for key in ks:
        row = []
        for m in ("ours", "pf", "rm"):
            s = cell(key, m)
            row.append("-" if s == "-" else
                       f"{s['crossing']['n_never_cross']}/"
                       f"{s['crossing']['n_cross'] + s['crossing']['n_never_cross']}")
        print(f"  k={key:>3}: ours {row[0]:>5} | pf {row[1]:>5} | rm {row[2]:>5}")
    print("\nfootnote b — rows are T = true_EOL - k = 588 - k; checkpoint = "
          "nearest per-SP with SP <= T; ‡ = protocol-internal (T is exactly "
          "the SP the checkpoint was trained for); ^ = T precedes every "
          "trained SP (nearest available, SP300, used).")
    print("MAE = trajectory MAE over the first k steps after the launch (mAh), "
          "mean ± std over 10 seeds.  AE = predicted crossing cycle - true "
          "EOL, signed cycles (- = called early), over the runs that crossed.")
    print("\n[already crossed inside the k-step launch window? (diagnostic)]")
    for key in ks:
        row = []
        for m in ("ours", "pf", "rm"):
            s = cell(key, m)
            row.append("-" if s == "-" else
                       f"{s['crossing_within_horizon']['n_cross']}/"
                       f"{s['crossing_within_horizon']['n_cross'] + s['crossing_within_horizon']['n_never_cross']}")
        print(f"  k={key:>3}: ours {row[0]:>5} | pf {row[1]:>5} | rm {row[2]:>5}")
    print("\n[shape diagnostics: fraction of rollout steps that move UP "
          "(pred / true)]")
    for key in ks:
        row = []
        for m in ("ours", "pf", "rm"):
            s = cell(key, m)
            row.append("-" if s == "-" else f"{s['pred_rise_frac']['mean']:.2f}"
                       f" / {s['true_rise_frac']['mean']:.2f}")
        print(f"  k={key:>3}: ours {row[0]} | pf {row[1]} | rm {row[2]}")
    print("\n[end-window std, mean over runs (1e-7 = frozen rollout, huge = "
          "diverged)]")
    for key in ks:
        row = []
        for m in ("ours", "pf", "rm"):
            s = cell(key, m)
            row.append("-" if s == "-" else
                       f"{s['end_window_std_common']['mean']:.1e}")
        print(f"  k={key:>3}: ours {row[0]} | pf {row[1]} | rm {row[2]}")
    print("\n[first rollout step vs the teacher-forced first step]")
    for m in ("ours", "pf", "rm"):
        vals = []
        for key in ks:
            s = cell(key, m)
            if s != "-" and s["first_step_dev_max"] is not None:
                vals.append(f"k={key}: {s['first_step_dev_max']:.1e}"
                            f"{'' if s['first_step_all_within_tol'] else ' FAIL'}")
        print(f"  {m:>4}: " + "  ".join(vals))


def _upgrade_record(roll: dict, true_below: int) -> None:
    """Map an older per-run record onto the current field names, in place.

    Older records named the in-window crossing `pred_crossing_cycle`/`crossed`
    and the continued-rollout crossing `pred_crossing_full`/`crossed_full`; the
    reported AE is now the signed distance of the LATTER from true EOL.  Only
    the naming/definition changed -- the rollout itself is identical, so the
    expensive part is not repeated.
    """
    if "crossed_within_horizon" in roll:
        return                                   # already current
    roll["crossed_within_horizon"] = bool(roll.pop("crossed", False))
    roll["pred_crossing_within_horizon"] = roll.pop("pred_crossing_cycle", None)
    fb = roll.pop("pred_crossing_full", None)
    roll["pred_crossing_cycle"] = fb
    roll["crossed"] = fb is not None
    roll["ae"] = None if fb is None else int(fb - true_below)
    roll.pop("ae_full", None)


def finalize() -> None:
    """Re-derive every summary from the stored per-run records (used after a
    metric definition change, so the expensive rollouts are not repeated)."""
    below = true_first_below()
    with open(OUT_JSON, encoding="utf-8") as f:
        doc = json.load(f)
    n = 0
    for sec in doc.get("models", {}).values():
        for pt in sec.get("points", {}).values():
            for rec in pt["runs"]:
                if rec.get("ok"):
                    _upgrade_record(rec["rollout"], below)
                    n += 1
            pt["summary"] = summarize(pt["runs"])
    write_doc(doc)
    print(f"rescored {n} runs from the stored rollout records -> {OUT_JSON}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["ours", "pf", "rm", "report", "finalize",
                             "trajmerge"])
    ap.add_argument("--ks", type=int, nargs="+", default=None)
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
    if args.model == "finalize":
        finalize()
        return
    if args.model == "trajmerge":
        traj_merge()
        return

    ks = tuple(args.ks) if args.ks else tuple(KS_TABLE)
    print(f"=== {args.model}  ks={ks}  runs={args.runs} ===", flush=True)

    if not os.path.exists(OUT_JSON):
        doc = {}
    else:
        with open(OUT_JSON, encoding="utf-8") as f:
            doc = json.load(f)

    sec = (run_ours(ks, args.runs) if args.model == "ours"
           else run_baseline(args.model, ks, args.runs))

    doc["models"] = doc.get("models", {})
    old = doc["models"].get(args.model, {}).get("points", {})
    # keep any point computed by an earlier invocation (older k grids included)
    doc["models"][args.model] = {"verification": sec["verification"],
                                 "points": {**old, **sec["points"]}}
    write_doc(doc)
    print(f"\nsaved {OUT_JSON}")


if __name__ == "__main__":
    main()
