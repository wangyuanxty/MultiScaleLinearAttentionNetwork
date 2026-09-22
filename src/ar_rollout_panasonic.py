"""All-ten-run AR (self-fed) rollout of Ours / PatchFormer / RUL-Mamba on
PANASONIC Cell03 at start points 300 / 400 / 500.

Same measurement as src/ar_rollout_10seed.py (CALCE) and its two parents,
only the dataset changes; machinery is IMPORTED, not copied:

  * baselines      -> src/ar_rollout_baselines.py  (load of the two
                       checkpoints aside, everything else is reused verbatim:
                       build_frames / teacher_forced / rollout / net_out /
                       decode / scale_series / rul_value_error).  That module
                       hard-codes the CALCE constants in module globals, so
                       they are rebound for PANASONIC before use rather than
                       the functions being re-implemented.
  * ours           -> src/test_ar_rollout.py       (load_ckpt / rollout)
  * ours, teacher  -> src/eval_per_sp_existing.eval_sp_batched (the exact
    forced math         aggregation the paper's Table A PANASONIC row and
                        eval_per_sp_existing.py use)

PANASONIC facts, taken from the checkpoints (authoritative) and the shared
`checkpoints/data_cache/load_series_panasonic.pkl`, not recomputed here:
W = 30, train = Cell01+Cell02, test = Cell03 (924 cycles), eol = 2.12 Ah,
lo = 1.7590938806533813, hi = 2.7784485816955566.

Both baselines were trained by src/run_pf_nasa_adapted.py and
src/run_rm_extra.py on that same pkl (rated 3.03 Ah, eol_frac 0.7), so all
three families share one series; their per-window min--max still differs
(PatchFormer / RUL-Mamba take it over train cells + the test cell's pre-SP
cycles), which is why headline numbers are put in one COMMON space -- our
model's train-cell min--max, the space the paper's MAE column lives in.

Two invocations, each writing only its own section of one output file,
because the two families need different conda envs:

    python src/ar_rollout_panasonic.py --models baselines   # env patchformer
    python src/ar_rollout_panasonic.py --models ours        # env py312
    python src/ar_rollout_panasonic.py --models report

Full-span MAE/R2 cover [SP, end of true series]; all three SPs run to the
same series end, so the spans are 624/524/424 cycles.  They are STILL not
comparable across SPs (a longer span accumulates more rollout error), so
they are never pooled; only MAE@k is pooled across SPs.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

OUT_JSON = os.path.join(HERE, "results", "ar_rollout_panasonic.json")
OURS_CKPT_DIR = os.path.join(PROOT, "checkpoints", "per_sp", "panasonic")
SERIES_CACHE = os.path.join(PROOT, "checkpoints", "data_cache",
                            "load_series_panasonic.pkl")
PF_STORED = os.path.join(PROOT, "reference_repos", "ref_patchformer",
                         "results", "pf_panasonic_selfrun_Cell03.json")
RM_PAYLOAD = os.path.join(PROOT, "reference_repos", "ref_rul_mamba",
                          "Results", "PANASONIC_Univariable_RULMamba_Cell03.pth")
RM_METRICS = os.path.join(PROOT, "reference_repos", "ref_rul_mamba", "Outputs",
                          "PANASONIC", "Univariable", "RULMamba",
                          "Repeat_{r}", "Metrics.json")

TEST_CELL = "Cell03"
TRAIN_CELLS = ("Cell01", "Cell02")
GID = {"Cell01": 0, "Cell02": 1, "Cell03": 2}
W = 30
RATED = 3.03
EOL_AH = 2.12                 # the checkpoint's / paper's threshold
EOL_AH_TRAINER = RATED * 0.7  # 2.121, the baselines' own trainer threshold
EPS = 1e-6
HORIZONS = (10, 25, 50, 100, 150, 200)
SPS = (300, 400, 500)
RUNS = tuple(range(1, 11))
STD_DDOF = 1
# Same window, same model, same decode -> equality to float32 roundoff.  Seen
# values are ~1e-7; the bound is loose enough not to be flaky and tight enough
# to catch a warm-up/denormalisation error (those show up at 1e-2 and above).
FIRST_STEP_TOL = 1e-5

# Teacher-forced reference numbers each family must reproduce, in the space
# the reference itself is quoted in.
REF_TF = {
    "ours": {"sp": {"300": 0.0037, "400": 0.0036, "500": 0.0040},
             "space": "common (= ours' own normalisation)",
             "src": "paper/DeltaCycle_Multi-Scale_Linear_Attention.tex tab:tableA "
                    "(PANASONIC, 10-seed mean)"},
    "pf": {"sp": {"300": 0.0033, "400": 0.0037, "500": 0.0167},
           "space": "Ah",
           "src": "ref_patchformer/results/pf_panasonic_selfrun_Cell03.json "
                  "(10-run mean) + paper tab:tableA"},
    "rm": {"sp": {"300": 0.0174, "400": 0.0182, "500": 0.0212},
           "space": "Ah",
           "src": "ref_rul_mamba Outputs/PANASONIC .../Summary.json "
                  "(10-repeat mean) + paper tab:tableA"},
}
REF_TF_REL_TOL = 0.02


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
    for i in range(len(series) - 1):
        if series[i] >= thr > series[i + 1]:
            return i
    return -1


def common_ref() -> tuple[float, float]:
    """(LO, HI) of the comparison space: our model's train-cell min--max."""
    import torch
    ck = torch.load(os.path.join(OURS_CKPT_DIR, "SP300_seed1.pt"),
                    map_location="cpu", weights_only=False)
    return float(ck["lo"]), float(ck["hi"])


def panasonic_caps() -> dict:
    """Per-cell Ah series straight off the shared cache (all three models)."""
    with open(SERIES_CACHE, "rb") as f:
        caps, tr, te, w, sps, eol = pickle.load(f)
    assert te == TEST_CELL and w == W and float(eol) == EOL_AH, \
        f"cache disagrees with the checkpoint metadata: {te} {w} {eol}"
    assert tuple(tr) == TRAIN_CELLS, tr
    return caps


def panasonic_series() -> dict:
    """Same series in ar_rollout_baselines' {name: [[Cycle, Capacity]]} shape."""
    caps = panasonic_caps()
    return {n: np.stack([np.arange(1, len(a) + 1),
                         np.asarray(a, np.float64)], axis=1)
            for n, a in caps.items()}


def patch_baseline_globals() -> None:
    """Rebind ar_rollout_baselines' dataset constants for PANASONIC.

    build_frames / teacher_forced / rollout read these off the module, so
    rebinding switches the dataset without touching the protocol code.
    """
    import ar_rollout_baselines as arb
    arb.TEST_CELL = TEST_CELL
    arb.GID = dict(GID)
    arb.RATED = RATED
    arb.SEQL = W
    arb.EOL_AH = EOL_AH
    arb.EOL_FRAC = 0.7
    assert arb.SEQL == W


def rollout_metrics(pv_c, tv_c, pv_ah, tv_ah, thr_c, thr_c_alt=None,
                    first_cycle=None) -> dict:
    """One-run rollout metrics; series already in the common space (and Ah).

    k-horizon MAEs are prefixes of this single rollout, not separate runs.
    `pred_rise_frac` / `true_rise_frac` are the regeneration diagnostics: the
    fraction of steps in which the rollout / the truth moves UP, which is
    what tells a freeze-or-monotone AR failure apart from a staircase truth.
    `thr_c_alt` re-detects the crossing at the trainers' own 2.121 Ah
    (rated x 0.7), so the crossing counts can be shown not to hinge on the
    0.001 Ah difference between that and the checkpoint/paper 2.12 Ah.

    NOTE the two families do not start on the same cycle: our model's
    evaluation (eval_sp_per_sp / test_ar_rollout) warm-starts on the W true
    cycles ENDING AT SP and so first predicts Cycle SP+1, while the baseline
    runners (ar_rollout_baselines.build_frames) warm-start on the W cycles
    BEFORE SP and first predict Cycle SP.  `first_cycle` records which.
    """
    n = min(len(pv_c), len(tv_c))
    pv_c, tv_c = np.asarray(pv_c[:n]), np.asarray(tv_c[:n])
    pv_a, tv_a = np.asarray(pv_ah[:n]), np.asarray(tv_ah[:n])
    ss_tot = float(np.sum((tv_c - tv_c.mean()) ** 2)) + EPS
    r2 = float(1 - np.sum((tv_c - pv_c) ** 2) / ss_tot)
    true_c = first_crossing(tv_c, thr_c)
    pred_c = first_crossing(pv_c, thr_c)
    crossed = pred_c >= 0
    ae = (abs(true_c - pred_c) if (true_c >= 0 and crossed) else None)
    dpv, dtv = np.diff(pv_c), np.diff(tv_c)
    alt = None
    if thr_c_alt is not None:
        at, ap = first_crossing(tv_c, thr_c_alt), first_crossing(pv_c, thr_c_alt)
        alt = {"true_crossing_within_span": (int(at) + 1) if at >= 0 else None,
               "pred_crossing_within_span": (int(ap) + 1) if ap >= 0 else None,
               "crossed": bool(ap >= 0),
               "ae": (abs(at - ap) if (at >= 0 and ap >= 0) else None)}
    return {
        "span_cycles": int(n),
        "first_cycle": first_cycle,
        "mae_common": float(np.mean(np.abs(pv_c - tv_c))),
        "mae_ah": float(np.mean(np.abs(pv_a - tv_a))),
        "r2": r2,
        "mae_at_k_common": {str(k): float(np.mean(np.abs(pv_c[:k] - tv_c[:k])))
                            for k in HORIZONS if n >= k},
        "mae_at_k_ah": {str(k): float(np.mean(np.abs(pv_a[:k] - tv_a[:k])))
                        for k in HORIZONS if n >= k},
        "true_crossing_within_span": (int(true_c) + 1) if true_c >= 0 else None,
        "pred_crossing_within_span": (int(pred_c) + 1) if crossed else None,
        "crossed": bool(crossed),
        "ae": ae,
        "crossing_at_trainer_threshold": alt,
        "pred_rise_frac": float(np.mean(dpv > 0)) if len(dpv) else None,
        "true_rise_frac": float(np.mean(dtv > 0)) if len(dtv) else None,
        "end_window_std_common": (float(np.std(pv_c[-W:]))
                                  if n >= W else None),
    }


def summarize(records: list[dict], name: str, sp: int) -> dict:
    """mean +- std over runs for one model x SP."""
    ok = [r for r in records if r.get("ok")]
    spans = sorted({r["rollout"]["span_cycles"] for r in ok})
    k_keys = [str(k) for k in HORIZONS]
    crossed = [r for r in ok if r["rollout"]["crossed"]]
    ae_vals = [r["rollout"]["ae"] for r in crossed]
    alt_crossed = [r for r in ok
                   if (r["rollout"]["crossing_at_trainer_threshold"] or {})
                   .get("crossed")]
    alt_ae_vals = [(r["rollout"]["crossing_at_trainer_threshold"] or {})
                   .get("ae") for r in alt_crossed]
    tf_mae_key = "mae_common" if name == "ours" else "mae_ah"
    return {
        "n_runs_requested": len(records),
        "n_runs_ok": len(ok),
        "span_cycles": spans,
        "full_span_not_comparable_across_sps": True,
        "full_span_mae_common": stats([r["rollout"]["mae_common"] for r in ok]),
        "full_span_mae_ah": stats([r["rollout"]["mae_ah"] for r in ok]),
        "full_span_r2": stats([r["rollout"]["r2"] for r in ok]),
        "mae_at_k_common": {k: stats([r["rollout"]["mae_at_k_common"].get(k)
                                      for r in ok]) for k in k_keys},
        "mae_at_k_ah": {k: stats([r["rollout"]["mae_at_k_ah"].get(k)
                                  for r in ok]) for k in k_keys},
        "teacher_forced_mae": stats([r["teacher_forced"][tf_mae_key]
                                     for r in ok]),
        "teacher_forced_r2": stats([r["teacher_forced"]["r2"] for r in ok]),
        "teacher_forced_ae_official": stats(
            [r["teacher_forced"]["ae_official"] for r in ok]),
        "pred_rise_frac": stats([r["rollout"]["pred_rise_frac"] for r in ok]),
        "true_rise_frac": stats([r["rollout"]["true_rise_frac"] for r in ok]),
        "end_window_std_common": stats([r["rollout"]["end_window_std_common"]
                                        for r in ok]),
        "crossing": {
            "n_cross": len(crossed),
            "n_never_cross": len(ok) - len(crossed),
            "ae_values_sorted": sorted(int(a) for a in ae_vals),
            "ae": stats(ae_vals),
            "pred_crossing_per_run": [r["rollout"]["pred_crossing_within_span"]
                                      for r in ok],
        },
        "crossing_at_trainer_threshold": {
            "thr_ah": EOL_AH_TRAINER,
            "n_cross": len(alt_crossed),
            "n_never_cross": len(ok) - len(alt_crossed),
            "ae_values_sorted": sorted(int(a) for a in alt_ae_vals),
            "ae": stats(alt_ae_vals),
            "pred_crossing_per_run": [
                (r["rollout"]["crossing_at_trainer_threshold"] or {})
                .get("pred_crossing_within_span") for r in ok],
        },
        "run_summaries": [
            {"run": r["run"], "ckpt": os.path.basename(r.get("ckpt", "")),
             "first_step_dev": r["first_step_dev"],
             "first_step_within_tol": r["first_step_within_tol"],
             "full_span_mae_common": r["rollout"]["mae_common"],
             "full_span_r2": r["rollout"]["r2"],
             "crossed": r["rollout"]["crossed"],
             "ae": r["rollout"]["ae"],
             "mae_at_k_common": r["rollout"]["mae_at_k_common"]}
            for r in ok],
    }


# ── baselines (env: patchformer) ────────────────────────────────────────
def load_pf(sp: int, run: int):
    """PatchFormer checkpoint of run `run` at start point `sp`.

    Path plumbing only; the load itself is ar_rollout_baselines.load_pf's.
    """
    import torch
    arb = __import__("ar_rollout_baselines")
    ckpt_dir = os.path.join(
        arb.REF_PF, "results_PANASONIC_RUL_prediction_sl_30",
        TEST_CELL, "PatchFormer", f"SP{sp}", f"run{run}", "checkpoints")
    ckpts = [os.path.join(ckpt_dir, f) for f in sorted(os.listdir(ckpt_dir))
             if f.endswith(".ckpt")]
    if not ckpts:
        raise FileNotFoundError(ckpt_dir)
    os.chdir(arb.REF_PF)
    sys.path.insert(0, arb.REF_PF)
    from ModelsModify.PatchFormer import PatchFormerNetModel
    model = PatchFormerNetModel.load_from_checkpoint(ckpts[0]).to(arb.DEV).eval()
    return model, ckpts[0]


def load_rm(sp: int, run: int):
    """RUL-Mamba best checkpoint of repeat `run` at start point `sp`."""
    arb = __import__("ar_rollout_baselines")
    ckpt_dir = os.path.join(arb.REF_RM, "Outputs", "PANASONIC", "Univariable",
                            "RULMamba", f"Repeat_{run}",
                            f"Start_Point_{sp}", "Checkpoints")
    ckpts = [os.path.join(ckpt_dir, f) for f in sorted(os.listdir(ckpt_dir))
             if f.endswith(".ckpt")]
    if not ckpts:
        raise FileNotFoundError(ckpt_dir)
    os.chdir(arb.REF_RM)
    sys.path.insert(0, arb.REF_RM)
    from Models.RULMamba import RULMambaNetModel
    model = RULMambaNetModel.load_from_checkpoint(ckpts[0]).to(arb.DEV).eval()
    return model, ckpts[0]


def stored_rm_preds(sp: int, run: int):
    """y_pred of repeat `run` at SP, as saved by the official trainer."""
    import torch
    path = RM_PAYLOAD
    if not os.path.exists(path):
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    try:
        return np.asarray(payload["predictions"][f"SP{sp}"][run - 1],
                          dtype=np.float64)
    except (KeyError, IndexError, TypeError):
        return None


def official_refs() -> dict:
    """Per-run teacher-forced references stored by the two baseline trainers."""
    out = {"pf": {}, "rm": {}}
    if os.path.exists(PF_STORED):
        with open(PF_STORED, encoding="utf-8") as f:
            doc = json.load(f)
        for sp_key, blk in doc.items():
            out["pf"][sp_key] = {r["seed"]: r for r in blk.get("runs", [])}
    for run in RUNS:
        p = RM_METRICS.format(r=run)
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            doc = json.load(f)
        for sp_key, rec in doc.items():
            out["rm"].setdefault(sp_key, {})[run] = rec
    return out


def run_baselines(LO, HI, only_sps, runs) -> dict:
    import ar_rollout_baselines as arb

    thr_c = (EOL_AH - LO) / (HI - LO)
    thr_c_alt = (EOL_AH_TRAINER - LO) / (HI - LO)
    series = panasonic_series()
    refs = official_refs()
    frames = {}   # (sp) -> build_frames result; run-independent
    out = {}
    for name, loader in (("pf", load_pf), ("rm", load_rm)):
        out[name] = {}
        for sp in only_sps:
            t0 = time.time()
            if sp not in frames:
                frames[sp] = arb.build_frames(series, sp)
            df_train, df_test, df_all, minv, maxv = frames[sp]
            i_sp = int(np.where(df_all["Cycle"].values == sp)[0][0])
            recs = []
            for run in runs:
                try:
                    model, ckpt = loader(sp, run)
                except (FileNotFoundError, IndexError, OSError) as e:
                    recs.append({"run": run, "ok": False, "error": repr(e)})
                    continue
                y_true, y_pred, man, off, diag = arb.teacher_forced(
                    model, df_all, minv, maxv, sp, i_sp, name)
                rr, rp, ae_tf, re = arb.rul_value_error(y_true, y_pred, EOL_AH)
                tf_a = float(np.mean(np.abs(y_true - y_pred)))
                tf_mae_c = tf_a / (HI - LO)
                tf_r2 = float(1 - np.sum((y_true - y_pred) ** 2)
                              / (np.sum((y_true - y_true.mean()) ** 2) + EPS))

                preds, _ = arb.rollout(
                    model, df_all, minv, maxv, sp, diag["scaler_mean"],
                    diag["scaler_scale"], name=name)
                tv_mm = df_all["Capacity"].values[i_sp:i_sp + len(preds)]
                pv_ah = preds * (maxv - minv) * RATED + minv * RATED
                tv_ah = tv_mm * (maxv - minv) * RATED + minv * RATED
                met = rollout_metrics((pv_ah - LO) / (HI - LO),
                                      (tv_ah - LO) / (HI - LO),
                                      pv_ah, tv_ah, thr_c, thr_c_alt,
                                      first_cycle=int(sp))
                dev = float(abs(pv_ah[0] - man[0]))
                stored = stored_rm_preds(sp, run) if name == "rm" else None
                ref = refs[name].get(f"SP{sp}", {}).get(run)
                recs.append({
                    "run": run, "ok": True, "ckpt": ckpt,
                    "first_step_dev": dev,
                    "first_step_dev_common": dev / (HI - LO),
                    "first_step_within_tol": bool(dev <= FIRST_STEP_TOL),
                    "teacher_forced": {
                        "mae_ah": tf_a, "mae_common": tf_mae_c,
                        "r2": tf_r2, "ae_official": int(ae_tf),
                        "rul_true": int(rr), "rul_pred": int(rp),
                        "n": int(diag["n"]),
                        "max_decode_err": float(diag["max_decode_err"]),
                        "stored_mae_ah": (float(ref["mae"]) if ref else None),
                        "stored_r2": (float(ref["r2"]) if ref else None),
                        "stored_ae": (int(ref["ae"]) if ref else None),
                        "mae_err_vs_stored": (abs(tf_a - float(ref["mae"]))
                                              if ref else None),
                        "max_err_vs_stored_predictions": (
                            float(np.abs(stored[:len(y_pred)] - y_pred).max())
                            if stored is not None and len(stored) >= len(y_pred)
                            else None),
                    },
                    "rollout": met,
                })
                del model
                ref_s = ("-" if ref is None else
                         f"{ref['mae']:.4f}/{ref['r2']:.4f}/{ref['ae']}")
                print(f"  {name} SP{sp} run{run:>2}: tf MAE={tf_a:.4f}Ah "
                      f"R2={tf_r2:.4f} AE={ae_tf} (stored {ref_s}) | roll "
                      f"MAE={met['mae_common']:.4f}(c) R2={met['r2']:.3f} "
                      f"cross={met['pred_crossing_within_span']} "
                      f"AE={met['ae']} | d1={dev:.1e} "
                      f"[{time.time() - t0:.0f}s]", flush=True)
            out[name][f"SP{sp}"] = {
                "summary": summarize(recs, name, sp), "runs": recs}
    return out


# ── ours (env: py312) ───────────────────────────────────────────────────
def run_ours(LO, HI, only_sps, runs) -> dict:
    import torch

    import test_ar_rollout as tar
    from eval_per_sp_existing import eval_sp_batched
    from make_figures import load_series

    caps, train_cells, test_cell, W_ds, sps_ds, eol_ds = load_series("panasonic")
    caps = {c: caps[c].astype(np.float32) for c in caps}
    all_tr = np.concatenate([caps[c] for c in train_cells])
    lo_ds, hi_ds = float(all_tr.min()), float(all_tr.max())
    print(f"  load_series: W={W_ds} eol={eol_ds} lo={lo_ds:.10f} "
          f"hi={hi_ds:.10f}; common lo={LO:.10f} hi={HI:.10f}")
    if abs(lo_ds - LO) > 1e-9 or abs(hi_ds - HI) > 1e-9:
        print("  WARNING: load_series lo/hi differ from the checkpoint's; "
              "the checkpoint's are used (CLAUDE.md).")
    assert W_ds == W and float(eol_ds) == EOL_AH and test_cell == TEST_CELL
    thr_c = (EOL_AH - LO) / (HI - LO)
    thr_c_alt = (EOL_AH_TRAINER - LO) / (HI - LO)

    out = {"ours": {}}
    for sp in only_sps:
        t0 = time.time()
        recs = []
        for seed in runs:
            try:
                model, ck = tar.load_ckpt("panasonic", sp, seed)
            except (FileNotFoundError, IndexError, OSError) as e:
                recs.append({"run": seed, "ok": False, "error": repr(e)})
                continue
            lo, hi, W_ck, eol_ah = ck["lo"], ck["hi"], ck["W"], ck["eol_ah"]
            if abs(lo - LO) > 1e-9 or abs(hi - HI) > 1e-9:
                print(f"  WARNING SP{sp} seed{seed}: ckpt lo/hi="
                      f"{lo:.10f}/{hi:.10f} != common; ckpt wins for the rollout.")

            # teacher-forced: eval_sp_batched is the aggregation the paper's
            # Table A PANASONIC row came from, so this doubles as verification
            rows = eval_sp_batched(model, caps, [test_cell], lo_ds, hi_ds,
                                   W_ds, sp, eol_ds)
            row = rows[0]
            seq = (caps[test_cell] - lo) / (hi - lo + EPS)
            # first teacher-forced step == the window the rollout warm-starts from
            x0 = seq[sp - W_ck:sp].reshape(1, W_ck, 1).astype(np.float32)
            wm, ws = float(x0[:, :, 0].mean()), float(x0[:, :, 0].std()) + EPS
            with torch.no_grad():
                tf_first = float(model(
                    torch.tensor(x0, device=tar.DEV)).item()) * ws + wm

            pv = tar.rollout(model, seq, sp, W_ck, len(seq) - sp)
            tv = seq[sp:sp + len(pv)]
            pv_ah = pv * (hi - lo) + lo
            tv_ah = tv * (hi - lo) + lo
            met = rollout_metrics(pv, tv, pv_ah, tv_ah, thr_c, thr_c_alt,
                                  first_cycle=int(sp) + 1)
            dev = float(abs(pv[0] - tf_first))
            recs.append({
                "run": seed, "ok": True, "ckpt": f"SP{sp}_seed{seed}.pt",
                "first_step_dev": dev, "first_step_dev_common": dev,
                "first_step_within_tol": bool(dev <= FIRST_STEP_TOL),
                "teacher_forced": {
                    "mae_common": float(row["MAE"]),
                    "rmse_common": float(row["RMSE"]),
                    "r2": float(row["R2"]), "ae_official": int(row["AE"]),
                    "trul": int(row["TRUL"]), "prul": int(row["PRUL"]),
                    "lo": lo, "hi": hi, "W": W_ck, "eol_ah": eol_ah,
                },
                "rollout": met,
            })
            del model
            print(f"  ours SP{sp} seed{seed:>2}: tf MAE={row['MAE']:.4f} "
                  f"R2={row['R2']:.4f} AE={row['AE']} | roll "
                  f"MAE={met['mae_common']:.4f} R2={met['r2']:.3f} "
                  f"cross={met['pred_crossing_within_span']} AE={met['ae']} | "
                  f"d1={dev:.1e} [{time.time() - t0:.0f}s]", flush=True)
        out["ours"][f"SP{sp}"] = {"summary": summarize(recs, "ours", sp),
                                  "runs": recs}
    return out


# ── verification ────────────────────────────────────────────────────────
def verify(section: str, data: dict) -> dict:
    """Teacher-forced pass must still reproduce the stored reference numbers."""
    res = {}
    for name, per_sp in data.items():
        ref = REF_TF.get(name)
        if ref is None:
            continue
        for sp_key, got in per_sp.items():
            sp = sp_key.replace("SP", "")
            if sp not in ref["sp"]:
                continue
            tfk = "mae_common" if name == "ours" else "mae_ah"
            got_v = got["summary"]["teacher_forced_mae"]["mean"]
            want = ref["sp"][sp]
            rel = abs(got_v - want) / want
            ok = rel <= REF_TF_REL_TOL
            res[f"{name}_{sp_key}"] = {
                "metric": tfk, "got": got_v, "want": want,
                "rel_err": float(rel), "pass": bool(ok),
                "space": ref["space"], "src": ref["src"],
                "r2_got": got["summary"]["teacher_forced_r2"]["mean"],
                "ae_got": got["summary"]["teacher_forced_ae_official"]["mean"]}
            print(f"  [verify] {name} {sp_key}: teacher-forced MAE "
                  f"{got_v:.6f} vs ref {want:.6f} ({ref['space']}) "
                  f"rel={rel:.2e} -> {'PASS' if ok else 'FAIL'}", flush=True)
    return res


def per_run_agreement(data: dict) -> dict:
    """Exact per-run agreement with what the baseline trainers themselves stored."""
    res = {}
    for name, per_sp in data.items():
        if name not in ("pf", "rm"):
            continue
        errs = {"mae": [], "r2": [], "ae": [], "traj": []}
        for sp_key, got in per_sp.items():
            for r in got["runs"]:
                if not r.get("ok"):
                    continue
                tf = r["teacher_forced"]
                if tf.get("mae_err_vs_stored") is not None:
                    errs["mae"].append(tf["mae_err_vs_stored"])
                    errs["r2"].append(abs(tf["r2"] - tf["stored_r2"]))
                    errs["ae"].append(abs(tf["ae_official"] - tf["stored_ae"]))
                if tf.get("max_err_vs_stored_predictions") is not None:
                    errs["traj"].append(tf["max_err_vs_stored_predictions"])
        res[name] = {
            "n_runs_compared": len(errs["mae"]),
            "max_abs_mae_err_vs_stored": (max(errs["mae"]) if errs["mae"] else None),
            "max_abs_r2_err_vs_stored": (max(errs["r2"]) if errs["r2"] else None),
            "max_abs_ae_err_vs_stored": (max(errs["ae"]) if errs["ae"] else None),
            "n_runs_traj_compared": len(errs["traj"]),
            "max_abs_traj_err_vs_stored": (max(errs["traj"])
                                           if errs["traj"] else None),
        }
        print(f"  [per-run] {name}: n={res[name]['n_runs_compared']} "
              f"max|dMAE|={res[name]['max_abs_mae_err_vs_stored']} "
              f"max|dR2|={res[name]['max_abs_r2_err_vs_stored']} "
              f"max|dAE|={res[name]['max_abs_ae_err_vs_stored']} "
              f"traj(n={res[name]['n_runs_traj_compared']}) "
              f"max|dy|={res[name]['max_abs_traj_err_vs_stored']}", flush=True)
    return res


def pool_at_k(data: dict) -> dict:
    """Mean +- std at fixed k over the 3 SPs x 10 runs (30 rollouts).

    Legitimate because a k-step prefix is comparable across SPs; the
    full-span numbers are not, and are never pooled here.
    """
    pooled = {}
    for name, per_sp in data.items():
        per_k = {}
        for k in (str(k) for k in HORIZONS):
            vals, per_sp_means = [], {}
            for sp_key, got in per_sp.items():
                v = [r["mae_at_k_common"].get(k) for r in got["summary"]
                     ["run_summaries"] if r["mae_at_k_common"].get(k) is not None]
                vals += v
                per_sp_means[sp_key] = float(np.mean(v)) if v else None
            per_k[k] = {**stats(vals), "per_sp_mean": per_sp_means}
        pooled[name] = per_k
    return pooled


def report() -> None:
    """Print the tables straight off OUT_JSON (no recomputation)."""
    with open(OUT_JSON, encoding="utf-8") as f:
        doc = json.load(f)
    print(f"common space lo={doc['common_space']['lo']:.6f} "
          f"hi={doc['common_space']['hi']:.6f}; "
          f"eol={doc['protocol']['eol_ah']} Ah; runs={doc['protocol']['runs']}")
    print("\n[verification]")
    for k, v in doc.get("verification", {}).items():
        print(f"  {k}: {v['metric']}={v['got']:.6f} vs ref {v['want']:.6f} "
              f"rel={v['rel_err']:.2e} {'PASS' if v['pass'] else 'FAIL'} "
              f"(r2 {v['r2_got']:.4f})")
    for fam in ("ours", "baselines"):
        d = doc.get(f"max_first_step_dev_{fam}")
        if d is not None:
            print(f"  max first-step deviation ({fam}): {d:.3e} "
                  f"(all within tol: {doc[f'first_step_all_ok_{fam}']})")

    hdr = (f"  {'model':>4} {'SP':>5} {'n':>3} "
           + " ".join(f"{'MAE@' + str(k):>15}" for k in HORIZONS)
           + f" {'MAE full':>15} {'R2 full':>15}  span")
    print("\n[full-span MAE/R2 and MAE@k, common space, mean +- std over runs]")
    print(hdr)
    for name, per_sp in doc["models"].items():
        for sp_key in sorted(per_sp):
            s = per_sp[sp_key]["summary"]
            cells = []
            for k in HORIZONS:
                v = s["mae_at_k_common"].get(str(k), {})
                cells.append(f"{v['mean']:.4f}+-{v['std']:.4f}"
                             if v.get("mean") is not None else "-")
            fs, r2 = s["full_span_mae_common"], s["full_span_r2"]
            print(f"  {name:>4} {sp_key:>5} {s['n_runs_ok']:>3} "
                  + " ".join(f"{c:>15}" for c in cells)
                  + f" {fs['mean']:>7.4f}+-{fs['std']:.4f}"
                  + f" {r2['mean']:>7.4f}+-{r2['std']:.4f}"
                  + f"  {s['span_cycles']} cyc")
    print("\n[pooled at fixed k over 3 SPs x runs, common space]")
    for name, per_k in doc.get("pooled_at_k_common", {}).items():
        cells = " ".join(f"k={k}:{v['mean']:.4f}+-{v['std']:.4f}"
                         for k, v in per_k.items())
        print(f"  {name:>4}  {cells}")
    print("\n[crossing: AE only where the predicted curve crosses the threshold]")
    for name, per_sp in doc["models"].items():
        for sp_key in sorted(per_sp):
            for label, key in (("2.12Ah", "crossing"),
                              ("2.121Ah", "crossing_at_trainer_threshold")):
                c = per_sp[sp_key]["summary"].get(key)
                if not c:
                    continue
                ae = (f"{c['ae']['mean']:.2f}+-{c['ae']['std']:.2f} "
                      f"{c['ae_values_sorted']}" if c["n_cross"] else "none")
                print(f"  {name:>4} {sp_key} @{label:>8}: cross {c['n_cross']}/"
                      f"{c['n_cross'] + c['n_never_cross']}"
                      f", never {c['n_never_cross']}, AE={ae}")
                if label == "2.12Ah":
                    print(f"        pred crossing per run: "
                          f"{c['pred_crossing_per_run']}")
    print("\n[regeneration / freeze diagnostics, common space]")
    for name, per_sp in doc["models"].items():
        for sp_key in sorted(per_sp):
            s = per_sp[sp_key]["summary"]
            ok_runs = [r for r in per_sp[sp_key]["runs"] if r.get("ok")]
            first = (ok_runs[0]["rollout"].get("first_cycle")
                     if ok_runs else None)
            print(f"  {name:>4} {sp_key} (span {s['span_cycles']} cyc from "
                  f"Cycle {first}): pred rises on "
                  f"{s['pred_rise_frac']['mean']:.3f} of steps, truth on "
                  f"{s['true_rise_frac']['mean']:.3f}; end-window std "
                  f"{s['end_window_std_common']['mean']:.2e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True,
                    choices=["ours", "baselines", "report"])
    ap.add_argument("--sps", type=int, nargs="+", default=list(SPS))
    ap.add_argument("--runs", type=int, nargs="+", default=list(RUNS))
    args = ap.parse_args()

    if args.models == "report":
        report()
        return

    LO, HI = common_ref()
    print(f"common space: lo={LO:.10f} hi={HI:.10f}", flush=True)

    doc = {}
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON, encoding="utf-8") as f:
            doc = json.load(f)
    doc.update({
        "protocol": {
            "dataset": "PANASONIC", "test_cell": TEST_CELL,
            "train_cells": list(TRAIN_CELLS),
            "warmup_cycles": W,
            "warmup": "W true cycles before SP, then only own predictions",
            "stop": "end of the true series",
            "eol_ah": EOL_AH, "rated_ah": RATED,
            "eol_ah_trainer_threshold": EOL_AH_TRAINER,
            "series_len_test_cell": len(panasonic_caps()[TEST_CELL]),
            "horizons": list(HORIZONS), "runs": args.runs, "sps": args.sps,
            "metric_space": "common = our model's train-cell min--max "
                            "(Ah = common*(HI-LO) + LO)",
            "std_ddof": STD_DDOF,
            "first_step_tol": FIRST_STEP_TOL,
        },
        "common_space": {"lo": LO, "hi": HI,
                         "src": "checkpoints/per_sp/panasonic/SP300_seed1.pt"},
        "reference": REF_TF,
        "notes": [
            "k-horizon MAEs are prefixes of ONE rollout per run, not separate "
            "runs.",
            "Full-span MAE/R2 are labelled with their span length.  Here all "
            "three families stop at the same series end, so the spans are "
            "624/524/424 cycles, but they are still not comparable across SPs "
            "and are never pooled; only MAE@k is pooled across SPs.",
            "A run whose predicted curve never crosses the threshold "
            "contributes no AE; it is counted separately, never imputed.",
            "The trainers' own AE used threshold = rated x 0.7 = %.4f Ah; the "
            "paper's and the checkpoint's threshold is %.2f Ah.  Headline "
            "numbers use %.2f Ah for all three families; the sensitivity is "
            "reported." % (EOL_AH_TRAINER, EOL_AH, EOL_AH),
        ],
        "checkpoints": {
            "ours": "checkpoints/per_sp/panasonic/SP{sp}_seed{s}.pt, s=1..10",
            "pf": "ref_patchformer/results_PANASONIC_RUL_prediction_sl_30/"
                  "Cell03/PatchFormer/SP{sp}/run{r}/checkpoints/*.ckpt, r=1..10",
            "rm": "ref_rul_mamba/Outputs/PANASONIC/Univariable/RULMamba/"
                  "Repeat_{r}/Start_Point_{sp}/Checkpoints/*.ckpt, r=1..10",
        },
    })
    if args.models == "baselines":
        patch_baseline_globals()
        sec = run_baselines(LO, HI, args.sps, args.runs)
        doc.setdefault("models", {}).update(sec)
        doc["verification"] = {**doc.get("verification", {}), **verify("b", sec)}
        doc["per_run_agreement"] = {**doc.get("per_run_agreement", {}),
                                    **per_run_agreement(sec)}
    else:
        sec = run_ours(LO, HI, args.sps, args.runs)
        doc.setdefault("models", {}).update(sec)
        doc["verification"] = {**doc.get("verification", {}), **verify("o", sec)}

    if "models" in doc:
        doc["pooled_at_k_common"] = pool_at_k(doc["models"])
        key = f"max_first_step_dev_{args.models}"
        devs = [r["first_step_dev"]
                for per_sp in doc["models"].values()
                for got in per_sp.values()
                for r in got["runs"] if r.get("ok")]
        doc[key] = float(max(devs)) if devs else None
        doc[f"first_step_all_ok_{args.models}"] = bool(
            all(r["first_step_within_tol"]
                for per_sp in doc["models"].values()
                for got in per_sp.values() for r in got["runs"] if r.get("ok")))

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    print(f"\nsaved {OUT_JSON}")


if __name__ == "__main__":
    main()
