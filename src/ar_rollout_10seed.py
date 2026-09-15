"""All-ten-run AR rollout of Ours / PatchFormer / RUL-Mamba on CALCE CS2_35.

Extends src/ar_rollout_baselines.py, which ran a single draw per model, to
every available run/seed, so the single-run ranking can be judged against the
spread.  The protocol is unchanged and the machinery is IMPORTED, not copied:

  * baselines      -> src/ar_rollout_baselines.py  (load_pf / load_rm /
                       build_frames / teacher_forced / rollout)
  * ours           -> src/test_ar_rollout.py       (load_ckpt / rollout)
  * ours, teacher  -> src/eval_per_sp_existing.eval_sp_batched (the exact
    forced math         aggregation per_sp_summary.json was produced with)

Two invocations, each writing only its own section of one output file, because
the two families need different conda envs:

    conda run -n patchformer python src/ar_rollout_10seed.py --models baselines
    conda run -n py312       python src/ar_rollout_10seed.py --models ours

Metrics are reported in the COMMON space -- our model's lo/hi, the space the
paper's MAE column lives in -- because the three models each normalise with
their own train-cell min--max and raw MAEs are not comparable across those.
The baselines' own-space MAE is kept alongside for continuity with the
single-run JSON.  MAE in Ah is a fixed constant multiple of the common-space
value (Ah = common*(HI-LO) + LO), so it does not change any ranking.

Full-span MAE/R2 cover [SP, end of true series], and the three SPs have
different span lengths (583/483/383 cycles for PatchFormer, 582/482/382 for
RUL-Mamba, 581/481/381 for ours).  Full-span values are therefore NOT
comparable across SPs and are deliberately not pooled; only MAE@k is pooled.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

OUT_JSON = os.path.join(HERE, "results", "ar_rollout_10seed.json")
OURS_CKPT_DIR = os.path.join(PROOT, "checkpoints", "per_sp", "calce")

EOL_AH = 0.77
RATED = 1.1
EPS = 1e-6
HORIZONS = (10, 25, 50, 100, 150, 200)
SPS = (300, 400, 500)
RUNS = tuple(range(1, 11))
STD_DDOF = 1
# Same window, same model, same decode -> equality to float32 roundoff.  Seen
# values are ~1e-7; the bound is loose enough not to be flaky and tight enough
# to catch a warm-up/denormalisation error (those show up at 1e-2 and above).
FIRST_STEP_TOL = 1e-5

# Teacher-forced numbers each family must still reproduce, and WHICH statistic
# the reference is: ours' stored numbers are 10-seed means (per_sp_summary),
# while the baselines' come from the single-run ar_rollout_baselines.json, so
# there the comparison is run 1 (different runs are different models, so a
# 10-run mean is not expected to hit a single-run reference).
REF_TF = {
    "ours": {"stat": "mean", "space": "common (= ours' own normalisation)",
             "src": "src/results/per_sp_summary.json (10-seed mean)",
             "sp": {300: {"mae": 0.006638, "r2": 0.99514},
                    400: {"mae": 0.007750, "r2": 0.99327},
                    500: {"mae": 0.009215, "r2": 0.98984}}},
    "pf": {"stat": "run1", "space": "Ah",
           "src": "src/results/ar_rollout_baselines.json run1",
           "sp": {300: {"mae": 0.0058, "r2": 0.9962}}},
    "rm": {"stat": "run1", "space": "Ah",
           "src": "src/results/ar_rollout_baselines.json run1",
           "sp": {300: {"mae": 0.0175, "r2": 0.9799, "ae": 11}}},
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
    """(LO, HI) of the comparison space: our model's train-cell min--max.

    Read straight off a checkpoint so the value is environment-independent
    (the baselines pass runs under a different torch build).
    """
    import torch
    ck = torch.load(os.path.join(OURS_CKPT_DIR, "SP300_seed1.pt"),
                    map_location="cpu", weights_only=False)
    return float(ck["lo"]), float(ck["hi"])


def rollout_metrics(pv_c, tv_c, pv_ah, tv_ah, thr_c) -> dict:
    """One-run rollout metrics; series already in the common space (and Ah).

    k-horizon MAEs are prefixes of this single rollout, not separate runs.
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
    return {
        "span_cycles": int(n),
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
        "end_window_std_common": (float(np.std(pv_c[-64:]))
                                  if n >= 64 else None),
    }


def summarize(records: list[dict], name: str, sp: int) -> dict:
    """mean +- std over runs for one model x SP."""
    ok = [r for r in records if r.get("ok")]
    spans = sorted({r["rollout"]["span_cycles"] for r in ok})
    k_keys = [str(k) for k in HORIZONS]
    crossed = [r for r in ok if r["rollout"]["crossed"]]
    ae_vals = [r["rollout"]["ae"] for r in crossed]
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
        "crossing": {
            "n_cross": len(crossed),
            "n_never_cross": len(ok) - len(crossed),
            "ae_values_sorted": sorted(int(a) for a in ae_vals),
            "ae": stats(ae_vals),
            "pred_crossing_per_run": [r["rollout"]["pred_crossing_within_span"]
                                      for r in ok],
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
def run_baselines(LO, HI, series_by_model, only_sps, runs) -> dict:
    import ar_rollout_baselines as arb

    thr_c = (EOL_AH - LO) / (HI - LO)
    out = {}
    for name, loader in (("pf", arb.load_pf), ("rm", arb.load_rm)):
        series = series_by_model[name]
        out[name] = {}
        for sp in only_sps:
            t0 = time.time()
            recs = []
            for run in runs:
                try:
                    model, ckpt = loader(sp, run)
                except (FileNotFoundError, IndexError, OSError) as e:
                    recs.append({"run": run, "ok": False, "error": repr(e)})
                    continue
                df_train, df_test, df_all, minv, maxv = arb.build_frames(
                    series, sp)
                i_sp = int(np.where(df_all["Cycle"].values == sp)[0][0])
                y_true, y_pred, man, off, diag = arb.teacher_forced(
                    model, df_all, minv, maxv, sp, i_sp, name)
                rr, rp, ae_tf, re = arb.rul_value_error(y_true, y_pred, EOL_AH)
                # teacher-forced metrics: MAE in Ah (the stored reference
                # space) and in common; R2 is affine-invariant
                tf_a = float(np.mean(np.abs(y_true - y_pred)))
                tf_mae_c = tf_a / (HI - LO)
                tf_r2 = float(1 - np.sum((y_true - y_pred) ** 2)
                              / (np.sum((y_true - y_true.mean()) ** 2) + EPS))

                preds, i_sp2 = arb.rollout(
                    model, df_all, minv, maxv, sp, diag["scaler_mean"],
                    diag["scaler_scale"], name=name)
                tv_mm = df_all["Capacity"].values[i_sp:i_sp + len(preds)]
                pv_ah = preds * (maxv - minv) * RATED + minv * RATED
                tv_ah = tv_mm * (maxv - minv) * RATED + minv * RATED
                met = rollout_metrics((pv_ah - LO) / (HI - LO),
                                      (tv_ah - LO) / (HI - LO),
                                      pv_ah, tv_ah, thr_c)
                dev = float(abs(pv_ah[0] - man[0]))
                recs.append({
                    "run": run, "ok": True, "ckpt": ckpt,
                    "first_step_dev": dev,
                    "first_step_dev_common": dev / (HI - LO),
                    "first_step_within_tol": bool(dev <= FIRST_STEP_TOL),
                    "teacher_forced": {
                        "mae_ah": float(tf_a), "mae_common": tf_mae_c,
                        "r2": tf_r2, "ae_official": int(ae_tf),
                        "rul_true": int(rr), "rul_pred": int(rp),
                        "n": int(diag["n"]),
                        "max_decode_err": float(diag["max_decode_err"]),
                    },
                    "rollout": met,
                })
                del model
                print(f"  {name} SP{sp} run{run:>2}: tf MAE={tf_a:.4f}Ah "
                      f"R2={tf_r2:.4f} AE={ae_tf} | roll MAE={met['mae_common']:.4f}"
                      f"(c) R2={met['r2']:.3f} cross={met['pred_crossing_within_span']}"
                      f" AE={met['ae']} | d1={dev:.1e} "
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

    caps, train_cells, test_cell, W_ds, sps_ds, eol_ds = load_series("calce")
    caps = {c: caps[c].astype(np.float32) for c in caps}
    all_tr = np.concatenate([caps[c] for c in train_cells])
    lo_ds, hi_ds = float(all_tr.min()), float(all_tr.max())
    print(f"  load_series: W={W_ds} eol={eol_ds} lo={lo_ds:.10f} "
          f"hi={hi_ds:.10f}; common lo={LO:.10f} hi={HI:.10f}")
    if abs(lo_ds - LO) > 1e-9 or abs(hi_ds - HI) > 1e-9:
        print("  WARNING: load_series lo/hi differ from the checkpoint's; "
              "the checkpoint's are used (CLAUDE.md).")
    thr_c = (EOL_AH - LO) / (HI - LO)

    out = {"ours": {}}
    for sp in only_sps:
        t0 = time.time()
        recs = []
        for seed in runs:
            try:
                model, ck = tar.load_ckpt("calce", sp, seed)
            except (FileNotFoundError, IndexError, OSError) as e:
                recs.append({"run": seed, "ok": False, "error": repr(e)})
                continue
            lo, hi, W, eol_ah = ck["lo"], ck["hi"], ck["W"], ck["eol_ah"]
            if abs(lo - LO) > 1e-9 or abs(hi - HI) > 1e-9:
                print(f"  WARNING SP{sp} seed{seed}: ckpt lo/hi="
                      f"{lo:.10f}/{hi:.10f} != common; ckpt wins for the rollout.")

            # teacher-forced: eval_sp_batched is the aggregation the stored
            # per_sp_summary.json came from, so this doubles as verification
            rows = eval_sp_batched(model, caps, [test_cell], lo_ds, hi_ds,
                                   W_ds, sp, eol_ds)
            row = rows[0]
            seq = (caps[test_cell] - lo) / (hi - lo + EPS)
            # first teacher-forced step == the window the rollout warm-starts from
            x0 = seq[sp - W:sp].reshape(1, W, 1).astype(np.float32)
            wm, ws = float(x0[:, :, 0].mean()), float(x0[:, :, 0].std()) + EPS
            with torch.no_grad():
                tf_first = float(model(
                    torch.tensor(x0, device=tar.DEV)).item()) * ws + wm

            pv = tar.rollout(model, seq, sp, W, len(seq) - sp)
            tv = seq[sp:sp + len(pv)]
            pv_ah = pv * (hi - lo) + lo
            tv_ah = tv * (hi - lo) + lo
            met = rollout_metrics(pv, tv, pv_ah, tv_ah, thr_c)
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
                    "lo": lo, "hi": hi, "W": W, "eol_ah": eol_ah,
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
def verify(data: dict) -> dict:
    """Teacher-forced pass must still reproduce the stored reference numbers.

    Reads the per-run records (so it can pick run 1 where the reference is a
    single run), and reports the 10-run mean/min/max next to it either way.
    """
    res = {}
    for name, per_sp in data.items():
        ref = REF_TF.get(name)
        if ref is None:
            continue
        tfk = "mae_common" if name == "ours" else "mae_ah"
        for sp_key, got in per_sp.items():
            sp = sp_key.replace("SP", "")
            if int(sp) not in ref["sp"]:
                continue
            want = ref["sp"][int(sp)]
            runs = [r for r in got["runs"] if r.get("ok")]
            vals = [r["teacher_forced"][tfk] for r in runs]
            r2s = [r["teacher_forced"]["r2"] for r in runs]
            if ref["stat"] == "run1":
                i = next(j for j, r in enumerate(runs) if r["run"] == 1)
                mae_got, r2_got = vals[i], r2s[i]
            else:
                mae_got, r2_got = float(np.mean(vals)), float(np.mean(r2s))
            rel = abs(mae_got - want["mae"]) / want["mae"]
            ok = rel <= REF_TF_REL_TOL
            entry = {
                "metric": tfk, "stat": ref["stat"], "space": ref["space"],
                "src": ref["src"],
                "mae_got": mae_got, "mae_want": want["mae"],
                "mae_rel_err": float(rel), "pass": bool(ok),
                "mae_run1": vals[0], "mae_min": float(min(vals)),
                "mae_max": float(max(vals)),
                "r2_got": r2_got,
                "r2_want": want.get("r2"),
                "n_runs": len(vals),
            }
            if "ae" in want:
                ae_got = (int(runs[i]["teacher_forced"]["ae_official"])
                          if ref["stat"] == "run1"
                          else float(np.mean([r["teacher_forced"]["ae_official"]
                                              for r in runs])))
                entry.update({"ae_got": ae_got, "ae_want": want["ae"],
                              "ae_pass": bool(ae_got == want["ae"]
                                              if ref["stat"] == "run1"
                                              else abs(ae_got - want["ae"]) <= 1)})
                ok = ok and entry["ae_pass"]
                entry["pass"] = bool(ok)
            print(f"  [verify] {name} {sp_key}: teacher-forced MAE "
                  f"{mae_got:.6f} vs ref {want['mae']:.6f} ({ref['space']}, "
                  f"{ref['stat']}) rel={rel:.2e} -> {'PASS' if ok else 'FAIL'}"
                  + (f"; AE {entry['ae_got']} vs {entry['ae_want']}"
                     if "ae" in want else ""), flush=True)
            res[f"{name}_{sp_key}"] = entry
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
          f"runs={doc['protocol']['runs']}")
    print("\n[verification: teacher-forced MAE vs the stored reference]")
    for k, v in doc.get("verification", {}).items():
        print(f"  {k}: {v['stat']:>5} {v['metric']}={v['mae_got']:.6f} vs ref "
              f"{v['mae_want']:.6f} rel={v['mae_rel_err']:.2e} "
              f"{'PASS' if v['pass'] else 'FAIL'} "
              f"(r2 {v['r2_got']:.4f} vs {v['r2_want']}; run1..max "
              f"{v['mae_run1']:.6f}..{v['mae_max']:.6f})")
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
    print("\n[crossing: AE only where the predicted curve crosses 0.77 Ah]")
    for name, per_sp in doc["models"].items():
        for sp_key in sorted(per_sp):
            c = per_sp[sp_key]["summary"]["crossing"]
            ae = (f"{c['ae']['mean']:.2f}+-{c['ae']['std']:.2f} "
                  f"{c['ae_values_sorted']}" if c["n_cross"] else "none")
            print(f"  {name:>4} {sp_key}: cross {c['n_cross']}/{c['n_cross'] + c['n_never_cross']}"
                  f", never {c['n_never_cross']}, AE={ae}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True,
                    choices=["ours", "baselines", "report", "verify"])
    ap.add_argument("--sps", type=int, nargs="+", default=list(SPS))
    ap.add_argument("--runs", type=int, nargs="+", default=list(RUNS))
    args = ap.parse_args()

    if args.models in ("report", "verify"):
        if args.models == "report":
            report()
            return
        with open(OUT_JSON, encoding="utf-8") as f:
            doc = json.load(f)
        doc["verification"] = verify(doc["models"])
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)
        print(f"updated {OUT_JSON}")
        return

    LO, HI = common_ref()
    print(f"common space: lo={LO:.10f} hi={HI:.10f}", flush=True)

    doc = {}
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON, encoding="utf-8") as f:
            doc = json.load(f)
    doc.update({
        "protocol": {
            "warmup_cycles": 64,
            "warmup": "W true cycles before SP, then only own predictions",
            "stop": "end of the true series",
            "eol_ah": EOL_AH, "rated_ah": RATED,
            "horizons": list(HORIZONS), "runs": args.runs, "sps": args.sps,
            "metric_space": "common = our model's train-cell min--max "
                            "(Ah = common*(HI-LO) + LO)",
            "std_ddof": STD_DDOF,
            "first_step_tol": FIRST_STEP_TOL,
        },
        "common_space": {"lo": LO, "hi": HI,
                         "src": "checkpoints/per_sp/calce/SP300_seed1.pt"},
        "reference": REF_TF,
        "notes": [
            "k-horizon MAEs are prefixes of ONE rollout per run, not separate "
            "runs.",
            "Full-span MAE/R2 come from different span lengths per SP "
            "(583/483/383 pf, 582/482/382 rm, 581/481/381 ours) and are NOT "
            "comparable across SPs; only MAE@k is pooled across SPs.",
            "A run whose predicted curve never crosses the 0.77 Ah threshold "
            "contributes no AE; it is counted separately, never imputed.",
        ],
    })
    if args.models == "baselines":
        series_by_model = {"pf": None, "rm": None}
        import ar_rollout_baselines as arb
        series_by_model = {"pf": arb.official_calce(), "rm": arb.ours_calce()}
        sec = run_baselines(LO, HI, series_by_model, args.sps, args.runs)
        doc.setdefault("models", {}).update(sec)
    else:
        sec = run_ours(LO, HI, args.sps, args.runs)
        doc.setdefault("models", {}).update(sec)
    doc["verification"] = verify(doc["models"])

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
