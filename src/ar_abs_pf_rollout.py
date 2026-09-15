"""AR (self-fed) rollout of the ABSOLUTE-TARGET PatchFormer baseline on CALCE.

Why this cannot reuse src/ar_rollout_baselines.py's rollout: that one decodes
every baseline with its own EncoderNormalizer statistics --

    z-score checkpoints:   y_hat = z * std(encoder window) + mean(encoder window)

-- which is correct for the checkpoints trained with `EncoderNormalizer()` and
wrong for the one trained by src/run_pf_abs.py, whose `target_normalizer` is
`TorchNormalizer()`: a single global centre/scale fitted on the training
frame.  Its decode carries no window statistics at all:

    absolute checkpoint:   y_hat = z * scale + centre          (constant affine)

That difference is the whole point of the experiment.  The z-score decode
multiplies the network output by the input window's own spread, so once the
window fills with the model's own smooth output the multiplier collapses and
the prediction sticks at the window mean.  The absolute decode has no
multiplier to collapse -- whether that is enough to keep the decline alive is
what this script measures.

Everything else is imported, not rewritten:

    src/ar_rollout_baselines.py   official_calce / build_frames / net_out
    src/run_pf_abs.py             make_dataset  (the TorchNormalizer recipe)
    src/ar_ktable_calce.py        horizon_points / crossing_cycle / common_ref

Protocol identical to `ar_ktable_calce.py --model pf`, so the two rows differ
only in the checkpoint:

  * true EOL = cycle 640 (CS2_35's last cycle at/above 0.77 Ah)
  * launch at T = true_EOL - k, warm up on the W = 64 true cycles before the
    launch cycle, then feed only the model's own predictions and run on to the
    end of the series; first predicted cycle = T + 1
  * MAE over the k launch steps in the common space; AE = predicted crossing
    cycle - 640, signed, crossers only (never imputed)

Three gates are checked before any number is reported:

  G1  target_scale is the same for every batch   (global, not per-window)
  G2  the hand decode reproduces official predict() value for value
  G3  the teacher-forced trajectory reproduces the recorded ABS run
      (reference_repos/ref_patchformer/results/pf_calce_ABS_selfrun_CS2_35.json:
       MAE 0.0126 / R2 0.9897 / AE 0)

Writes only NEW paths -- nothing existing is touched:
    src/results/ar_ktable_calce_pfabs.json
    src/results/ar_traj_calce_pfabs.npz

Run with the `patchformer` conda env:
    D:/anaconda/envs/patchformer/python.exe src/ar_abs_pf_rollout.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import ar_ktable_calce as K            # noqa: E402
import ar_rollout_baselines as arb     # noqa: E402
import run_pf_abs as R                 # noqa: E402  (chdirs into ref_patchformer)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-6

OUT_JSON = os.path.join(SRC, "results", "ar_ktable_calce_pfabs.json")
TRAJ_NPZ = os.path.join(SRC, "results", "ar_traj_calce_pfabs.npz")
ABS_JSON = os.path.join(R.REPO, "results", "pf_calce_ABS_selfrun_CS2_35.json")

# The recorded teacher-forced ABS run this script must reproduce.
REF_TF_ABS = {"mae": 0.0126, "r2": 0.9897, "ae": 0}
G_TOL = 1e-4


# ── checkpoint ──────────────────────────────────────────────────────────
def load_abs(cfg: dict, sp: int, run: int):
    """Absolute-target PatchFormer checkpoint of run `run` at start point `sp`."""
    ckpt_dir = os.path.join(R.REPO, cfg["out_dir"], cfg["test"], "PatchFormer",
                            f"SP{sp}", f"run{run}", "checkpoints")
    ckpts = [os.path.join(ckpt_dir, f) for f in sorted(os.listdir(ckpt_dir))
             if f.endswith(".ckpt")]
    if not ckpts:
        raise FileNotFoundError(ckpt_dir)
    from ModelsModify.PatchFormer import PatchFormerNetModel
    model = PatchFormerNetModel.load_from_checkpoint(ckpts[0]).to(DEV).eval()
    return model, ckpts[0]


# ── teacher-forced pass with the ABSOLUTE decode ────────────────────────
def teacher_forced_abs(model, cfg, df_all, sp):
    """Official predict() + the hand absolute decode, in Ah.

    The only line that differs from ar_rollout_baselines.teacher_forced is the
    decode: `z * target_scale[1] + target_scale[0]` instead of the encoder
    window's own std/mean.
    """
    i_sp = int(np.where(df_all["Cycle"].values == sp)[0][0])
    df_test = df_all.loc[
        df_all["Cycle"] >= sp - cfg["seql"],
        ["time_idx", "group_id", "Cycle", "Capacity", "target"]
    ].copy().reset_index(drop=True)
    testing, dl = R.make_dataset(df_test, cfg["batch"], False, False,
                                 cfg["seql"])
    sc = testing.scalers["Capacity"]
    sc_m, sc_s = float(sc.mean_[0]), float(sc.scale_[0])

    official, manual, xs, tss = [], [], [], []
    for batch in dl:
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        x_dev = {k: (v.to(DEV) if torch.is_tensor(v) else v)
                 for k, v in x.items()}
        with torch.no_grad():
            official.append(model(x_dev)["prediction"]
                            .detach().cpu().numpy().reshape(-1))
        xe = x["encoder_cont"][:, :, :-1].detach().cpu().numpy()
        ts = x["target_scale"].detach().cpu().numpy().reshape(len(xe), -1)
        xs.append(xe)
        tss.append(ts)
        manual.append(arb.net_out(model, xe, "pf") * ts[:, 1] + ts[:, 0])
    official = np.concatenate(official)
    manual = np.concatenate(manual)
    xs = np.concatenate(xs)
    tss = np.concatenate(tss)

    # G1: a global normaliser must give every sample the same target_scale
    ts_centre_spread = float(tss[:, 0].max() - tss[:, 0].min())
    ts_scale_spread = float(tss[:, 1].max() - tss[:, 1].min())

    # G2: the hand decode must reproduce the official one value for value
    decode_err = float(np.abs(manual - official).max())

    # the window fed to the network must be the true Capacity around SP:
    # the dataloader's i-th sample covers df_all rows [i_sp-seql+i, i_sp+i)
    tr = xs[:, :, 0]
    expect = arb.scale_series(
        np.lib.stride_tricks.sliding_window_view(
            df_all["Capacity"].values, cfg["seql"])[
                i_sp - cfg["seql"]:i_sp - cfg["seql"] + len(tr)], sc_m, sc_s)
    x_is_true = bool(np.allclose(tr, expect, atol=1e-5))

    actuals = df_all.loc[df_all["Cycle"] >= sp, "target"].values
    n = min(len(actuals), len(official))
    y_true = actuals[:n] * cfg["rated"]
    y_pred = official[:n] * cfg["rated"]
    ma = manual[:n] * cfg["rated"]
    return y_true, y_pred, ma, {
        "scaler_mean": sc_m, "scaler_scale": sc_s,
        "target_scale_centre": float(tss[0, 0]),
        "target_scale_scale": float(tss[0, 1]),
        "ts_centre_spread": ts_centre_spread,
        "ts_scale_spread": ts_scale_spread,
        "decode_err": decode_err, "x_is_true_capacity": x_is_true, "n": n,
    }


# ── rollout with the ABSOLUTE decode ────────────────────────────────────
def rollout_abs(model, cap_mm, i_start, cfg, ts_c, ts_s, sc_m, sc_s,
                minv, maxv, max_steps=None):
    """Self-fed rollout in Ah.

    `cap_mm` is the min--max `Capacity` column the network consumes (scaled by
    sc_m/sc_s on the way in); the network emits the globally normalised target,
    which comes back through the constant affine `z * ts_s + ts_c`.
    """
    seql = cfg["seql"]
    window = list(cap_mm[i_start - seql:i_start])
    if len(window) < seql:
        raise ValueError(f"only {len(window)} warm-up cycles")

    steps = max_steps if max_steps is not None else len(cap_mm) - i_start
    preds = []
    for _ in range(steps):
        x = np.asarray(window[-seql:], dtype=np.float64)
        x_sc = ((x - sc_m) / sc_s).astype(np.float32)
        z = arb.net_out(model, x_sc.reshape(1, seql, 1), "pf")
        p_rated = z[0] * ts_s + ts_c                  # absolute: no window stats
        p_mm = (p_rated - minv) / (maxv - minv)
        preds.append(p_rated * cfg["rated"])          # Ah
        window.append(p_mm)
    return np.asarray(preds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="+", default=list(K.KS))
    ap.add_argument("--sp", type=int, default=500,
                    help="checkpoint start point (only SP500 exists for ABS)")
    ap.add_argument("--run", type=int, default=1)
    args = ap.parse_args()

    cfg = R.prepare_cfg("calce")
    cfg["_ds"] = "calce"
    caps, true_eol = K.calce_series()
    points = K.horizon_points(args.ks, true_eol)
    for key, p in points.items():
        if p["sp"] != args.sp:
            raise SystemExit(f"k={key} needs SP{p['sp']}; only SP{args.sp} "
                             f"exists for the absolute-target run")

    series = arb.official_calce()
    i_pf = K.first_crossing(series[K.TEST_CELL][:, 1], K.EOL_AH)
    assert i_pf + 1 == true_eol, (f"official series crosses at {i_pf + 1}, "
                                  f"load_series at {true_eol}")
    LO, HI = K.common_ref()

    print("=" * 88)
    print(f"  PatchFormer ABSOLUTE-TARGET  SP{args.sp} run{args.run}  "
          f"seql={cfg['seql']} rated={cfg['rated']}")
    print(f"  true EOL = cycle {true_eol} (threshold {K.EOL_AH} Ah); "
          f"common space lo={LO:.6f} hi={HI:.6f}")
    print("=" * 88, flush=True)

    _, _, df_all, minv, maxv = arb.build_frames(series, args.sp)
    cyc = df_all["Cycle"].values
    cap_mm = df_all["Capacity"].values.astype(np.float64)

    model, ckpt = load_abs(cfg, args.sp, args.run)
    print(f"  ckpt = {os.path.relpath(ckpt, R.REPO)}\n", flush=True)

    # ── gates ───────────────────────────────────────────────────────────
    y_true, y_pred, man, diag = teacher_forced_abs(model, cfg, df_all,
                                                   args.sp)
    rr, rp, ae, _re = R.rul_value_error(
        y_true, y_pred, threshold=cfg["rated"] * cfg["eol_frac"])
    tf_mae = float(np.mean(np.abs(y_true - y_pred)))
    tf_r2 = float(1 - np.sum((y_true - y_pred) ** 2)
                  / (np.sum((y_true - y_true.mean()) ** 2) + EPS))
    g1 = max(diag["ts_centre_spread"], diag["ts_scale_spread"]) < G_TOL
    g2 = diag["decode_err"] < G_TOL
    g3 = (abs(tf_mae - REF_TF_ABS["mae"]) / REF_TF_ABS["mae"] < 0.02
          and abs(tf_r2 - REF_TF_ABS["r2"]) < 0.005
          and ae == REF_TF_ABS["ae"])
    print("[gates]")
    print(f"  G1 target_scale global (spread {diag['ts_centre_spread']:.2e}/"
          f"{diag['ts_scale_spread']:.2e}); centre="
          f"{diag['target_scale_centre']:.6f} "
          f"scale={diag['target_scale_scale']:.6f} -> "
          f"{'PASS' if g1 else 'FAIL'}")
    print(f"  G2 hand decode == official predict  (max err "
          f"{diag['decode_err']:.2e}, windows are true Capacity: "
          f"{diag['x_is_true_capacity']}) -> {'PASS' if g2 else 'FAIL'}")
    print(f"  G3 teacher-forced reproduces the recorded ABS run: "
          f"MAE={tf_mae:.4f} (ref {REF_TF_ABS['mae']}) R2={tf_r2:.4f} "
          f"(ref {REF_TF_ABS['r2']}) AE={ae} (ref {REF_TF_ABS['ae']}) -> "
          f"{'PASS' if g3 else 'FAIL'}")
    print(f"  [teacher-forced] RUL={rr}/{rp}, n={diag['n']} cycles\n",
          flush=True)

    # ── rollouts ────────────────────────────────────────────────────────
    out = {"checkpoint": ckpt, "sp": args.sp, "run": args.run,
           "true_eol": true_eol, "threshold_ah": K.EOL_AH,
           "common_space": {"lo": LO, "hi": HI},
           "diag": diag,
           "gates": {"g1_target_scale_global": g1,
                     "g2_decode_matches_predict": g2,
                     "g3_teacher_forced_reproduces": g3},
           "teacher_forced": {"mae_ah": tf_mae, "r2": tf_r2,
                              "rul_real": int(rr), "rul_pred": int(rp),
                              "ae": int(ae), "mae_ref": REF_TF_ABS["mae"]},
           "points": {}}
    trajs: dict = {}

    def to_ah(q_mm):
        return np.asarray(q_mm) * (maxv - minv) * cfg["rated"] \
            + minv * cfg["rated"]

    for key in sorted(points, key=lambda x: int(x)):
        p = points[key]
        k, t = p["k"], p["T"]
        start = t + 1                       # first predicted cycle
        i_start = int(np.where(cyc == start)[0][0])
        t0 = time.time()
        full = rollout_abs(model, cap_mm, i_start, cfg,
                           diag["target_scale_centre"],
                           diag["target_scale_scale"],
                           diag["scaler_mean"], diag["scaler_scale"],
                           minv, maxv, max_steps=len(cap_mm) - i_start)
        if not np.all(np.isfinite(full)):
            raise SystemExit(f"k={k}: non-finite rollout")

        pv_ah = full[:k]
        tv_ah = to_ah(cap_mm[i_start:i_start + k])
        pv_c = (pv_ah - LO) / (HI - LO)
        tv_c = (tv_ah - LO) / (HI - LO)

        # first rollout step vs the teacher-forced prediction of the same cycle
        j = start - args.sp
        tf_first = float(man[j])
        dev = float(abs(pv_ah[0] - tf_first))

        pred_cross = K.crossing_cycle(full, K.EOL_AH, start)
        trajs[f"k{k}_s{args.run}"] = (start, full)

        out["points"][key] = {
            **p, "first_predicted_cycle": start,
            "first_step_dev_ah": dev,
            "first_step_within_tol": bool(dev <= K.FIRST_STEP_TOL),
            "teacher_forced_first_step": tf_first,
            "rollout": {
                "n_steps": int(len(full)),
                "mae_common": float(np.mean(np.abs(pv_c - tv_c))),
                "mae_ah": float(np.mean(np.abs(pv_ah - tv_ah))),
                "pred_crossing_cycle": pred_cross,
                "crossed": pred_cross is not None,
                "ae": None if pred_cross is None else int(pred_cross - true_eol),
                "last_ah": float(full[-1]),
                "end_window_std_ah": (float(np.std(full[-cfg["seql"]:]))
                                      if len(full) >= cfg["seql"] else None),
                "pred_rise_frac": (float(np.mean(np.diff(full) > 0))
                                   if len(full) > 1 else None),
            },
        }
        r = out["points"][key]["rollout"]
        print(f"  k={k:>3} (T={t}, start={start}): "
              f"MAE_common={r['mae_common']:.5f} "
              f"MAE={r['mae_ah'] * 1000:.2f}mAh cross={r['pred_crossing_cycle']} "
              f"AE={r['ae']} last={r['last_ah']:.4f}Ah d1={dev:.1e} "
              f"[{time.time() - t0:.0f}s]", flush=True)

    cycles = np.arange(1, len(caps[K.TEST_CELL]) + 1, dtype=np.int64)
    arrs = {"cycles": cycles, "true": caps[K.TEST_CELL]}
    for key, (start, vals) in trajs.items():
        a = np.full(len(cycles), np.nan)
        i0 = start - 1
        n = min(len(vals), len(cycles) - i0)
        a[i0:i0 + n] = vals[:n]
        arrs[f"pred_pfabs_{key}"] = a
    np.savez_compressed(TRAJ_NPZ, **arrs)

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n  trajectories -> {os.path.relpath(TRAJ_NPZ, ROOT)}")
    print(f"  metrics      -> {os.path.relpath(OUT_JSON, ROOT)}")
    if os.path.exists(ABS_JSON):
        with open(ABS_JSON, encoding="utf-8") as f:
            rec = json.load(f)["SP500"]["runs"][0]
        print(f"  recorded ABS run in {os.path.relpath(ABS_JSON, ROOT)}: "
              f"MAE={rec['mae']:.4f} R2={rec['r2']:.4f} AE={rec['ae']}")


if __name__ == "__main__":
    main()
