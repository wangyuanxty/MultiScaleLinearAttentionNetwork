"""AR (self-fed) rollout of the two self-run baselines on CALCE.

Table A's trajectory is teacher-forced: every input window is sliced out of
the ground-truth capacity series, so each point is a one-step prediction
anchored to real data.  This script measures the other regime -- from the
starting point SP onward the model sees only its own predictions, so error
compounds as it would in deployment.

Protocol copied verbatim from src/test_ar_rollout.py (the same measurement
for our own model), so the three land in one table:

  * warm-up = the W true cycles immediately before SP;
  * one step at a time: feed the window, take the one-step prediction,
    append it, slide the window by ONE cycle, feed no ground truth after SP;
  * run to the end of the true series, then find the first downward
    threshold crossing in the predicted series (test_ar_rollout.first_crossing);
  * MAE / R2 over [SP, end], AE = |true crossing - predicted crossing|.

The one thing that cannot be copied verbatim is the decoding: our GDN model
emits a z-score against its input window and so does PatchFormer/RUL-Mamba,
but the baselines' per-window normaliser is pytorch_forecasting's
EncoderNormalizer, whose center/scale are the mean/std of the ENCODER
window's target.  `_step` therefore decodes each baseline with its own
per-window statistics -- the same math its official eval uses -- and the
`--verify` pass proves that by reproducing the official `predict()` output
value for value.

Run with the `patchformer` conda env:
    python src/ar_rollout_baselines.py --model both --sps 300 400 500
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch

PROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_PF = os.path.join(PROOT, "reference_repos", "ref_patchformer")
REF_RM = os.path.join(PROOT, "reference_repos", "ref_rul_mamba")
SERIES_CACHE = os.path.join(PROOT, "checkpoints", "data_cache",
                            "load_series_calce.pkl")
OFFICIAL_CACHE = os.path.join(PROOT, "checkpoints", "data_cache",
                              "ar_rollout_calce_official.pkl")
OUT_JSON = os.path.join(PROOT, "src", "results", "ar_rollout_baselines.json")

TEST_CELL = "CS2_35"
GID = {"CS2_35": 0, "CS2_36": 1, "CS2_37": 2, "CS2_38": 3}
RATED, SEQL, EOL_AH, EOL_FRAC = 1.1, 64, 0.77, 0.7
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-6

torch.serialization.add_safe_globals(["numpy.core.multiarray.scalar"])
_orig_load = torch.load


def _patched_load(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_load(*a, **kw)


torch.load = _patched_load


# ── data ────────────────────────────────────────────────────────────────
def official_calce() -> dict:
    """The CALCE series *PatchFormer* was trained on.

    Upstream `CALCEDataPreProcess.BatteryDataRead` (drop_outlier + cycle
    renumbering).  This is NOT the series RUL-Mamba and our own model use:
    run_rm_extra.py builds the RUL-Mamba cache from our `load_series` pkl,
    and the two disagree by a few cycles / up to 0.18 Ah at a handful of
    indices.  Cached, because the xlsx parse takes minutes.
    """
    if os.path.exists(OFFICIAL_CACHE):
        with open(OFFICIAL_CACHE, "rb") as f:
            return pickle.load(f)
    cwd = os.getcwd()
    os.chdir(REF_PF)
    sys.path.insert(0, REF_PF)
    try:
        import CALCEDataPreProcess as mod
        data = mod.BatteryDataRead(list(GID), "data/CALCE data/")
        out = {n: g[["Cycle", "Capacity"]].to_numpy(dtype=np.float64)
               for n, g in data.items()}
    finally:
        os.chdir(cwd)
    with open(OFFICIAL_CACHE, "wb") as f:
        pickle.dump(out, f)
    return out


def ours_calce() -> dict:
    """Our `load_series` CALCE series -- what RUL-Mamba (and Ours) use.

    Same construction as run_rm_extra.make_cache_npy: Cycle 1..N, capacity
    in Ah.
    """
    with open(SERIES_CACHE, "rb") as f:
        caps, tr, te, W, sps, eol = pickle.load(f)
    return {n: np.stack([np.arange(1, len(a) + 1), np.asarray(a, np.float64)],
                        axis=1) for n, a in caps.items()}


def build_frames(series: dict, sp: int):
    """Official per-SP frames.

    Identical in run_pf_nasa_adapted.build_dfs and run_rm_extra.make_data_process:
    train = every other cell in full + the test cell's cycles before SP;
    min--max over that train part; test frame truncated at Cycle >= SP - seql.
    Returns (df_train, df_test, df_all, minv, maxv).
    """
    import pandas as pd

    parts = []
    for name, arr in series.items():
        r = pd.DataFrame({"BatteryName": name, "Cycle": arr[:, 0],
                          "Capacity": arr[:, 1]})
        if name == TEST_CELL:
            r = r[r["Cycle"] < sp]
        parts.append(r)
    df_train = pd.concat(parts).reset_index(drop=True)
    df_all = pd.DataFrame({"BatteryName": TEST_CELL,
                           "Cycle": series[TEST_CELL][:, 0],
                           "Capacity": series[TEST_CELL][:, 1]})
    for d in (df_train, df_all):
        d["Capacity"] = d["Capacity"] / RATED
        d["target"] = d["Capacity"]
        d["time_idx"] = d["Cycle"].map(lambda x: int(x - 1))
        d["group_id"] = d["BatteryName"].map(GID)
    minv = float(df_train["Capacity"].min())
    maxv = float(df_train["Capacity"].max())
    for d in (df_train, df_all):
        d["Capacity"] = (d["Capacity"] - minv) / (maxv - minv)

    def fin(d):
        d = d.drop(["BatteryName"], axis=1)
        d["idx"] = range(len(d))
        d.set_index("idx", inplace=True)
        return d

    df_train = fin(df_train)
    df_all = fin(df_all)
    df_test = df_all.loc[df_all["Cycle"] >= sp - SEQL,
                         ["time_idx", "group_id", "Cycle", "Capacity",
                          "target"]].copy()
    df_test["idx"] = range(len(df_test))
    df_test.set_index("idx", inplace=True)
    return df_train, df_test, df_all, minv, maxv


def mk_dataset(df):
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data.encoders import EncoderNormalizer

    return TimeSeriesDataSet(
        df, time_idx="time_idx", target="target", group_ids=["group_id"],
        min_encoder_length=SEQL, max_encoder_length=SEQL,
        min_prediction_length=1, max_prediction_length=1,
        time_varying_known_reals=["Capacity"],
        time_varying_unknown_reals=["target"],
        target_normalizer=EncoderNormalizer(), add_encoder_length=False)


# ── models ──────────────────────────────────────────────────────────────
def stored_rm_preds(sp: int, run: int):
    """y_pred of repeat `run` at SP, as saved by the official trainer.

    `Results/CALCE_Univariable_RULMamba_CS2_35.pth` is the results payload
    (not a model checkpoint) and holds every repeat's teacher-forced
    trajectory -- an independent check that our reproduction is the same
    model the paper's row came from.
    """
    path = os.path.join(REF_RM, "Results",
                        "CALCE_Univariable_RULMamba_CS2_35.pth")
    if not os.path.exists(path):
        return None
    payload = _orig_load(path, map_location="cpu", weights_only=False)
    try:
        return np.asarray(payload["predictions"][f"SP{sp}"][run - 1],
                          dtype=np.float64)
    except (KeyError, IndexError, TypeError):
        return None


def load_pf(sp: int, run: int):
    """PatchFormer checkpoint of run `run` at start point `sp`."""
    ckpt_dir = os.path.join(
        REF_PF, f"results_CALCE_RUL_prediction_sl_64", TEST_CELL,
        "PatchFormer", f"SP{sp}", f"run{run}", "checkpoints")
    ckpts = [os.path.join(ckpt_dir, f) for f in sorted(os.listdir(ckpt_dir))
             if f.endswith(".ckpt")]
    if not ckpts:
        raise FileNotFoundError(ckpt_dir)
    os.chdir(REF_PF)
    sys.path.insert(0, REF_PF)
    from ModelsModify.PatchFormer import PatchFormerNetModel
    model = PatchFormerNetModel.load_from_checkpoint(ckpts[0]).to(DEV).eval()
    return model, ckpts[0]


def load_rm(sp: int, run: int):
    """RUL-Mamba best checkpoint of repeat `run` at start point `sp`."""
    ckpt_dir = os.path.join(
        REF_RM, "Outputs", "CALCE", "Univariable", "RULMamba", f"Repeat_{run}",
        f"Start_Point_{sp}", "Checkpoints")
    ckpts = [os.path.join(ckpt_dir, f) for f in sorted(os.listdir(ckpt_dir))
             if f.endswith(".ckpt")]
    if not ckpts:
        raise FileNotFoundError(ckpt_dir)
    os.chdir(REF_RM)
    sys.path.insert(0, REF_RM)
    from Models.RULMamba import RULMambaNetModel
    model = RULMambaNetModel.load_from_checkpoint(ckpts[0]).to(DEV).eval()
    return model, ckpts[0]


# ── shared decode ───────────────────────────────────────────────────────
def net_out(model, x_sc: np.ndarray, name: str = "pf") -> np.ndarray:
    """Raw network output (a per-window z-score), no de-normalisation.

    x_sc is the tensor the model actually consumes, i.e.
    `encoder_cont[:, :, :-1]`: the `Capacity` known real AFTER
    pytorch_forecasting standardised it against the frame the dataset was
    built from (mean_/scale_ of `ds.scalers`).  RULMamba.forward takes the
    decoder input explicitly (None = the one-step head).
    """
    x = torch.tensor(np.asarray(x_sc), dtype=torch.float32, device=DEV)
    with torch.no_grad():
        z = (model.network(x_enc=x, x_dec=None) if name == "rm"
             else model.network(x))
    return z.detach().cpu().numpy().reshape(len(x_sc))


def scale_series(cap_mm, sc_m: float, sc_s: float):
    """min--max Capacity -> the network's known real."""
    return (np.asarray(cap_mm, dtype=np.float64) - sc_m) / sc_s


def decode(z: np.ndarray, x_sc: np.ndarray, minv: float, maxv: float,
           sc_m: float, sc_s: float):
    """EncoderNormalizer decode: z * std(encoder target) + mean(encoder target).

    The encoder target is `Capacity / rated`, i.e. the min--max normalised
    window mapped back with the same affine map the frame was built with;
    the network's input is that column times a StandardScaler, so undo that
    first.  Scale is the sample std (torch.std / unbiased), centre the mean.
    """
    cap_mm = np.asarray(x_sc).reshape(len(x_sc), -1) * sc_s + sc_m
    tgt = cap_mm * (maxv - minv) + minv
    return z * tgt.std(axis=1, ddof=1) + tgt.mean(axis=1)


def first_crossing(series: np.ndarray, thr: float) -> int:
    """First downward crossing index, or -1 (test_ar_rollout convention)."""
    for i in range(len(series) - 1):
        if series[i] >= thr > series[i + 1]:
            return i
    return -1


def rul_value_error(y_true, y_pred, threshold):
    """Verbatim from RUL_Prediction_PatchFormer_NASA.py / the RUL-Mamba trainer."""
    true_re, pred_re = len(y_true), 0
    for i in range(len(y_true) - 1):
        if y_true[i] <= threshold >= y_true[i + 1]:
            true_re = i - 1
            break
    for i in range(len(y_pred) - 1):
        if y_pred[i] <= threshold:
            pred_re = i - 1
            break
    return (true_re + 1, pred_re + 1, abs(true_re - pred_re),
            min(1.0, abs(true_re - pred_re) / (true_re + 1)))


# ── the three passes ────────────────────────────────────────────────────
def teacher_forced(model, df_all, minv, maxv, sp, i_sp, name="pf"):
    """Official `predict()` over the test frame + the same trajectory decoded
    by hand (the decode the rollout uses).  Returns (y_true, y_pred, manual,
    official, x_all, n), all in Ah except the two arrays named _mm."""
    df_test = df_all.loc[df_all["Cycle"] >= sp - SEQL,
                         ["time_idx", "group_id", "Cycle", "Capacity",
                          "target"]].copy()
    df_test["idx"] = range(len(df_test))
    df_test.set_index("idx", inplace=True)
    ds = mk_dataset(df_test)
    dl = ds.to_dataloader(train=False, batch_size=256, num_workers=0)
    sc = ds.scalers["Capacity"]
    sc_m, sc_s = float(sc.mean_[0]), float(sc.scale_[0])

    official, manual, x_all, ts_all = [], [], [], []
    for batch in dl:
        # pytorch_forecasting dataloaders yield (x, y) tuples
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        x_dev = {k: (v.to(DEV) if torch.is_tensor(v) else v)
                 for k, v in x.items()}
        with torch.no_grad():
            official.append(model(x_dev)["prediction"]
                            .detach().cpu().numpy().reshape(-1))
        xe = x["encoder_cont"][:, :, :-1].detach().cpu().numpy()
        x_all.append(xe)
        ts_all.append(x["target_scale"].detach().cpu().numpy())
        manual.append(decode(net_out(model, xe, name), xe, minv, maxv,
                            sc_m, sc_s))
    official = np.concatenate(official)
    manual = np.concatenate(manual)
    x_all = np.concatenate(x_all)

    # the hand decode must reproduce the official one value for value, and
    # the window it decodes must be the true Capacity window around SP
    tr = x_all[:, :, 0]
    allc = df_all["Capacity"].values
    expect = scale_series(
        np.lib.stride_tricks.sliding_window_view(allc, SEQL)[
            i_sp - SEQL:i_sp - SEQL + len(tr)], sc_m, sc_s)
    ok_x = bool(np.allclose(tr, expect, atol=1e-5))
    ts = np.concatenate(ts_all).reshape(len(tr), -1)
    tgt = (tr * sc_s + sc_m) * (maxv - minv) + minv
    centre_err = float(np.abs(ts[:, 0] - tgt.mean(axis=1)).max())
    scale_err = float(np.abs(ts[:, 1] - tgt.std(axis=1, ddof=1)).max())

    actuals = df_all.loc[df_all["Cycle"] >= sp, "target"].values
    n = min(len(actuals), len(official))
    y_true = actuals[:n] * RATED
    y_pred = official[:n] * RATED
    return (y_true, y_pred, manual[:n] * RATED, official[:n] * RATED,
            {"x_is_true_capacity": ok_x,
             "max_centre_err": centre_err, "max_scale_err": scale_err,
             "max_decode_err": float(np.abs(manual - official).max()),
             "scaler_mean": sc_m, "scaler_scale": sc_s,
             "n": n})


def rollout(model, df_all, minv, maxv, sp, sc_m, sc_s, max_steps=None,
            name="pf"):
    """Self-fed rollout in the model's own min--max space.

    Warm-up is the SEQL true cycles before SP -- exactly what the
    teacher-forced pass feeds there -- then nothing real is ever fed again.
    """
    cap_mm = df_all["Capacity"].values.astype(np.float64)
    cyc = df_all["Cycle"].values
    i_sp = int(np.where(cyc == sp)[0][0])
    window = list(cap_mm[i_sp - SEQL:i_sp])
    if len(window) < SEQL:
        raise ValueError(f"SP{sp}: only {len(window)} warm-up cycles")

    steps = max_steps if max_steps is not None else len(cap_mm) - i_sp
    preds = []
    for _ in range(steps):
        x = np.asarray(window[-SEQL:], dtype=np.float64)
        x_sc = scale_series(x, sc_m, sc_s).astype(np.float32)
        z = net_out(model, x_sc.reshape(1, SEQL, 1), name)
        p_rated = decode(z, x_sc.reshape(1, SEQL), minv, maxv,
                         sc_m, sc_s)[0]
        p_mm = (p_rated - minv) / (maxv - minv)
        preds.append(p_mm)
        window.append(p_mm)
    return np.array(preds), i_sp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="both", choices=["pf", "rm", "both"])
    ap.add_argument("--sps", type=int, nargs="+", default=[300, 400, 500])
    ap.add_argument("--run", type=int, default=1,
                    help="checkpoint repeat/run index (1 = the paper's run 1)")
    ap.add_argument("--verify", action="store_true",
                    help="extra diagnostics: decode vs predict, target_scale")
    args = ap.parse_args()

    series_by_model = {"pf": official_calce(), "rm": ours_calce()}
    for tag, s in series_by_model.items():
        print(f"{tag} series:")
        for n, a in s.items():
            print(f"  {n}: {len(a)} cycles, {a[:,1].min():.4f}-{a[:,1].max():.4f} Ah")

    ours_series = series_by_model["rm"]
    all_tr = np.concatenate([ours_series[c][:, 1] for c in
                             ["CS2_36", "CS2_37", "CS2_38"]])
    LO, HI = float(all_tr.min()), float(all_tr.max())
    print(f"our protocol: lo={LO:.4f} hi={HI:.4f} (range {HI-LO:.4f}) "
          f"W={SEQL} eol={EOL_AH}")

    out = {"series_len": {t: {n: len(a) for n, a in s.items()}
                          for t, s in series_by_model.items()},
           "ours_lo": LO, "ours_hi": HI, "run": args.run, "results": {}}

    for name, loader in (("pf", load_pf), ("rm", load_rm)):
        if args.model not in (name, "both"):
            continue
        series = series_by_model[name]
        print(f"\n================ {name.upper()} ================")
        for sp in args.sps:
            try:
                model, ckpt = loader(sp, args.run)
            except (FileNotFoundError, IndexError, OSError) as e:
                print(f"  SP{sp}: NO CHECKPOINT ({e})")
                continue
            df_train, df_test, df_all, minv, maxv = build_frames(series, sp)
            thr = (EOL_AH / RATED - minv) / (maxv - minv)
            i_sp = int(np.where(df_all["Cycle"].values == sp)[0][0])

            y_true, y_pred, man, off, diag = teacher_forced(
                model, df_all, minv, maxv, sp, i_sp, name)
            n = diag["n"]
            if args.verify:
                print(f"  [verify] window==true Capacity: "
                      f"{diag['x_is_true_capacity']}; "
                      f"max|decode-predict|={diag['max_decode_err']:.2e} "
                      f"(n={n}); EncoderNormalizer "
                      f"centre/scale err={diag['max_centre_err']:.2e}/"
                      f"{diag['max_scale_err']:.2e}")

            rr, rp, ae, re = rul_value_error(y_true, y_pred, EOL_AH)
            # same crossing rule as the rollout, so the two rows of the table
            # differ only in what the model is fed
            tf_re = first_crossing(y_true, EOL_AH)
            tf_pr = first_crossing(y_pred, EOL_AH)
            ae_same_rule = ((tf_re + 1) - (tf_pr + 1)
                            if (tf_re >= 0 and tf_pr >= 0) else None)
            tf_mae = float(np.mean(np.abs(y_true - y_pred)))
            tf_r2 = float(1 - np.sum((y_true - y_pred) ** 2)
                          / (np.sum((y_true - y_true.mean()) ** 2) + EPS))
            stored = stored_rm_preds(sp, args.run) if name == "rm" else None
            stored_err = (float(np.abs(stored[:len(y_pred)] - y_pred).max())
                          if stored is not None
                          and len(stored) >= len(y_pred) else None)

            preds, i_sp = rollout(model, df_all, minv, maxv, sp,
                                  diag["scaler_mean"], diag["scaler_scale"],
                                  name=name)
            tv = df_all["Capacity"].values[i_sp:i_sp + len(preds)]
            m = min(len(preds), len(tv))
            pv, tv = preds[:m], tv[:m]

            true_re = first_crossing(tv, thr)
            pred_re = first_crossing(pv, thr)
            tru = (true_re + 1) if true_re >= 0 else len(tv)
            pru = (pred_re + 1) if pred_re >= 0 else None

            # first-step check: rollout step 1 consumed the same true warm-up
            # window as the teacher-forced first sample
            first_tf = float(off[0])
            first_ro = float(pv[0]) * (maxv - minv) * RATED + minv * RATED
            tf_first_ah = float(man[0])
            step1 = abs(first_ro - tf_first_ah)

            # per-horizon error, the shape the reviewers asked for
            # (multi-step rollout at k = 25/50/100)
            hz = {k: float(np.mean(np.abs(pv[:k] - tv[:k])))
                  for k in (25, 50, 100) if len(pv) >= k}

            # native space: the model's own min--max (that is what the
            # rollout decodes into, and what test_ar_rollout.py reports for
            # ours -- the two spaces differ only in lo/hi, see below)
            mae_n = float(np.mean(np.abs(pv - tv)))
            r2_n = float(1 - np.sum((tv - pv) ** 2)
                         / (np.sum((tv - tv.mean()) ** 2) + EPS))
            c_true = first_crossing(tv, thr)
            c_pred = first_crossing(pv, thr)

            # strict cross-model space: our checkpoint's min--max, i.e. the
            # MAE column of the paper's table
            to_common = lambda q_ah: (q_ah - LO) / (HI - LO + EPS)  # noqa: E731
            ro_common = to_common(pv * (maxv - minv) * RATED + minv * RATED)
            tv_common = to_common(tv * (maxv - minv) * RATED + minv * RATED)
            mae_c = float(np.mean(np.abs(ro_common - tv_common)))

            rec = {
                "checkpoint": ckpt,
                "n_cycles_official": int(len(df_all)),
                "cycles_after_sp": int(len(df_all) - i_sp),
                "minv": minv, "maxv": maxv,
                "eol_mm": thr,
                "teacher_forced": {
                    "mae_ah": tf_mae, "r2": tf_r2,
                    "rul_real": int(rr), "rul_pred": int(rp), "ae": int(ae),
                    "re": float(re),
                    "true_crossing": (int(tf_re) + 1) if tf_re >= 0 else None,
                    "pred_crossing": (int(tf_pr) + 1) if tf_pr >= 0 else None,
                    "ae_first_crossing_rule": ae_same_rule,
                    "max_err_vs_stored_predictions": stored_err,
                },
                "rollout": {
                    "mae_mm": mae_n, "r2": r2_n, "mae_common": mae_c,
                    "mae_ah": float(np.mean(np.abs(
                        pv * (maxv - minv) * RATED + minv * RATED
                        - (tv * (maxv - minv) * RATED + minv * RATED)))),
                    "true_crossing": (int(c_true) + 1) if c_true >= 0 else None,
                    "pred_crossing": (int(c_pred) + 1) if c_pred >= 0 else None,
                    "ae": (abs((c_true + 1) - (c_pred + 1))
                           if (c_true >= 0 and c_pred >= 0) else None),
                    "first_pred_ah": first_ro,
                    "mae_at_k": hz,
                },
                "first_step_vs_teacher_forced": step1,
                "end_window_std_mm": (float(np.std(pv[-SEQL:]))
                                      if len(pv) >= SEQL else None),
            }
            out["results"].setdefault(name, {})[f"SP{sp}"] = rec

            print(f"  SP{sp}: ckpt={os.path.basename(ckpt)} "
                  f"({len(df_all)} cycles, {rec['cycles_after_sp']} after SP)")
            print(f"    teacher-forced (Ah)  MAE={tf_mae:.4f} R2={tf_r2:.4f} "
                  f"AE={ae} RUL={rr}/{rp} (first-crossing rule: "
                  f"{rec['teacher_forced']['true_crossing']}/"
                  f"{rec['teacher_forced']['pred_crossing']} -> "
                  f"{ae_same_rule})"
                  + (f"  [stored-pred err={stored_err:.2e}]"
                     if stored_err is not None else ""))
            print(f"    rollout   MAE_mm={mae_n:.4f} (common {mae_c:.4f}) "
                  f"R2={r2_n:.4f} "
                  f"true_cross={rec['rollout']['true_crossing']} "
                  f"pred_cross={rec['rollout']['pred_crossing']} "
                  f"AE={rec['rollout']['ae']}")
            print(f"    first-step |rollout - teacher-forced| = {step1:.3e} Ah; "
                  f"end-window std = {rec['end_window_std_mm']:.2e} (mm)")
            print("    MAE@k " + "  ".join(f"k={k}:{v:.4f}"
                                           for k, v in hz.items()))
            del model

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {OUT_JSON}")


if __name__ == "__main__":
    main()
