"""Autoregressive probe for the self-run PatchFormer on MIT.

PatchFormer's AR behaviour has been measured on CALCE and PANASONIC, but never
on MIT.  It matters here because MIT is where the training-cell count can be
varied, and this establishes the baseline the 39-cell run is compared against.

Everything shared with the CALCE measurement is imported from
ar_rollout_baselines.py (build_frames / teacher_forced / rollout / net_out), so
the launch protocol, the per-window EncoderNormalizer decode and the crossing
rule are identical -- only the cell directory and the series source differ.

The checkpoint was trained on the 10-cell cache split: 9 training cells
(the other cache cells) and batch2_cell5 held out.

Run with the `patchformer` conda env:
    D:/anaconda/envs/patchformer/python.exe src/ar_mit_pf_probe.py
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys

import numpy as np
import torch

SRC = os.path.dirname(os.path.abspath(__file__))
PROOT = os.path.dirname(SRC)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import ar_rollout_baselines as arb     # noqa: E402

TEST_CELL = "batch2_cell5"
RATED = 1.075
EOL_AH = 0.86
LAUNCH_BEFORE_EOL = 100
CKPT_ROOT = os.path.join(arb.REF_PF, "results_MIT_RUL_prediction_sl_64")


def mit_series():
    """PatchFormer's MIT source: the pipeline's cached per-cell series."""
    with open(os.path.join(PROOT, "checkpoints", "data_cache",
                           "load_series_mit.pkl"), "rb") as f:
        caps, _tr, _te, _w, _sps, _eol = pickle.load(f)
    return {n: np.stack([np.arange(1, len(a) + 1), np.asarray(a, np.float64)],
                        axis=1) for n, a in caps.items()}


def load_pf_mit(sp, run, root):
    ckpt_dir = os.path.join(root, TEST_CELL, "PatchFormer", f"SP{sp}",
                            f"run{run}", "checkpoints")
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")))
    if not ckpts:
        raise FileNotFoundError(ckpt_dir)
    os.chdir(arb.REF_PF)
    sys.path.insert(0, arb.REF_PF)
    from ModelsModify.PatchFormer import PatchFormerNetModel
    model = PatchFormerNetModel.load_from_checkpoint(ckpts[-1]).to(arb.DEV)
    return model.eval(), ckpts[-1]


def first_crossing(series, thr):
    for i in range(len(series) - 1):
        if series[i] >= thr > series[i + 1]:
            return i
    return -1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sp", type=int, default=300)
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--root", default="results_MIT_RUL_prediction_sl_64",
                    help="checkpoint root under ref_patchformer "
                         "(default: the 10-cell split; "
                         "results_MIT43_RUL_prediction_sl_64 is the 39-cell one)")
    args = ap.parse_args()
    root = os.path.join(arb.REF_PF, args.root)

    series = mit_series()
    # ar_rollout_baselines resolves TEST_CELL / GID / RATED from module globals
    # that are hard-coded to CALCE; point them at MIT before calling into it
    arb.TEST_CELL = TEST_CELL
    arb.GID = {c: i for i, c in enumerate(sorted(series))}
    arb.RATED = RATED
    arb.EOL_AH = EOL_AH
    arb.EOL_FRAC = EOL_AH / RATED
    print(f"  MIT series: {len(series)} cells, test={TEST_CELL}")
    model, ckpt = load_pf_mit(args.sp, args.run, root)
    os.chdir(PROOT)
    df_tr, df_te, df_all, minv, maxv = arb.build_frames(series, args.sp)
    i_sp = int(np.where(df_all["Cycle"].values == args.sp)[0][0])
    _, _, _, _, diag = arb.teacher_forced(model, df_all, minv, maxv,
                                          args.sp, i_sp, "pf")
    sc_m, sc_s = diag["scaler_mean"], diag["scaler_scale"]

    cap_mm = df_all["Capacity"].values.astype(np.float64)
    cyc = df_all["Cycle"].values
    thr = (EOL_AH / RATED - minv) / (maxv - minv)
    eol = first_crossing(cap_mm, thr) + 1

    t0 = eol - LAUNCH_BEFORE_EOL
    i0 = int(np.where(cyc == t0)[0][0])
    full, _ = arb.rollout(model, df_all, minv, maxv, t0, sc_m, sc_s,
                          max_steps=len(cap_mm) - i0, name="pf")
    full = np.asarray(full)
    truth = cap_mm[i0:i0 + len(full)]
    n = len(full)
    W = arb.SEQL
    win = list(cap_mm[i0 - W:i0])
    sigmas = []
    for j in range(n):
        sigmas.append(float(np.std(win[-W:])))
        win.append(full[j])
    floor = float(min(cap_mm[i - W:i].std() for i in range(W, len(cap_mm))))

    def mae(k):
        return (float(np.mean(np.abs(full[:k] - truth[:k])))
                if n >= k else float("nan"))

    print(f"  ckpt      {os.path.relpath(ckpt, arb.REF_PF)}")
    print(f"  decode    EncoderNormalizer (official), "
          f"max|decode-predict| = {diag['max_decode_err']:.2e}")
    print(f"  launch    cycle {t0}  ({LAUNCH_BEFORE_EOL} before EOL {eol}),"
          f"  rollout {n} steps, W={W}")
    print(f"  MAE@50    {mae(50):.5f}")
    print(f"  MAE@100   {mae(100):.5f}")
    print(f"  sigma end {sigmas[-1]:.5f}   floor {floor:.5f}"
          f"   ratio {sigmas[-1] / floor:.1f}"
          f"   ({'COLLAPSED' if sigmas[-1] < floor else 'in range'})")
    print(f"  final     {float(full[-1]):.4f}"
          f"   (truth at end {float(truth[-1]):.4f})")
    print()
    print("  ours, same cell / same launch / same metric:")
    print("    8 cells :  MAE@50 0.03167  MAE@100 0.10737  ratio 45.9  "
          "final 0.3827")
    print("    39 cells:  (see ar_mit43_probe.py)")


if __name__ == "__main__":
    main()
