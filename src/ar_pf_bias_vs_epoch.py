"""Does the PatchFormer absolute-target run's one-step bias shrink with training?

Two explanations were left unseparated for why the CALCE absolute-target
PatchFormer diverges upward under autoregressive rollout while its z-score
counterpart freezes (docs/ar_freeze_findings.md section 7):

  A  the run stopped early -- EarlyStopping picked epoch 25, against 56-88
     for the z-score runs, while train loss was still falling
  B  the absolute target itself is worse -- a global TorchNormalizer target
     changes what SMAPE weights, independently of how long it trains

`run_pf_abs.py --no-early-stop` keeps a checkpoint every 10 epochs, so the
answer is a curve rather than a single endpoint:

  * teacher-forced one-step residual (pred - true) over the launch region,
    the quantity the AR feedback integrates;
  * teacher-forced MAE / R2 / AE, i.e. whether the model itself improves;
  * the k=100 autoregressive rollout -- crossing and final value.

If the residual decays toward the z-score run's magnitude and the rollout
stops running away, it was (A).  If the residual is flat in the epoch, it is
(B).

Read-only: loads checkpoints, writes nothing.

    D:/anaconda/envs/patchformer/python.exe src/ar_pf_bias_vs_epoch.py
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import ar_abs_pf_rollout as A          # noqa: E402
import ar_ktable_calce as K            # noqa: E402
import ar_rollout_baselines as arb     # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# the region the AR rollout is seeded from, and where the bias does its work
LAUNCH = (500, 540)
KS = (100,)


def epoch_of(path: str) -> int:
    """Epoch number, or -1 for `last.ckpt` (a duplicate of the final epoch).

    Matches Lightning's own `epoch=NNN-step=NNN.ckpt` naming as well as the
    `epepoch=NNN.ckpt` this run produces -- the `{epoch:03d}` in
    ModelCheckpoint's filename template is not substituted, so Lightning
    appends its own `epoch=NNN` instead.
    """
    m = re.search(r"epoch=(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="nostop")
    ap.add_argument("--sp", type=int, default=500)
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--ks", type=int, nargs="+", default=list(KS))
    args = ap.parse_args()

    cfg = A.R.prepare_cfg("calce")
    cfg["_ds"] = "calce"
    out_dir = f'{cfg["out_dir"]}_{args.suffix}'
    run_dir = os.path.join(A.R.REPO, out_dir, cfg["test"], "PatchFormer",
                           f"SP{args.sp}", f"run{args.run}")
    # Lightning appends a `checkpoints/` subdirectory when the callback keeps
    # its default dirpath, and does NOT when dirpath is given explicitly
    # (which --no-early-stop does), so accept either layout.
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        ckpt_dir = run_dir
    ckpts = sorted((f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")),
                   key=epoch_of)
    ckpts = [f for f in ckpts if epoch_of(f) >= 0]   # drop `last.ckpt` dupe
    if not ckpts:
        raise SystemExit(f"no checkpoints in {ckpt_dir}")

    series = arb.official_calce()
    _, _, df_all, minv, maxv = arb.build_frames(series, args.sp)
    cyc = df_all["Cycle"].values
    cap_mm = df_all["Capacity"].values.astype(np.float64)
    lo, hi = K.common_ref()

    def to_ah(q_mm):
        return np.asarray(q_mm) * (maxv - minv) * cfg["rated"] \
            + minv * cfg["rated"]

    print("=" * 100)
    print(f"  PatchFormer ABSOLUTE-TARGET, CALCE SP{args.sp} run{args.run}  "
          f"(dir {out_dir})")
    print(f"  launch region {LAUNCH[0]}-{LAUNCH[1]}; z-score reference there: "
          f"+0.00245 Ah, 73% positive")
    print("=" * 100)
    hdr = (f"  {'ckpt':<12}{'TF MAE':>9}{'TF R2':>8}{'AE':>4}"
           f"{'bias(launch)':>14}{'pos%':>7}{'bias(all)':>11}"
           + "".join(f"{'k=' + str(k) + ' MAE':>10}{'cross':>7}{'final':>8}"
                     for k in args.ks))
    print(hdr)
    print("-" * len(hdr))

    from ModelsModify.PatchFormer import PatchFormerNetModel
    for fn in ckpts:
        model = PatchFormerNetModel.load_from_checkpoint(
            os.path.join(ckpt_dir, fn)).to(DEV).eval()
        yt, yp, man, diag = A.teacher_forced_abs(model, cfg, df_all, args.sp)
        flt = df_all.loc[df_all["Cycle"] >= args.sp, "target"].values \
            * cfg["rated"]
        n = min(len(flt), len(yp))
        yt, yp = yt[:n], yp[:n]
        err = yp - yt
        i0, i1 = LAUNCH[0] - args.sp, LAUNCH[1] - args.sp
        r = err[i0:i1 + 1]
        bias_l = float(r.mean())
        pos = 100.0 * float((r > 0).mean())
        _rr, _rp, ae, _re = A.R.rul_value_error(
            yt, yp, threshold=cfg["rated"] * cfg["eol_frac"])
        tf_mae = float(np.mean(np.abs(err)))
        tf_r2 = float(1 - np.sum((yt - yp) ** 2)
                      / (np.sum((yt - yt.mean()) ** 2) + 1e-6))
        row = (f"  {fn[:12]:<12}{tf_mae:>9.5f}{tf_r2:>8.4f}{ae:>4}"
               f"{bias_l:>14.5f}{pos:>6.0f}%{err.mean():>11.5f}")
        for k in args.ks:
            start = K.horizon_points((k,), 640)[str(k)]["T"] + 1
            i_start = int(np.where(cyc == start)[0][0])
            full = A.rollout_abs(model, cap_mm, i_start, cfg,
                                 diag["target_scale_centre"],
                                 diag["target_scale_scale"],
                                 diag["scaler_mean"], diag["scaler_scale"],
                                 minv, maxv,
                                 max_steps=len(cap_mm) - i_start)
            pv_c = (to_ah(full[:k]) - lo) / (hi - lo)
            tv_c = (to_ah(cap_mm[i_start:i_start + k]) - lo) / (hi - lo)
            cross = K.crossing_cycle(full, K.EOL_AH, start)
            row += (f"{float(np.mean(np.abs(pv_c - tv_c))):>10.5f}"
                    f"{str(cross):>7}{float(full[-1]):>8.4f}")
        print(row, flush=True)
        del model
        torch.cuda.empty_cache()

    print()
    print("  bias(launch) = mean(pred - true) over the launch region, in Ah.")
    print("  The AR feedback integrates exactly this.  If it decays toward")
    print("  the z-score run's +0.0025 it was undertraining; if it is flat")
    print("  in the epoch, the absolute target itself is the cause.")


if __name__ == "__main__":
    main()
