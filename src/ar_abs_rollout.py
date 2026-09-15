"""Autoregressive rollout of an absolute-target model.

Same rollout as everywhere else in this directory -- warm up on the W true
cycles before the launch, then feed only the model's own output -- with one
difference that is the whole point:

    z-score model:      y_hat = model(x) * std(x) + mean(x)
    absolute model:     y_hat = model(x)

The z-score decode multiplies the model's output by the input window's own
standard deviation.  Under rollout the window fills with the model's own
(smooth) output, that std collapses 34x, and the prediction sticks at the
window mean.  An absolute-target model has no such multiplication, so the
mechanism is absent by construction -- this script checks whether that is
enough to keep the decline alive.

Reported, per horizon:
  MAE       trajectory error over the first k steps, normalised capacity
  crossed   how many of the seeds produced a threshold crossing at all
  AE        |predicted crossing - true EOL|, cycles

    python src/ar_abs_rollout.py --dataset calce --ks 50 100
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ar_grow_window import DS, load_caps  # noqa: E402
from gdn_model import build_gdn_model  # noqa: E402

EPS = 1e-6
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "checkpoints", "per_sp")


def first_crossing(series, thr):
    for i in range(len(series) - 1):
        if series[i] >= thr > series[i + 1]:
            return i
    return -1


def run(ds, ks, steps, seeds, ckpt_name):
    cfg = DS[ds]
    caps = load_caps(ds)
    tr = np.concatenate([caps[c] for c in cfg["train"]])
    lo, hi = tr.min(), tr.max()
    seq = (caps[cfg["test"]] - lo) / (hi - lo)
    cycles = np.arange(1, len(seq) + 1)
    thr = (cfg["thr"] - lo) / (hi - lo)

    print()
    print("=" * 84)
    print(f"  {ds.upper()}   ABSOLUTE-TARGET model   ckpt={ckpt_name}")
    print(f"  EOL={cfg['eol']}  thr={cfg['thr']} Ah (norm {thr:.4f})  "
          f"W={cfg['W']}  {len(seeds)} seeds")
    print("=" * 84)
    print(f"  {'k':>5} {'MAE':>18} {'crossed':>9} {'AE (cycles)':>18} "
          f"{'final (median)':>16}")

    for k in ks:
        i0 = int(np.where(cycles == cfg["eol"] - k)[0][0])
        maes, aes, finals = [], [], []
        for s in seeds:
            path = os.path.join(CKPT, ds, ckpt_name.format(seed=s))
            ck = torch.load(path, map_location=DEV, weights_only=False)
            W = ck["W"]
            model = build_gdn_model(multiscale=True, stage_query=True,
                                    input_dim=1, window_size=W, output_len=1,
                                    readout="last").to(DEV)
            model.load_state_dict(ck["state_dict"])
            model.eval()
            win = list(seq[i0 - W:i0])
            preds = []
            n_steps = len(seq) - i0
            if steps:
                n_steps = min(n_steps, steps)
            with torch.no_grad():
                for _ in range(n_steps):
                    x = np.asarray(win[-W:], dtype=np.float32)
                    t = torch.tensor(x).reshape(1, W, 1).to(DEV)
                    p = float(model(t).item())   # absolute: no decode
                    preds.append(p)
                    win.append(p)
            preds = np.array(preds)
            truth = seq[i0:i0 + len(preds)]
            n = min(len(preds), k)
            maes.append(float(np.mean(np.abs(preds[:n] - truth[:n]))))
            finals.append(float(preds[-1]))
            xi = first_crossing(preds, thr)
            if xi >= 0:
                aes.append(abs(int(cycles[i0 + xi]) - cfg["eol"]))
        nc = len(aes)
        ae = ("--" if nc == 0 else
              f"{np.mean(aes):.0f}+/-{np.std(aes, ddof=1):.0f}" if nc > 1
              else f"{aes[0]}")
        print(f"  {k:>5} {np.mean(maes):>9.5f}+/-{np.std(maes, ddof=1):<7.5f} "
              f"{nc:>5}/{len(seeds)} {ae:>18} {np.median(finals):>16.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="calce", choices=list(DS))
    ap.add_argument("--ks", type=int, nargs="+", default=[50, 100])
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[11])
    ap.add_argument("--ckpt", default="SP500_seed{seed}.pt")
    args = ap.parse_args()
    run(args.dataset, args.ks, args.steps, tuple(args.seeds), args.ckpt)


if __name__ == "__main__":
    main()
