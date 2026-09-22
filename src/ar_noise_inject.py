"""Does measurement noise in the fed-back predictions break the AR freeze?

The reasoning being tested.  In an autoregressive rollout the window is
replaced, one value at a time, by the model's own output -- and those outputs
are smooth.  Real capacity measurements are not: they carry measurement noise
and regeneration wiggles.  So the window drifts away from anything the model
saw in training, and its response (the z it emits) collapses towards zero.

Self-consistency says a steady decline needs z ~ -(W/2+1)*sqrt(12)/W ~= -1.73,
independent of the fade rate.  The measured z starts at -2.13 (below the
threshold, so the window's spread grows -- observed) and crosses the threshold
around step 40-80, after which the spread collapses (also observed).  If the
trigger is the window losing its noise, then adding noise back should keep z
below the threshold and keep the decline alive.

A deployment argument for the same intervention: a BMS measures capacity with
error, so a rollout that assumes noiseless future measurements is the
unrealistic one.

Noise scale is estimated from the data rather than guessed: take the launch
window, remove its linear trend, and use the residual std.

    python src/ar_noise_inject.py --dataset calce --ks 50
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

# noise scale multipliers: 0 = the current (noiseless) behaviour
LEVELS = (0.0, 0.5, 1.0, 2.0, 4.0)


def detrended_noise(window):
    """Residual std of a window after removing its best-fit line.

    That residual is the part of the signal that is not trend -- measurement
    noise plus wiggles -- and it is the scale the fed-back predictions are
    missing.
    """
    x = np.arange(len(window), dtype=float)
    a, b = np.polyfit(x, window, 1)
    return float(np.std(window - (a * x + b)))


def first_crossing(series, thr):
    for i in range(len(series) - 1):
        if series[i] >= thr > series[i + 1]:
            return i
    return -1


def build(ds, seed):
    ck = torch.load(os.path.join(CKPT, ds, f"SP{DS[ds]['sp']}_seed{seed}.pt"),
                    map_location=DEV, weights_only=False)
    model = build_gdn_model(multiscale=True, stage_query=True, input_dim=1,
                            window_size=ck["W"], output_len=1,
                            readout="last").to(DEV)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def run(ds, k, steps, seeds):
    cfg = DS[ds]
    caps = load_caps(ds)
    tr = np.concatenate([caps[c] for c in cfg["train"]])
    lo, hi = tr.min(), tr.max()
    seq = (caps[cfg["test"]] - lo) / (hi - lo)
    cycles = np.arange(1, len(seq) + 1)
    thr = (cfg["thr"] - lo) / (hi - lo)
    i0 = int(np.where(cycles == cfg["eol"] - k)[0][0])

    out = {lv: dict(mae=[], cross=[], final=[]) for lv in LEVELS}
    for s in seeds:
        model, ck = build(ds, s)
        W = ck["W"]
        warm = seq[i0 - W:i0]
        ns = detrended_noise(warm)
        rng = np.random.default_rng(1000 + s)
        for lv in LEVELS:
            win = list(warm)
            preds = []
            with torch.no_grad():
                for _ in range(steps):
                    x = np.asarray(win[-W:], dtype=np.float32)
                    wm, ws = float(x.mean()), float(x.std()) + EPS
                    t = torch.tensor(x).reshape(1, W, 1).to(DEV)
                    p = float(model(t).item()) * ws + wm
                    preds.append(p)
                    # The appended value stands in for a future measurement,
                    # so it should look like one: prediction plus noise.
                    win.append(p + (lv * ns) * rng.standard_normal())
            preds = np.array(preds)
            truth = seq[i0:i0 + len(preds)]
            n = min(len(preds), k)
            out[lv]["mae"].append(float(np.mean(np.abs(preds[:n] - truth[:n]))))
            out[lv]["final"].append(float(preds[-1]))
            xi = first_crossing(preds, thr)
            out[lv]["cross"].append(0 if xi < 0 else int(cycles[i0 + xi]) - cfg["eol"])

    print()
    print("=" * 80)
    print(f"  {ds.upper()}  k={k}  W={DS[ds]['W']}  noise x detrended residual "
          f"of the launch window")
    print(f"  launch cycle {cfg['eol'] - k}, rolled {steps} steps, "
          f"{len(seeds)} seeds, true EOL = {cfg['eol']}")
    print("=" * 80)
    print(f"  {'noise x':>8} {'MAE':>17} {'crossed':>9} "
          f"{'AE (cycles)':>16} {'final (median)':>15}")
    for lv in LEVELS:
        r = out[lv]
        # AE = |predicted crossing - true EOL| (DeltaCycle_Multi-Scale_Linear_Attention.tex); a run
        # that never crosses contributes no AE and is counted separately.
        aes = [abs(c) for c in r["cross"] if c != 0]
        nc = len(aes)
        ae_txt = ("--" if nc == 0 else
                  f"{np.mean(aes):.0f}+/-{np.std(aes, ddof=1):.0f}" if nc > 1
                  else f"{aes[0]}")
        print(f"  {lv:>8.1f} {np.mean(r['mae']):>9.5f}+/-{np.std(r['mae'], ddof=1):<7.5f} "
              f"{nc:>5}/{len(seeds)} {ae_txt:>16} {np.median(r['final']):>15.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="calce", choices=list(DS))
    ap.add_argument("--ks", type=int, nargs="+", default=[50])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    args = ap.parse_args()
    for k in args.ks:
        run(args.dataset, k, args.steps, tuple(args.seeds))


if __name__ == "__main__":
    main()
