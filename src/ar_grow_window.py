"""Growing-window autoregressive rollout: never drop the oldest cycle.

The sliding rollout freezes.  The mechanism is structural: every step drops
the most extreme value in the window and appends one that sits near the window
mean, so the window's own std decays towards zero -- and the decode is
`y = z * std(window) + mean(window)`, so once the std is gone the model's
output is multiplied away and the series sticks at the window mean.

This variant keeps everything: the window starts at the W true cycles before
the launch and grows by one on every step, never discarding.  The true prefix
stays inside the window forever, so its spread puts a floor under
std(window) ~ std_true * sqrt(W / (W + N)) rather than letting it reach zero.

Caveat this measures: the model was trained on windows of exactly W.  Growing
the window is a length-shift the model has never seen, so the outcome is an
empirical question, not a foregone one -- which is the point.  A linear
attention model whose state is decoupled from context is exactly the
architecture that should tolerate it.

    python src/ar_grow_window.py --dataset calce
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gdn_model import build_gdn_model  # noqa: E402

EPS = 1e-6
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "checkpoints", "per_sp")

# dataset -> test cell, train cells, W, EOL threshold Ah, true EOL, SP used
DS = {
    "calce": dict(test="CS2_35", train=["CS2_36", "CS2_37", "CS2_38"],
                  W=64, thr=0.77, eol=640, sp=500),
    "panasonic": dict(test="Cell03", train=["Cell01", "Cell02"],
                      W=30, thr=2.12, eol=588, sp=400),
}

KS = (25, 50, 75, 100)
SEEDS = tuple(range(1, 11))

# Rollout length cap.  None runs to the end of the capacity series (~266-341
# steps for CALCE); a small value gives the same qualitative answer in a
# fraction of the time, which matters because the growing window costs
# O(L) per step with L rising on every step.
MAX_STEPS: int | None = None


def load_caps(ds):
    """Raw capacity series per cell, the same loaders the other scripts use."""
    if ds == "calce":
        from load_datasets import load_calce_cells_multivar
        caps_all, _, _ = load_calce_cells_multivar()
    else:
        from load_datasets import load_panasonic_cells
        caps_all = load_panasonic_cells()
    return {c: np.asarray(caps_all[c], dtype=np.float64) for c in caps_all}


def build(ds, seed):
    """One per-SP checkpoint and the model it belongs to."""
    ck = torch.load(os.path.join(CKPT, ds, f"SP{DS[ds]['sp']}_seed{seed}.pt"),
                    map_location=DEV, weights_only=False)
    model = build_gdn_model(multiscale=True, stage_query=True, input_dim=1,
                            window_size=ck["W"], output_len=1,
                            readout="last").to(DEV)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def first_crossing(series, thr):
    """First downward crossing index, or -1 (same rule as the other scripts)."""
    for i in range(len(series) - 1):
        if series[i] >= thr > series[i + 1]:
            return i
    return -1


def run(ds):
    cfg = DS[ds]
    eol = cfg["eol"]
    caps = load_caps(ds)
    tr = np.concatenate([caps[c] for c in cfg["train"]])
    lo, hi = tr.min(), tr.max()
    seq = (caps[cfg["test"]] - lo) / (hi - lo)
    cycles = np.arange(1, len(seq) + 1)
    # The series is normalised, so the EOL threshold has to be too -- passing
    # the raw Ah value finds no crossing at all (2.12 vs a series in 0..1).
    thr = (cfg["thr"] - lo) / (hi - lo)

    out = {}
    for k in KS:
        T = eol - k
        hits = np.where(cycles == T)[0]
        if len(hits) == 0:
            continue
        i0 = int(hits[0])
        maes, aes, finals = [], [], []
        n_never = 0
        for s in SEEDS:
            model, ck = build(ds, s)
            W = ck["W"]
            win = list(seq[i0 - W:i0])          # warm-up: TRUE cycles
            preds = []
            total = len(seq) - i0
            if MAX_STEPS is not None:
                total = min(total, MAX_STEPS)
            with torch.no_grad():
                for _ in range(total):
                    x = np.asarray(win, dtype=np.float32)   # grows, never pops
                    wm = float(x.mean())
                    ws = float(x.std()) + EPS
                    t = torch.tensor(x).reshape(1, len(x), 1).to(DEV)
                    p = float(model(t).item()) * ws + wm
                    preds.append(p)
                    win.append(p)
            preds = np.array(preds)
            truth = seq[i0:i0 + len(preds)]
            n = min(len(preds), k)
            maes.append(float(np.mean(np.abs(preds[:n] - truth[:n]))))
            finals.append(float(preds[-1]))
            xi = first_crossing(preds, thr)
            if xi < 0:
                n_never += 1
            else:
                aes.append(abs(int(T + xi) - eol))
        out[k] = dict(
            mae=float(np.mean(maes)), mae_std=float(np.std(maes, ddof=1)),
            ae=float(np.mean(aes)) if aes else float("nan"),
            ae_std=float(np.std(aes, ddof=1)) if len(aes) > 1 else 0.0,
            n_cross=len(aes), n_never=n_never,
            final_med=float(np.median(finals)),
        )
        r = out[k]
        print(f"  [{ds} k={k}] MAE={r['mae']:.5f}  "
              f"crossed={r['n_cross']}/{r['n_cross'] + r['n_never']}  "
              f"final_med={r['final_med']:.4f}", flush=True)
    return out


def main():
    global KS, SEEDS, MAX_STEPS
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="calce", choices=list(DS))
    ap.add_argument("--ks", type=int, nargs="+", default=list(KS))
    ap.add_argument("--max-steps", type=int, default=None,
                    help="cap the rollout length (default: to the series end)")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = ap.parse_args()

    KS = tuple(args.ks)
    SEEDS = tuple(args.seeds)
    MAX_STEPS = args.max_steps

    res = run(args.dataset)
    cfg = DS[args.dataset]
    print()
    print("=" * 84)
    print(f"  {args.dataset.upper()}  GROWING window (never drop)   "
          f"EOL={cfg['eol']}  thr={cfg['thr']} Ah  W={cfg['W']}  "
          f"SP{cfg['sp']}")
    print("=" * 84)
    print(f"  {'k':>5} {'MAE (norm.)':>20} {'AE (cycles, crossed)':>24} "
          f"{'final pred (median)':>21}")
    for k in KS:
        r = res.get(k)
        if r is None:
            continue
        ae = ("-- (0/10)" if r["n_cross"] == 0 else
              f"{r['ae']:.0f}+/-{r['ae_std']:.0f} "
              f"({r['n_cross']}/{r['n_cross'] + r['n_never']})")
        print(f"  {k:>5} {r['mae']:.5f}+/-{r['mae_std']:.5f} {ae:>24} "
              f"{r['final_med']:>21.4f}")
    print()
    caps = load_caps(args.dataset)
    tr = np.concatenate([caps[c] for c in cfg["train"]])
    lo, hi = tr.min(), tr.max()
    seq = (caps[cfg["test"]] - lo) / (hi - lo)
    print(f"  reference: true series ends at {seq[-1]:.4f} "
          f"(normalised); EOL threshold {cfg['thr']} Ah")


if __name__ == "__main__":
    main()
