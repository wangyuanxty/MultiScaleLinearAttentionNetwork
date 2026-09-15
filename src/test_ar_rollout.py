"""Recursive (autoregressive) rollout of the K=1 capacity model.

Why this exists. Table A's trajectory is teacher-forced: eval_per_sp_existing.py
stacks every window out of the ground-truth series, so each point is a one-step
prediction anchored to real data and the curve tracks the truth by construction.
That makes the per-step errors a fair like-for-like comparison against the
baselines, which are evaluated the same way, but it means the reported AE is a
local quantity -- where the one-step predictor's output crosses the threshold --
and not a long-horizon RUL forecast.

This script measures the other thing: from the starting point onward the model
sees only its own predictions, so error compounds as it would in deployment.

Two decoding details are copied verbatim from eval_per_sp_existing.py because
getting either wrong silently produces garbage:

  * the model emits a z-score relative to its input window, not a capacity, so
    the output must be denormalised with that window's own mean and std;
  * the checkpoint is one of checkpoints/per_sp/{ds}/SP{sp}_seed{n}.pt, which
    carries its own lo/hi/W/eol_ah. Those are used rather than recomputed, so
    the normalisation cannot drift from what the model was trained under.

Note what the per-window denormalisation does in AR mode. The window is now
made of the model's own outputs; if those are smooth, the window's std shrinks,
and dividing by a shrinking std drags the decoded value toward the window mean.
That is a structural property of the normalisation scheme, not a coding error,
and it is the same mechanism that freezes PatchFormer under rollout.

Run with py312:
    python src/test_ar_rollout.py --dataset calce --seed 1
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gdn_model import build_gdn_model  # noqa: E402
from load_datasets import (  # noqa: E402
    load_calce_cells_multivar,
    load_gotion_cells,
    load_mit_stanford,
    load_nasa_multivar,
    load_panasonic_cells,
)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "checkpoints")
EPS = 1e-6


def load_caps(ds: str):
    """Raw capacity series per cell, matching eval_per_sp_existing.load_series."""
    if ds == 'calce':
        caps_all, _, _ = load_calce_cells_multivar()
        return {c: caps_all[c].copy().astype(np.float32) for c in caps_all}, 'CS2_35', [300, 400, 500]
    if ds == 'nasa':
        caps = {b: load_nasa_multivar(b)['capacity'].astype(np.float32)
                for b in ['B0005', 'B0006', 'B0007', 'B0018']}
        return caps, 'B0005', [50, 70, 90]
    if ds == 'mit':
        caps_all = load_mit_stanford()
        tcells = ['batch2_cell11', 'batch2_cell26', 'batch2_cell32',
                  'batch2_cell36', 'batch2_cell37', 'batch2_cell42',
                  'batch2_cell46']
        return ({c: caps_all[c].copy().astype(np.float32)
                 for c in tcells + ['batch2_cell5']}, 'batch2_cell5',
                [200, 300, 400])
    if ds == 'panasonic':
        caps_all = load_panasonic_cells()
        caps = {c: caps_all[c].copy().astype(np.float32) for c in caps_all}
        cells = sorted(caps.keys())
        return caps, cells[-1], [300, 500, 700]
    if ds == 'gotion':
        caps_all = load_gotion_cells()
        return ({c: caps_all[c].copy().astype(np.float32)
                 for c in ['Cell01', 'Cell02', 'Cell03']}, 'Cell01',
                [500, 800, 1100])
    raise ValueError(ds)


def load_ckpt(ds: str, sp: int, seed: int):
    """Load one per-SP checkpoint and build the model it belongs to.

    stage_query=True matches eval_per_sp_existing.py; the k=1 checkpoints under
    checkpoints/*.pt are an older rope-era lineage and will not load.
    """
    path = os.path.join(CKPT, "per_sp", ds, f"SP{sp}_seed{seed}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ck = torch.load(path, map_location=DEV, weights_only=False)
    model = build_gdn_model(
        multiscale=True, stage_query=True, input_dim=1,
        window_size=ck["W"], output_len=1, readout="last",
    ).to(DEV)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def rollout(model, seq: np.ndarray, sp: int, W: int, max_steps: int) -> np.ndarray:
    """Predict forward from sp using only the model's own outputs.

    Warm-up is the W true cycles before sp -- exactly the input the
    teacher-forced evaluation gets there. After that nothing real is fed.
    Decoding mirrors eval_per_sp_existing.py: the model's output is a z-score
    against the current window, denormalised with that window's mean and std.
    """
    window = list(seq[sp - W:sp].astype(float))
    preds = []
    with torch.no_grad():
        for _ in range(max_steps):
            x = np.asarray(window[-W:], dtype=np.float32)
            wmean = float(x.mean())
            wstd = float(x.std()) + EPS
            t = torch.tensor(x).reshape(1, W, 1).to(DEV)
            p = float(model(t).item()) * wstd + wmean
            preds.append(p)
            window.append(p)
    return np.array(preds)


def first_crossing(series: np.ndarray, thr: float) -> int:
    """First downward crossing index, or -1. Same convention as the
    teacher-forced evaluation so the two remain comparable."""
    for i in range(len(series) - 1):
        if series[i] >= thr > series[i + 1]:
            return i
    return -1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="calce")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=0,
                    help="0 = run to the end of the true series")
    args = ap.parse_args()

    caps, test_cell, sps = load_caps(args.dataset)
    print(f"=== {args.dataset}  test={test_cell}  seed={args.seed} ===")
    print(f"  {'SP':>5} {'TRUL':>6} {'PRUL':>6} {'MAE':>8} {'R2':>9} {'AE':>7}"
          f"  {'wstd(end)':>10}")

    for sp in sps:
        try:
            model, ck = load_ckpt(args.dataset, sp, args.seed)
        except FileNotFoundError as e:
            print(f"  {sp:>5}  no checkpoint: {e}")
            continue
        lo, hi, W, eol_ah = ck["lo"], ck["hi"], ck["W"], ck["eol_ah"]
        seq = (caps[test_cell] - lo) / (hi - lo + EPS)
        thr = (eol_ah - lo) / (hi - lo + EPS)

        max_steps = args.max_steps or (len(seq) - sp)
        pv = rollout(model, seq, sp, W, max_steps)
        tv = seq[sp:sp + len(pv)]
        n = min(len(pv), len(tv))
        pv_c, tv_c = pv[:n], tv[:n]

        mae = float(np.mean(np.abs(pv_c - tv_c)))
        r2 = float(1 - np.sum((tv_c - pv_c) ** 2)
                   / (np.sum((tv_c - tv_c.mean()) ** 2) + EPS))

        true_re = first_crossing(tv, thr)
        pred_re = first_crossing(pv, thr)
        tru = (true_re + 1) if true_re >= 0 else len(tv)
        # How flat the window the model ended up feeding itself: if this
        # collapses, the denormalisation is what is holding the rollout, not
        # the model's forecasting.
        end_wstd = float(np.std(pv[-W:])) if len(pv) >= W else float("nan")

        if pred_re < 0:
            print(f"  {sp:>5} {tru:>6} {'never':>6} {mae:>8.4f} {r2:>9.4f} "
                  f"{'n/a':>7}  {end_wstd:>10.2e}")
        else:
            print(f"  {sp:>5} {tru:>6} {pred_re + 1:>6} {mae:>8.4f} {r2:>9.4f} "
                  f"{abs(tru - (pred_re + 1)):>7}  {end_wstd:>10.2e}")


if __name__ == "__main__":
    main()
