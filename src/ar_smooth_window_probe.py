"""Is the z-score model able to use a low-amplitude window at all?

Diagnostic for choosing between the two retraining routes for the AR freeze
(see docs/ar_freeze_findings.md):

  A  augment the training windows so they LOOK like what autoregression
     feeds back -- smoothed / amplitude-shrunk, same true target
  B  scheduled sampling -- train on the model's own rollout windows

A is only worth trying if the model can, in principle, read a trend out of a
low-amplitude window.  If its output already tracks the ideal z on a smoothed
window, the information is there and the problem is purely the AR path (-> B).
If the output collapses to ~0 on every smoothed window, the model never saw
that input and has no way to use it (-> A is the direct fix).

Method: take the TRUE windows in the evaluation region (no rollout, no
feedback), transform the input, and ask two questions per transform:

  * one-step MAE after the model's own per-window decode -- does accuracy
    survive the transform?
  * mean|z_out| / mean|z_ideal| -- z_ideal = (y - mean(x')) / std(x') is what a
    perfect one-step predictor would emit for that transformed window.  A
    ratio near 1 means the model tracks it; near 0 means it collapsed onto the
    window mean regardless of the input.

Nothing is written; no checkpoint is modified.

    D:/anaconda/envs/py312/python.exe src/ar_smooth_window_probe.py
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import test_ar_rollout as tar          # noqa: E402
from gdn_model import build_gdn_model  # noqa: E402
from make_figures import load_series   # noqa: E402

EPS = 1e-6
# the AR collapse measured in docs/ar_freeze_findings.md section 2.2:
# sigma goes 0.00828 -> 0.00024, i.e. a 34x amplitude shrink.
AR_SHRINK = 34.0
SCALE_LAMBDAS = (0.5, 0.2, 0.1, 1.0 / AR_SHRINK)
MOVE_AVGS = (5, 11, 21)


def transform_names() -> list[str]:
    return (["identity (real window)"]
            + [f"scale lambda={lam:.3f}" for lam in SCALE_LAMBDAS]
            + [f"moving average k={k}" for k in MOVE_AVGS])


def transforms(x: np.ndarray) -> dict:
    """Input-side transforms, all amplitude-reducing like autoregression is."""
    out = {"identity (real window)": x}
    mu = x.mean(axis=1, keepdims=True)
    for lam in SCALE_LAMBDAS:
        out[f"scale lambda={lam:.3f}"] = mu + lam * (x - mu)
    for k in MOVE_AVGS:
        pad = k // 2
        xp = np.pad(x, ((0, 0), (pad, pad)), mode="edge")
        ker = np.ones(k) / k
        out[f"moving average k={k}"] = np.apply_along_axis(
            lambda r: np.convolve(r, ker, mode="valid"), 1, xp)
    return out


def load_ckpt_suffixed(ds: str, sp: int, seed: int, suffix: str):
    """tar.load_ckpt with a checkpoint-name suffix (e.g. "_ampreg" / "_abs").

    Kept here rather than in test_ar_rollout so nothing shared changes; the
    model construction is copied verbatim from there.
    """
    if not suffix:
        return tar.load_ckpt(ds, sp, seed)
    path = os.path.join(tar.CKPT, "per_sp", ds, f"SP{sp}_seed{seed}{suffix}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ck = torch.load(path, map_location=tar.DEV, weights_only=False)
    model = build_gdn_model(
        multiscale=True, stage_query=True, input_dim=1,
        window_size=ck["W"], output_len=1, readout="last").to(tar.DEV)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="calce")
    ap.add_argument("--sp", type=int, default=500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--ckpt-suffix", default="",
                    help='checkpoint name suffix, e.g. "_ampreg" or "_abs"')
    args = ap.parse_args()

    caps, _train, test_cell, W, _sps, _eol = load_series(args.dataset)
    caps32 = {c: caps[c].astype(np.float32) for c in caps}
    names = transform_names()
    mae = {k: [] for k in names}
    absz = {k: [] for k in names}
    absz_ideal = {k: [] for k in names}
    n_win = 0

    for seed in args.seeds:
        model, ck = load_ckpt_suffixed(args.dataset, args.sp, seed,
                                       args.ckpt_suffix)
        lo, hi, w = ck["lo"], ck["hi"], ck["W"]
        assert w == W, f"ckpt W={w} != {W}"
        seq = (caps32[test_cell] - lo) / (hi - lo + EPS)
        idx = np.arange(args.sp, len(seq))
        X = np.stack([seq[i - W:i] for i in idx]).astype(np.float64)
        Y = seq[idx].astype(np.float64)
        assert len(X) > 100, f"only {len(X)} eval windows"
        n_win = len(X)

        for name, xt in transforms(X).items():
            mu = xt.mean(axis=1)
            sd = xt.std(axis=1) + EPS
            z_ideal = (Y - mu) / sd
            chunks = []
            with torch.no_grad():
                for j in range(0, len(xt), 512):
                    b = torch.tensor(
                        xt[j:j + 512, :, None].astype(np.float32),
                        device=tar.DEV)
                    chunks.append(model(b).cpu().numpy().reshape(-1))
            z_out = np.concatenate(chunks).astype(np.float64)
            pred = z_out * sd + mu
            mae[name].append(float(np.mean(np.abs(pred - Y))))
            absz[name].append(float(np.mean(np.abs(z_out))))
            absz_ideal[name].append(float(np.mean(np.abs(z_ideal))))
        del model
        torch.cuda.empty_cache()

    print()
    print("=" * 96)
    print(f"  {args.dataset.upper()}  z-score checkpoints SP{args.sp}  "
          f"seeds={args.seeds}  ckpt-suffix={args.ckpt_suffix or '(none)'}")
    print(f"  {n_win} true windows per seed, cycles {args.sp}..{len(seq) - 1}"
          f"  (teacher-forced: no feedback, only the input is transformed)")
    print("=" * 96)
    base = None
    hdr = (f"  {'input transform':<26}{'1-step MAE':>12}{'x base':>9}"
           f"{'mean|z_out|':>13}{'mean|z_ideal|':>15}{'ratio':>8}")
    print(hdr)
    print("-" * len(hdr))
    for name in names:
        m = float(np.mean(mae[name]))
        a = float(np.mean(absz[name]))
        ai = float(np.mean(absz_ideal[name]))
        if base is None:
            base = m
        print(f"  {name:<26}{m:>12.5f}{m / base:>9.2f}"
              f"{a:>13.4f}{ai:>15.4f}{a / (ai + EPS):>8.3f}")
    print()
    print("  ratio = mean|z_out| / mean|z_ideal|:  1.0 = the model reads the")
    print("  trend out of this window;  ~0 = it collapsed onto the window mean")
    print("  regardless of the input, i.e. it has never learned this input.")


if __name__ == "__main__":
    main()
