"""Put the three target modes back into ONE space so they can be compared.

The training loss is `masked_mae(pred, target_of(y, ...))` -- an MAE in TARGET
space.  That space is not the same for the three modes:

    zscore   MAE in z units         (y - mu) / sigma
    abs      MAE in capacity units  y
    anchor   MAE in capacity units  y - mu

A run's printed epoch loss therefore carries its own units.  z-score's ~0.008
and anchor's ~0.005 look like the same kind of number and are not: one is a
fraction of a window sigma, the other a fraction of a capacity.  Putting them
in one table is the same class of mistake as decoding an `_abs` checkpoint with
the z-score rule (docs/ar_freeze_findings.md section 7) -- silent, and it makes
the wrong option look good.

This evaluates every available checkpoint for one (dataset, SP, seed) on the
SAME windows -- build_windows from the trainer, so they are literally the
training objective's own windows -- decodes each with its own rule, and reports
everything in capacity units.

Two baselines come along because an MAE alone cannot say whether the model is
doing anything: `window-last` (predict that nothing changes) and a `ridge` fit
on the same windows (least squares, one weight per window position).  A mode
that loses to window-last has learned nothing the input did not already say.

Also prints the fitted `pred = a*Y + b`.  a << 1 is the shrinkage that made the
absolute-target rollout drift: the model compresses its output toward the mean,
and under autoregression that bias is fed back in every step.

Read-only: loads checkpoints and series, prints.  Writes nothing.

    cd src && D:/anaconda/envs/py312/python.exe diag_mode_mae.py \
        --dataset calce --sp 300 --seed 1
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from gdn_model import build_gdn_model                    # noqa: E402
from make_figures import load_series                     # noqa: E402
from train_per_sp import (DEV, EPS, build_windows,       # noqa: E402
                          decode_pred)


def mode_of(path: str, ck: dict) -> str:
    """The checkpoint is the authority; the filename is only a fallback."""
    m = ck.get("tgt_mode")
    if m in ("zscore", "abs", "anchor"):
        return m
    base = os.path.basename(path)
    return "abs" if "_abs" in base else ("anchor" if "_anchor" in base
                                         else "zscore")


def predict(model, X, mode, batch=256) -> np.ndarray:
    """Capacity-space predictions, decoded according to `mode`.

    The window statistics must be recomputed from the SAME input the model
    sees -- that is what the decode is defined against -- so this mirrors
    train_one's own wmean/wstd rather than caching one window's stats.
    """
    out = []
    with torch.no_grad():
        for s in range(0, len(X), batch):
            x = torch.tensor(X[s:s + batch]).to(DEV)
            pred = model(x).squeeze(-1)
            wmean = x[:, :, 0].mean(dim=1)
            wstd = x[:, :, 0].std(dim=1) + EPS
            out.append(decode_pred(pred, wmean, wstd, mode).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def report(tag, pred, Y) -> dict:
    err = pred - Y
    a, b = np.polyfit(Y, pred, 1)
    return dict(tag=tag, mae=float(np.mean(np.abs(err))),
                rmse=float(np.sqrt(np.mean(err ** 2))),
                bias=float(np.mean(err)), frac_neg=float(np.mean(err < 0)),
                slope=float(a), intercept=float(b))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="calce")
    p.add_argument("--sp", type=int, default=300)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--cells", choices=["all", "train", "test"], default="all",
                   help="which window set to score on; the trainer fits on "
                        "'all' (train cells full + test cells < SP)")
    args = p.parse_args()

    caps, train_cells, test_cells, W_ds, _sps, eol_ah = load_series(args.dataset)
    # load_series returns test_cell as a STRING (CLAUDE.md: the canonical tuple
    # is `(caps, train_cells, test_cell, W, sps, eol_ah)`), so `for c in
    # test_cells` walks its characters -- CS2_35 becomes C, S, 2, ...  and the
    # first one dies with KeyError: 'C'.  Normalise both to lists.
    if isinstance(test_cells, str):
        test_cells = [test_cells]
    if isinstance(train_cells, str):
        train_cells = [train_cells]
    ckpt_dir = os.path.join("..", "checkpoints", "per_sp", args.dataset)
    pat = os.path.join(ckpt_dir, "SP%d_seed%d*.pt" % (args.sp, args.seed))
    files = sorted(f for f in glob.glob(pat)
                   if "_ep" not in os.path.basename(f))
    if not files:
        raise SystemExit("no checkpoints match %s" % pat)

    print("=" * 96)
    print("  %s  SP%d  seed%d   --  every number below is CAPACITY-space "
          "(normalised)" % (args.dataset, args.sp, args.seed))
    print("=" * 96)

    rows, W = [], None
    for path in files:
        ck = torch.load(path, map_location=DEV, weights_only=False)
        mode = mode_of(path, ck)
        W, lo, hi = int(ck["W"]), float(ck["lo"]), float(ck["hi"])
        if args.cells == "train":
            X, Y = build_windows(caps, train_cells, lo, hi, W)
        elif args.cells == "test":
            X, Y = build_windows(caps, test_cells, lo, hi, W,
                                 max_cycle=args.sp)
        else:
            # exactly what train_one's X_all is: train cells full + test cell
            # cycles < SP, which is the set the objective was minimised on
            Xtr, Ytr = build_windows(caps, train_cells, lo, hi, W)
            Xts, Yts = build_windows(caps, test_cells, lo, hi, W,
                                     max_cycle=args.sp)
            X, Y = np.vstack([Xtr, Xts]), np.concatenate([Ytr, Yts])
        Y = np.asarray(Y, dtype=np.float64)

        model = build_gdn_model(
            multiscale=True, stage_query=True, input_dim=1, window_size=W,
            output_len=1, readout="last").to(DEV)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        pred = predict(model, X, mode)
        r = report("%s  [%s]" % (os.path.basename(path), mode), pred, Y)
        rows.append(r)
        del model
        torch.cuda.empty_cache()

    # Baselines on the same windows.  window-last is the model-free statement
    # "capacity does not change"; ridge is the best linear map from the whole
    # window, so it is the bar a sequence model has to clear.
    rows.append(report("window-last  [baseline]",
                       X[:, -1, 0].astype(np.float64), Y))
    Xf = X.reshape(len(X), -1).astype(np.float64)
    A = np.hstack([Xf, np.ones((len(Xf), 1))])
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    rows.append(report("ridge        [baseline]", A @ coef, Y))

    print("  windows scored: %d   W=%d   cells=%s" % (len(Y), W, args.cells))
    print()
    print("  %-40s %10s %10s %10s %8s %8s"
          % ("", "MAE", "RMSE", "bias", "frac<0", "slope a"))
    print("  " + "-" * 92)
    for r in rows:
        print("  %-40s %10.5f %10.5f %+10.5f %8.2f %8.4f"
              % (r["tag"], r["mae"], r["rmse"], r["bias"], r["frac_neg"],
                 r["slope"]))
    print()
    print("  slope a from pred = a*Y + b  (a ~ 1 means unbiased; a < 1 means the")
    print("  output is compressed toward the mean, which is what feeds the")
    print("  autoregressive drift).  frac<0 = share of windows under-predicted.")


if __name__ == "__main__":
    main()
