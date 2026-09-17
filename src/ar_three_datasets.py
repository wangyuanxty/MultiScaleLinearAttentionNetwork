"""Autoregressive MSE / AE on the three clean-decline datasets.

MIT, TJU and GOTION have little capacity regeneration and a comparatively clean
downward trend, so they are the fairest test of whether a self-fed rollout
holds up at all -- unlike CALCE, where every configuration tried so far freezes
at window std = 0 (docs/ar_freeze_findings.md).

Protocol, fixed for all three so the numbers are comparable:

    launch  = SP itself.  That is the protocol's own evaluation start (the
              model trained on cycles < SP), so it needs no extra parameter to
              choose or defend.
    steps   = 1000.  The longest gap from SP to EOL across the three is
              GOTION's 922 cycles, so every cell gets the chance to cross.
    seeds   = all 10.  A single seed says nothing: 6/10 against 3/10 read as a
              difference this session but is Fisher p = 0.37.

The smallest SP is used per dataset (MIT/TJU SP200, GOTION SP450) because a
smaller SP means less of the test cell's own history is available, i.e. fewer
independent trajectories to learn the degradation shape from.

AE is always defined.  A rollout that never crosses is reported with a ">="
bound, (launch + steps - 1) - EOL, rather than dropped -- a frozen rollout is a
result, and it is the one CALCE produces.

Read-only w.r.t. checkpoints; writes only results/ar_three_ds.json.

    cd src && D:/anaconda/envs/py312/python.exe ar_three_datasets.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from gdn_model import build_gdn_model                    # noqa: E402
from make_figures import load_series                     # noqa: E402
from train_per_sp import DEV, EPS                        # noqa: E402

JOBS = [("mit", 200, 1000), ("tju", 200, 1000), ("gotion", 450, 1000)]
SEEDS = range(1, 11)

# Where a run's checkpoints land depends on the cwd it was launched from:
# train_per_sp.py saves to "../checkpoints/per_sp/{ds}/...", so a run started
# from src/ writes INSIDE the repo and one started from the repo root writes
# one level ABOVE it.
#
# The two trees are NOT copies of each other -- they hold different generations
# of the same filenames.  GOTION's paper SPs (450/600/750, ten seeds each) live
# only outside the repo; MIT's ten-seed SP200 lives only inside it, while the
# outside tree has a NEWER but 4-seed SP200 from a later run.  Picking a root by
# a fixed order therefore silently swaps in a different model, so the root is
# chosen per (dataset, SP) by which one actually holds the most seeds.
CKPT_ROOTS = ["../../checkpoints/per_sp", "../checkpoints/per_sp"]


def pick_root(ds: str, sp: int, n_seeds: int = 10) -> str | None:
    """The root holding the most seeds for this (dataset, SP)."""
    best, best_n = None, 0
    for root in CKPT_ROOTS:
        n = sum(os.path.exists(os.path.join(root, ds, "SP%d_seed%d.pt" % (sp, s)))
                for s in range(1, n_seeds + 1))
        if n > best_n:
            best, best_n = root, n
    return best


def find_ckpt(ds, sp, seed, root):
    if root is None:
        return None
    p = os.path.join(root, ds, "SP%d_seed%d.pt" % (sp, seed))
    return p if os.path.exists(p) else None


def first_cross(series, thr) -> int:
    for i in range(len(series) - 1):
        if series[i] >= thr > series[i + 1]:
            return i
    return -1


def rollout_batch(model, seqs, t0, W, steps, mode):
    """Self-fed rollout for several series at once.

    All the test cells of a dataset share a launch cycle and a window length, so
    they go through one forward per step instead of one per cell -- the same
    batch-over-samples point that made the scheduled-sampling rollout 8x
    cheaper.  Series shorter than t0+steps stop contributing real truth; the
    rollout keeps running, which is what a deployment would do.
    """
    win = np.stack([np.asarray(s[t0 - W:t0], dtype=np.float64)
                    for s in seqs])                        # (B, W)
    out = np.zeros((len(seqs), steps))
    with torch.no_grad():
        for k in range(steps):
            x = torch.tensor(win, dtype=torch.float32, device=DEV)
            wm = x.mean(dim=1)
            ws = x.std(dim=1) + EPS
            z = model(x[:, :, None]).reshape(-1)
            if mode == "abs":
                y = z
            elif mode == "anchor":
                y = z + wm
            elif mode == "anchor_last":
                y = z + x[:, -1]
            else:
                y = z * ws + wm
            out[:, k] = y.cpu().numpy()
            win = np.concatenate([win[:, 1:], y.cpu().numpy()[:, None]], axis=1)
    return out


def main() -> None:
    all_res = {}
    for ds, sp, steps in JOBS:
        caps, _tr, _te, _W, _sps, eol_ah = load_series(ds)
        root = pick_root(ds, sp)
        n_here = 0 if root is None else sum(
            os.path.exists(os.path.join(root, ds, "SP%d_seed%d.pt" % (sp, s)))
            for s in SEEDS)
        print("=" * 104)
        print("  %s   SP%d   steps=%d   ckpt root=%s (%d seeds)"
              % (ds.upper(), sp, steps, root, n_here))
        print("=" * 104, flush=True)
        print("  %-4s %-14s %8s %8s %11s %11s %8s %6s"
              % ("seed", "cell", "trueEOL", "final", "MSE@50", "MSE", "AE", "cross"))
        rows = []
        for seed in SEEDS:
            path = find_ckpt(ds, sp, seed, root)
            if path is None:
                continue
            ck = torch.load(path, map_location=DEV, weights_only=False)
            W, lo, hi = int(ck["W"]), float(ck["lo"]), float(ck["hi"])
            mode = ck.get("tgt_mode", "zscore")
            tcs = [c for c in ck.get("test_cells", []) if c in caps]
            if not tcs:
                continue
            model = build_gdn_model(
                multiscale=True, stage_query=True, input_dim=1, window_size=W,
                output_len=1, readout="last").to(DEV)
            model.load_state_dict(ck["state_dict"])
            model.eval()
            seqs = [(np.asarray(caps[c], dtype=np.float64) - lo) /
                    (hi - lo + EPS) for c in tcs]
            thr = (eol_ah - lo) / (hi - lo + EPS)
            full = rollout_batch(model, seqs, sp, W, steps, mode)
            for i, c in enumerate(tcs):
                s = seqs[i]
                eol = first_cross(s, thr) + 1
                f = full[i]
                xi = first_cross(np.concatenate([[s[sp - 1]], f]), thr)
                if xi < 0:
                    bound = (sp + steps - 1) - eol
                    ae_val, ae_lbl, crossed = float(bound), ">=%d" % bound, False
                else:
                    ae_val = float(abs((sp + xi) - eol))
                    ae_lbl, crossed = "%d" % ae_val, True
                n_true = max(0, min(steps, len(s) - sp))
                truth = s[sp:sp + n_true]
                n50 = min(50, n_true)
                mse50 = (float(np.mean((f[:n50] - truth[:n50]) ** 2))
                         if n50 else float("nan"))
                mse = (float(np.mean((f[:n_true] - truth) ** 2))
                       if n_true else float("nan"))
                print("  %-4d %-14s %8d %8.4f %11.3e %11.3e %8s %6s"
                      % (seed, c, eol, f[-1], mse50, mse, ae_lbl,
                         "Y" if crossed else "n"), flush=True)
                rows.append(dict(seed=seed, cell=c, true_eol=int(eol),
                                 final=float(f[-1]), mse50=mse50, mse=mse,
                                 ae=ae_val, ae_lbl=ae_lbl, crossed=crossed))
            del model
            torch.cuda.empty_cache()
        if rows:
            print("  %-4s %-14s %8s %8s %11.3e %11.3e %8.1f %4d/%d"
                  % ("MEAN", "", "", "",
                     float(np.nanmean([r["mse50"] for r in rows])),
                     float(np.nanmean([r["mse"] for r in rows])),
                     float(np.mean([r["ae"] for r in rows])),
                     sum(r["crossed"] for r in rows), len(rows)))
        all_res[ds] = dict(sp=sp, steps=steps, rows=rows)
        print(flush=True)
    with open("results/ar_three_ds.json", "w") as fh:
        json.dump(all_res, fh, indent=2)
    print("  -> results/ar_three_ds.json")


if __name__ == "__main__":
    main()
