"""The three clean-decline datasets at near-EOL launch points, with long rollouts.

The SP-launch measurement (ar_three_datasets.py) puts the launch 289-922 cycles
before EOL, which is where every model freezes -- and a freeze at that distance
is a floor effect: it cannot separate a model that would work at a shorter
horizon from one that would not.

This re-measures the same checkpoints at EOL-50 and EOL-100, the launch points
the MIT dose experiment used, so the two are directly comparable.

Steps are 1000, not the dose experiment's 150.  That mattered on CALCE: at 150
steps the main model's finals looked like plausible declining values
(0.51-0.53) and MSE sat at ~1e-03, but the same rollout at 1000 steps runs away
to -0.14 and MSE 2.4e-02.  The 150-step cap was hiding the runaway, not
measuring a healthy trajectory.  AE is unaffected -- the crossing happens in the
first ~50 steps -- but `final` and `MSE` are only meaningful at a horizon long
enough for the failure to actually appear.

Both launch points of every test cell go through one forward per step, since
they share the model, the window length and the step count.

Read-only w.r.t. checkpoints; writes only results/ar_three_ds_neareol.json.

    cd src && D:/anaconda/envs/py312/python.exe ar_three_ds_neareol.py
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

from ar_three_datasets import pick_root                  # noqa: E402
from gdn_model import build_gdn_model                    # noqa: E402
from make_figures import load_series                     # noqa: E402
from train_per_sp import DEV, EPS                        # noqa: E402

JOBS = [("mit", 200), ("tju", 200), ("gotion", 450)]
LAUNCH_K = (50, 100)
STEPS = 1000
SEEDS = range(1, 11)


def first_cross(series, thr) -> int:
    for i in range(len(series) - 1):
        if series[i] >= thr > series[i + 1]:
            return i
    return -1


def rollout_multi(model, wins, steps, mode):
    """Self-fed rollout for several windows at once (same model, same length)."""
    win = np.stack(wins)
    out = np.zeros((len(wins), steps))
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
    for ds, sp in JOBS:
        caps, _tr, _te, _W, _sps, eol_ah = load_series(ds)
        root = pick_root(ds, sp)
        print("=" * 100)
        print("  %s   SP%d   launch=EOL-50 / EOL-100   steps=%d   root=%s"
              % (ds.upper(), sp, STEPS, root))
        print("=" * 100, flush=True)
        print("  %-4s %-14s %6s %9s %11s %11s %8s %6s"
              % ("seed", "cell", "dEOL", "final", "MSE@50", "MSE", "AE", "cross"))
        rows = []
        for seed in SEEDS:
            path = (os.path.join(root, ds, "SP%d_seed%d.pt" % (sp, seed))
                    if root else None)
            if not path or not os.path.exists(path):
                continue
            ck = torch.load(path, map_location=DEV, weights_only=False)
            W, lo, hi = int(ck["W"]), float(ck["lo"]), float(ck["hi"])
            mode = ck.get("tgt_mode", "zscore")
            tcs = [c for c in ck.get("test_cells", []) if c in caps]
            if not tcs:
                continue
            seqs = [(np.asarray(caps[c], dtype=np.float64) - lo) / (hi - lo + EPS)
                    for c in tcs]
            thr = (eol_ah - lo) / (hi - lo + EPS)
            eols = [first_cross(s, thr) + 1 for s in seqs]
            wins, meta = [], []
            for i, c in enumerate(tcs):
                for k in LAUNCH_K:
                    t0 = eols[i] - k
                    if t0 < W:
                        continue
                    wins.append(seqs[i][t0 - W:t0])
                    meta.append((c, k, t0, i))
            if not wins:
                continue
            model = build_gdn_model(
                multiscale=True, stage_query=True, input_dim=1, window_size=W,
                output_len=1, readout="last").to(DEV)
            model.load_state_dict(ck["state_dict"])
            model.eval()
            full = rollout_multi(model, wins, STEPS, mode)
            for j, (c, k, t0, i) in enumerate(meta):
                s, f, eol = seqs[i], full[j], eols[i]
                xi = first_cross(np.concatenate([[s[t0 - 1]], f]), thr)
                if xi < 0:
                    ae_val, ae_lbl, crossed = None, "n", False
                else:
                    ae_val = float(abs((t0 + xi) - eol))
                    ae_lbl, crossed = "%d" % ae_val, True
                n_true = max(0, min(STEPS, len(s) - t0))
                truth = s[t0:t0 + n_true]
                n50 = min(50, n_true)
                mse50 = (float(np.mean((f[:n50] - truth[:n50]) ** 2))
                         if n50 else float("nan"))
                mse = (float(np.mean((f[:n_true] - truth) ** 2))
                       if n_true else float("nan"))
                print("  %-4d %-14s %6d %9.4f %11.3e %11.3e %8s %6s"
                      % (seed, c, k, f[-1], mse50, mse, ae_lbl,
                         "Y" if crossed else "n"), flush=True)
                rows.append(dict(seed=seed, cell=c, launch_k=k, true_eol=int(eol),
                                 final=float(f[-1]), mse50=mse50, mse=mse,
                                 ae=ae_val, ae_lbl=ae_lbl, crossed=crossed))
            del model
            torch.cuda.empty_cache()
        if rows:
            for k in LAUNCH_K:
                sub = [r for r in rows if r["launch_k"] == k]
                if not sub:
                    continue
                cr = [r["ae"] for r in sub if r["ae"] is not None]
                print("  EOL-%-4d MEAN  final=%7.4f  MSE@50=%.3e  MSE=%.3e  "
                      "cross=%2d/%2d  AE=%s"
                      % (k, float(np.mean([r["final"] for r in sub])),
                         float(np.nanmean([r["mse50"] for r in sub])),
                         float(np.nanmean([r["mse"] for r in sub])),
                         len(cr), len(sub),
                         "%.1f" % np.mean(cr) if cr else "n/a"))
        all_res[ds] = dict(sp=sp, steps=STEPS, launch_k=list(LAUNCH_K), rows=rows)
        print(flush=True)
    with open("results/ar_three_ds_neareol.json", "w") as fh:
        json.dump(all_res, fh, indent=2)
    print("  -> results/ar_three_ds_neareol.json")


if __name__ == "__main__":
    main()
