"""Recompute Table A under both window-std conventions, without overwriting it.

Why this exists
---------------
Training builds its target scale from a torch tensor (`x[:, :, 0].std(dim=1)`),
and torch.std is UNBIASED by default while numpy's .std() is biased.  The
evaluation paths used numpy, so every Table A number was decoded 0.78% away
from the convention the model was trained in.  That is now fixed in the four
active paths (train_per_sp.eval_sp, ar_probe, eval_per_sp_existing,
eval_multiseed) -- this script quantifies what the fix is worth, by producing
BOTH decodes from ONE forward pass and diffing them.

One forward pass, two decodes
-----------------------------
The model output is identical either way; only the scalar it is multiplied by
changes.  So the expensive part is shared and the comparison is exact rather
than two runs that might differ for other reasons.

Scope: 6 datasets x 3 SPs x 10 seeds, the same checkpoints Table A was built
from, using each checkpoint's OWN lo/hi (the authoritative pair -- it is what
the model was trained and written with).

Writes only results/tableA_two_conventions.json; nothing existing is touched.

    cd src && D:/anaconda/envs/py312/python.exe recompute_tableA.py
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
from eval_multiseed import true_rul                      # noqa: E402
from gdn_model import build_gdn_model                    # noqa: E402
from make_figures import load_series                     # noqa: E402
from train_per_sp import DEV, EPS                        # noqa: E402

DATASETS = ["calce", "nasa", "mit", "panasonic", "tju", "gotion"]
SEEDS = range(1, 11)
OUT = "results/tableA_two_conventions.json"


def eval_both(model, caps, test_cells, lo, hi, W, sp, eol_ah, mode):
    """One forward pass per window set; MAE/RMSE/R2/AE under both conventions."""
    model.eval()
    th = (eol_ah - lo) / (hi - lo + EPS)
    out = []
    with torch.no_grad():
        for tc in test_cells:
            seq = (np.asarray(caps[tc], dtype=np.float64) - lo) / (hi - lo + EPS)
            idx = np.arange(sp, len(seq))
            if len(idx) == 0:
                continue
            X = np.stack([seq[i - W:i, None] for i in idx]).astype(np.float32)
            wmean = X[:, :, 0].mean(axis=1)
            xt = torch.tensor(X, device=DEV)
            # torch.std on the fed tensor == training's convention
            wstd_t = xt[:, :, 0].std(dim=1).cpu().numpy()
            # the old, biased one, for the diff
            wstd_n = X[:, :, 0].std(axis=1)
            raw = model(xt).squeeze(-1).cpu().numpy()
            rec = {"cell": tc}
            for name, ws in (("torch", wstd_t), ("numpy_biased", wstd_n)):
                seg = raw if mode == "abs" else raw * ws + wmean
                tv = seq[sp:]
                n = min(len(tv), len(seg))
                tv, sg = tv[:n], seg[:n]
                rec[name] = {
                    "MAE": float(np.mean(np.abs(tv - sg))),
                    "RMSE": float(np.sqrt(np.mean((tv - sg) ** 2))),
                    "R2": float(1 - np.sum((tv - sg) ** 2) /
                                (np.sum((tv - tv.mean()) ** 2) + EPS)),
                    "AE": int(abs(true_rul(tv, th) - true_rul(sg, th))),
                }
            out.append(rec)
    return out


def main() -> None:
    res, flips = {}, []
    for ds in DATASETS:
        caps, _tr, _te, _W, sps, eol_ah = load_series(ds)
        res[ds] = {}
        for sp in sps:
            root = pick_root(ds, sp)
            if root is None:
                print("  %s SP%d: no checkpoints" % (ds, sp), flush=True)
                continue
            rows = []
            for seed in SEEDS:
                p = os.path.join(root, ds, "SP%d_seed%d.pt" % (sp, seed))
                if not os.path.exists(p):
                    continue
                ck = torch.load(p, map_location=DEV, weights_only=False)
                W, lo, hi = int(ck["W"]), float(ck["lo"]), float(ck["hi"])
                mode = ck.get("tgt_mode", "zscore")
                tcs = [c for c in ck.get("test_cells", []) if c in caps]
                if not tcs:
                    continue
                model = build_gdn_model(
                    multiscale=True, stage_query=True, input_dim=1,
                    window_size=W, output_len=1, readout="last").to(DEV)
                model.load_state_dict(ck["state_dict"])
                for r in eval_both(model, caps, tcs, lo, hi, W, sp, eol_ah, mode):
                    r["seed"] = seed
                    rows.append(r)
                    if r["torch"]["AE"] != r["numpy_biased"]["AE"]:
                        flips.append((ds, sp, seed, r["cell"],
                                      r["numpy_biased"]["AE"], r["torch"]["AE"]))
                del model
                torch.cuda.empty_cache()
            res[ds][str(sp)] = rows
            if rows:
                def mean(k, f):
                    return float(np.mean([r[k][f] for r in rows]))
                print("  %-10s SP%-4d n=%2d  MAE %.6f -> %.6f   R2 %.4f -> %.4f   "
                      "AE %.2f -> %.2f"
                      % (ds, sp, len(rows), mean("numpy_biased", "MAE"),
                         mean("torch", "MAE"), mean("numpy_biased", "R2"),
                         mean("torch", "R2"), mean("numpy_biased", "AE"),
                         mean("torch", "AE")), flush=True)
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=2)
    print()
    print("  AE changed on %d of the scored cells:" % len(flips))
    for f in flips:
        print("     %-10s SP%-4d seed%-3d %-14s  %d -> %d" % f)
    print("  -> %s" % OUT)


if __name__ == "__main__":
    main()
