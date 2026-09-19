"""Evaluate the PANASONIC ablation at seeds 1/2/3 (the 2026-09-17 run).

The checkpoints live in the per-seed layout written by
train_ablation_fullseq.py (checkpoints/abl_seed/{ds}/{config}/seed{n}.pt)
and were trained under that script's protocol: min--max input, per-window
z-score targets, last-token readout, and per-window de-normalization at eval.

recompute_ablation_metrics.evaluate() is the wrong harness for them -- it
performs no per-window de-normalization, so it compares the model's
z-score output against min--max-normalized targets and returns nonsense
(MAE ~1.7, R2 ~-90 on this run).  This script therefore reuses the training
script's own eval_fullseq() and load_series() so the protocol matches by
construction, and evaluates the already-trained checkpoints without
retraining.

Known caveat carried over unchanged: eval_fullseq computes its window sd with
numpy's .std() (biased, ddof=0) while training used torch's .std (unbiased,
ddof=1).  At W=30 that is a 1.68% difference in the de-normalization
denominator.  Left as-is here so these numbers are directly comparable with
the run's own reported metrics; flagged rather than silently corrected.

Reads only: checkpoints/abl_seed/panasonic/{single,multi,xchg}/seed{1,2,3}.pt
Writes: nothing.

Usage: cd src && python eval_abl_panasonic_seeds.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import train_ablation_fullseq as T            # noqa: E402

DS = "panasonic"
SEEDS = (1, 2, 3)
CKROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "checkpoints", "abl_seed", DS,
)


def main() -> None:
    caps, train_cells, test_cell, W, sps, eol_ah = T.load_series(DS)
    caps = {c: np.asarray(caps[c], dtype=np.float32) for c in caps}
    all_tr = np.concatenate([caps[c] for c in train_cells])
    lo, hi = float(all_tr.min()), float(all_tr.max())
    print(f"{DS}: W={W} lo={lo:.4f} hi={hi:.4f} test_cell={test_cell} "
          f"eol_ah={eol_ah}")

    res = {}
    for cfg_name, cfg in T.CONFIGS.items():
        for seed in SEEDS:
            p = os.path.join(CKROOT, cfg_name, f"seed{seed}.pt")
            if not os.path.exists(p):
                print(f"  missing {p}")
                continue
            ck = torch.load(p, map_location=T.DEV, weights_only=False)
            model = T.build_gdn_model(
                multiscale=cfg["multiscale"], stage_query=cfg["stage_query"],
                input_dim=1, window_size=W, output_len=1, readout="last",
            ).to(T.DEV)
            model.load_state_dict(ck["state_dict"])
            res.setdefault(cfg_name, []).append(
                T.eval_fullseq(model, caps, test_cell, W, lo, hi, eol_ah))

    print(f"\nPANASONIC ablation, seeds {list(SEEDS)}, "
          f"full-sequence single-value metrics (mean +/- std over seeds)\n")
    hdr = f"{'config':10s} {'MAE':>18s} {'R2':>18s} {'regen':>18s} {'AE':>16s}"
    print(hdr)
    print("-" * len(hdr))
    for cfg_name, ms in res.items():
        row = f"{cfg_name:10s}"
        for key, fmt in (("mae", "%.4f"), ("r2", "%.4f"),
                         ("regen", "%.4f"), ("ae", "%.1f")):
            v = np.array([m[key] for m in ms], dtype=float)
            val = (fmt % v.mean()) + "+/-" + (fmt % v.std(ddof=0))
            row += f" {val:>18s}" if key != "ae" else f" {val:>16s}"
        print(row)


if __name__ == "__main__":
    main()
