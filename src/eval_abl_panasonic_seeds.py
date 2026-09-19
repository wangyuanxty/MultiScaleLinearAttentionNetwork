"""Evaluate the PANASONIC ablation at seeds 1/2/3 (the 2026-09-17 run).

recompute_ablation_metrics.evaluate() hard-codes the flat checkpoint name
f"{CKPT}/abl_{ds}_{cfg}.pt" and evaluates a single checkpoint per config,
so it cannot see the newer per-seed layout written by
train_ablation_fullseq.py: checkpoints/abl_seed/{ds}/{config}/seed{n}.pt.

Rather than reimplement the metric code, this wrapper copies one seed's
checkpoint into a shim directory under the flat name evaluate() expects,
points the module's CKPT at that shim, and calls evaluate() unchanged.
That keeps a single implementation of the MAE/RMSE/R2 computation.

Reads only: checkpoints/abl_seed/panasonic/{single,multi,xchg}/seed{1,2,3}.pt
Writes only: a temp shim directory (removed on exit). No results file.

Usage: cd src && python eval_abl_panasonic_seeds.py
"""
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import recompute_ablation_metrics as R          # noqa: E402

CFG_DIR = {"single-cap": "single", "multi-cap": "multi", "multi-cap-xchg": "xchg"}
SEEDS = (1, 2, 3)
DS = "panasonic"


def main() -> None:
    src_root = os.path.join(R.CKPT, "abl_seed", DS)
    missing = [
        os.path.join(src_root, d, f"seed{s}.pt")
        for d in CFG_DIR.values() for s in SEEDS
        if not os.path.exists(os.path.join(src_root, d, f"seed{s}.pt"))
    ]
    if missing:
        print("missing checkpoints:")
        for m in missing:
            print("   ", m)
        return

    shim = tempfile.mkdtemp(prefix="ablshim_")
    saved = R.CKPT
    R.CKPT = shim
    try:
        # results[cfg][seed] = [(mae, rmse, r2) per SP]
        results = {cfg: {} for cfg in CFG_DIR}
        for seed in SEEDS:
            for cfg, d in CFG_DIR.items():
                src = os.path.join(src_root, d, f"seed{seed}.pt")
                # abl_seed/ stores {"state_dict", "seed", "lo", "hi", "W",
                # "eol_ah", "config"}; evaluate() loads a bare state_dict.
                obj = torch.load(src, map_location="cpu", weights_only=False)
                sd = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
                torch.save(sd, os.path.join(shim, f"abl_{DS}_{cfg}.pt"))
                results[cfg][seed] = R.evaluate(DS, cfg)
    finally:
        R.CKPT = saved
        shutil.rmtree(shim, ignore_errors=True)

    n_sp = len(results["single-cap"][SEEDS[0]])
    print(f"\nPANASONIC ablation, seeds {list(SEEDS)} "
          f"(mean +/- std over seeds)\n")
    hdr = f"{'config':18s}"
    for i in range(n_sp):
        hdr += f" | {'SP'+str(i+1)+' MAE':>16s} {'R2':>16s}"
    print(hdr)
    print("-" * len(hdr))
    for cfg in CFG_DIR:
        row = f"{cfg:18s}"
        for i in range(n_sp):
            m = np.array([results[cfg][s][i][0] for s in SEEDS], dtype=float)
            r = np.array([results[cfg][s][i][2] for s in SEEDS], dtype=float)
            row += f" | {m.mean():.4f}+/-{m.std(ddof=0):.4f} {r.mean():.4f}+/-{r.std(ddof=0):.4f}"
        print(row)


if __name__ == "__main__":
    main()
