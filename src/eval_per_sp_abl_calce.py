"""Re-evaluate the CALCE per-SP ablation pilot from its checkpoints.

train_per_sp_ablation.py evaluates inline and only writes
results/per_sp_ablation_calce_{config}.json at the very end of a config's
loop, so an interrupted run leaves checkpoints behind with no results file.
That is what happened here: 3 configs x 3 SPs x 3 seeds = 27 checkpoints
under checkpoints/per_sp_abl/calce/, and no JSON.

Each checkpoint stores the metadata the evaluation needs (lo, hi, W, sp,
eol_ah, test_cells), so this script rebuilds the model from the config
flags, loads the state dict, and calls train_per_sp.eval_sp() -- the same
evaluator the training script uses -- so the protocol matches by
construction.  Unit convention therefore follows train_per_sp, which is the
script the 2026-09-17 std sweep unified onto window_std() (unbiased).

Reads only.  Writes nothing; prints a table.

Usage: cd src && python eval_per_sp_abl_calce.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gdn_model import build_gdn_model      # noqa: E402
from make_figures import load_series       # noqa: E402
from train_per_sp import eval_sp, DEV   # noqa: E402

DS = "calce"
SPS = (300, 400, 500)
SEEDS = (1, 2, 3)
CONFIGS = {
    "single": {"multiscale": False, "stage_query": False},
    "multi": {"multiscale": True, "stage_query": False},
    "xchg": {"multiscale": True, "stage_query": True},
    # Capacity-matched control: same single branch widened so its 483,514
    # params sit within 352 of xchg's 483,866 (see train_per_sp_ablation_wide).
    "single_wide": {"multiscale": False, "stage_query": False, "d_model": 304},
}
ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "checkpoints", "per_sp_abl", DS,
)


def main(only: str | None = None) -> None:
    if only:
        print(f"(evaluating {only} only)")
    caps, train_cells, test_cell, W, sps, eol_ah = load_series(DS)
    caps = {c: np.asarray(caps[c], dtype=np.float32) for c in caps}
    test_cells = [test_cell] if isinstance(test_cell, str) else list(test_cell)
    print(f"{DS}: W={W} test_cells={test_cells} eol_ah={eol_ah}")

    res = {}                      # res[cfg][sp] -> list of per-seed row-lists
    for cfg_name, cfg in CONFIGS.items():
        if only and cfg_name != only:
            continue
        for sp in SPS:
            for seed in SEEDS:
                p = os.path.join(ROOT, cfg_name, f"SP{sp}_seed{seed}.pt")
                if not os.path.exists(p):
                    print(f"  missing {p}")
                    continue
                ck = torch.load(p, map_location=DEV, weights_only=False)
                model = build_gdn_model(
                    multiscale=cfg["multiscale"], stage_query=cfg["stage_query"],
                    input_dim=1, window_size=int(ck["W"]), output_len=1,
                    readout="last", d_model=cfg.get("d_model", 64),
                ).to(DEV)
                model.load_state_dict(ck["state_dict"])
                rows = eval_sp(model, caps, test_cells, ck["lo"], ck["hi"],
                               int(ck["W"]), sp, ck["eol_ah"])
                res.setdefault(cfg_name, {}).setdefault(sp, []).append(rows)

    print("\nCALCE per-SP ablation, seeds 1-3 (mean +/- std over seeds, "
          "averaged over test cells)\n")
    hdr = f"{'config':8s} {'SP':>5s} {'MAE':>18s} {'RMSE':>18s} {'R2':>18s} {'AE':>14s}"
    print(hdr)
    print("-" * len(hdr))
    for cfg_name in CONFIGS:
        if only and cfg_name != only:
            continue
        for sp in SPS:
            runs = res.get(cfg_name, {}).get(sp, [])
            if not runs:
                continue
            agg = {}
            for k in ("MAE", "RMSE", "R2", "AE"):
                per_seed = [np.mean([r[k] for r in run]) for run in runs]
                v = np.array(per_seed, dtype=float)
                agg[k] = (v.mean(), v.std(ddof=0))
            row = f"{cfg_name:8s} {sp:5d}"
            for k, w in (("MAE", 18), ("RMSE", 18), ("R2", 18), ("AE", 14)):
                m, s = agg[k]
                row += f" {m:.4f}+/-{s:.4f}".rjust(w) if k != "AE" \
                    else f" {m:.2f}+/-{s:.2f}".rjust(w)
            print(row)
        print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
