"""Recompute ablation MAE/RMSE/R2 (per SP, non-recursive) from abl_*.pt
checkpoints to fill the paper's ablation table. Batched forward."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from gdn_model import build_gdn_model
from load_datasets import (
    load_calce_cells_multivar,
    load_nasa_multivar,
    load_mit_stanford,
    load_panasonic_cells,
    load_tju_cells,
)

CKPT = "D:/research/degradation_prognostics/Transformer_and_Multi_Scale_Models/checkpoints"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIGS = ["single-cap", "multi-cap", "multi-cap-xchg", "multi-cap-physics"]


def load_series(ds):
    if ds == "calce":
        caps_all, _, _ = load_calce_cells_multivar()
        caps = {c: caps_all[c].copy().astype(np.float32) for c in caps_all}
        return caps, ["CS2_36", "CS2_37", "CS2_38"], "CS2_35", 64, [300, 400, 500], 0.77
    if ds == "nasa":
        caps = {}
        for b in ["B0005", "B0006", "B0007", "B0018"]:
            caps[b] = load_nasa_multivar(b)["capacity"].astype(np.float32)
        return caps, ["B0006", "B0007", "B0018"], "B0005", 30, [50, 70, 90], 1.40
    if ds == "mit":
        from load_datasets import MIT_TRAIN_CELLS, MIT_TEST_CELLS
        caps_all = load_mit_stanford()
        caps = {c: caps_all[c].copy().astype(np.float32)
                for c in MIT_TRAIN_CELLS + MIT_TEST_CELLS}
        return caps, MIT_TRAIN_CELLS, MIT_TEST_CELLS[0], 64, [200, 300, 400], 0.86
    if ds == "panasonic":
        caps_all = load_panasonic_cells()
        caps = {c: caps_all[c].copy().astype(np.float32) for c in caps_all}
        cells = sorted(caps.keys())
        return caps, cells[:-1], cells[-1], 30, [300, 500], 2.12
    caps_all = load_tju_cells()
    caps = {c: caps_all[c].copy().astype(np.float32) for c in caps_all}
    return caps, ["CY25_2", "CY25_3"], "CY25_1", 64, [200, 300, 400], 1.75


def evaluate(ds, cfg):
    caps, train_cells, test_cell, W, sps, eol_ah = load_series(ds)
    all_tr = np.concatenate([caps[c] for c in train_cells])
    lo, hi = all_tr.min(), all_tr.max()
    tc = (caps[test_cell] - lo) / (hi - lo + 1e-8)
    ckpt = f"{CKPT}/abl_{ds}_{cfg}.pt"
    if not os.path.exists(ckpt):
        return None
    model = build_gdn_model(
        multiscale=(cfg != "single-cap"),
        cross_exchange=(cfg == "multi-cap-xchg"),
        input_dim=1, window_size=W, output_len=1, readout="last",
        use_physics=False,  # physics is a separate loss-only reg in run_ablation
    ).to(DEV)
    model.load_state_dict(torch.load(ckpt, map_location=DEV, weights_only=True))
    model.eval()

    if ds == "mit":
        from load_datasets import MIT_TEST_CELLS
        test_cells = MIT_TEST_CELLS
    else:
        test_cells = [test_cell]

    per_cell = []
    for tcell in test_cells:
        tc = (caps[tcell] - lo) / (hi - lo + 1e-8)
        windows = np.stack([tc[i - W:i] for i in range(W, len(tc))])
        cin = torch.tensor(windows, dtype=torch.float32).unsqueeze(-1).to(DEV)
        with torch.no_grad():
            pv = model(cin).cpu().numpy()[:, 0]
        pv = pv[: len(tc) - W]
        out = []
        for sp in sps:
            seg_p, seg_t = pv[sp - W:], tc[sp:]
            n = min(len(seg_p), len(seg_t))
            if n == 0:
                out.append((np.nan, np.nan, np.nan))
                continue
            seg_p, seg_t = seg_p[:n], seg_t[:n]
            mae = np.mean(np.abs(seg_p - seg_t))
            rmse = np.sqrt(np.mean((seg_p - seg_t) ** 2))
            r2 = 1 - np.sum((seg_t - seg_p) ** 2) / (np.sum((seg_t - seg_t.mean()) ** 2) + 1e-8)
            out.append((mae, rmse, r2))
        per_cell.append(out)
    return [tuple(float(np.nanmean([pc[i][j] for pc in per_cell])) for j in range(3))
            for i in range(len(sps))]


if __name__ == "__main__":
    for ds in ["calce", "nasa", "panasonic", "mit", "tju"]:
        print(f"\n=== {ds} ===")
        for cfg in CONFIGS:
            res = evaluate(ds, cfg)
            if res is None:
                print(f"  {cfg:24s} no checkpoint")
                continue
            cells = " ".join(f"({m:.4f},{r:.4f},{s:.4f})" for m, r, s in res)
            print(f"  {cfg:24s} {cells}")
