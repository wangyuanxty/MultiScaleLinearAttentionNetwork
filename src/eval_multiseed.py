"""Multi-seed evaluation: recompute Table A (K=1) and Table B (K=32)
metrics for each seed checkpoint, then report mean±std across seeds.

Checkpoints: checkpoints/unified_{ds}_K{K}_seed{seed}.pt
(seed 42 falls back to the legacy name without _seed42).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from gdn_model import build_gdn_model
from make_figures import load_series

CKPT = "D:/research/degradation_prognostics/Transformer_and_Multi_Scale_Models/checkpoints"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [42, 43, 44]

SPS = {
    "calce": [300, 400, 500], "nasa": [50, 70, 90],
    "mit": [200, 300, 400], "panasonic": [300, 500],
    "tju": [200, 300, 400],
}


def load_model(ds, K, seed):
    W = {"calce": 64, "nasa": 30, "mit": 64,
         "panasonic": 30, "tju": 64}[ds]
    model = build_gdn_model(multiscale=True, cross_exchange=False,
                            stage_query=True, input_dim=1, window_size=W,
                            output_len=K, readout="last").to(DEV)
    path = f"{CKPT}/unified_{ds}_K{K}_seed{seed}.pt"
    if not os.path.exists(path) and seed == 42:
        path = f"{CKPT}/unified_{ds}_K{K}.pt"
    if not os.path.exists(path):
        return None, W
    model.load_state_dict(torch.load(path, map_location=DEV, weights_only=True))
    model.eval()
    return model, W


def true_rul(seg, th):
    """First index where the sequence crosses th from above
    (seg[i] >= th > seg[i+1]); matches test_recompute_ae_k32."""
    for i in range(len(seg) - 1):
        if seg[i] >= th > seg[i + 1]:
            return i
    return len(seg)


def eval_table_a(ds, seed):
    """Per-SP (MAE, RMSE, R2, AE) for K=1, non-recursive.

    MIT uses 8 hold-out test cells (80/20); results are averaged
    across the test cells per SP.
    """
    caps, train_cells, test_cell, W, sps, eol_ah = load_series(ds)
    all_tr = np.concatenate([caps[c] for c in train_cells])
    lo, hi = all_tr.min(), all_tr.max()
    threshold_n = (eol_ah - lo) / (hi - lo + 1e-8)

    if ds == "mit":
        from load_datasets import MIT_TEST_CELLS
        test_cells = MIT_TEST_CELLS
    else:
        test_cells = [test_cell]

    model, W = load_model(ds, 1, seed)
    if model is None:
        return None

    per_cell = []
    for tcell in test_cells:
        tc = (caps[tcell] - lo) / (hi - lo + 1e-8)
        out = []
        for sp in sps:
            # PatchFormer-style: test sequence truncated to start at SP-W,
            # so the first window's prediction target is exactly SP.
            n = len(tc) - sp
            if n <= 0:  # short cell: SP beyond cell length
                out.append((np.nan, np.nan, np.nan, np.nan))
                continue
            windows = np.stack([tc[sp - W + i:sp + i] for i in range(n)])
            wmean = windows.mean(axis=1)
            wstd = windows.std(axis=1) + 1e-6
            cin = torch.tensor(windows, dtype=torch.float32).unsqueeze(-1).to(DEV)
            with torch.no_grad():
                seg_p = model(cin).cpu().numpy()[:, 0]
            seg_p = seg_p * wstd + wmean  # per-window de-normalize
            seg_t = tc[sp:]
            mae = np.mean(np.abs(seg_p - seg_t))
            rmse = np.sqrt(np.mean((seg_p - seg_t) ** 2))
            r2 = 1 - np.sum((seg_t - seg_p) ** 2) / (np.sum((seg_t - seg_t.mean()) ** 2) + 1e-8)
            ae = abs(true_rul(seg_t, threshold_n) - true_rul(seg_p, threshold_n))
            out.append((mae, rmse, r2, ae))
        per_cell.append(out)
    # average across test cells, per SP (skip nan cells/sp)
    return [tuple(float(np.nanmean([pc[i][j] for pc in per_cell])) for j in range(4))
            for i in range(len(sps))]


def eval_table_b(ds, seed):
    """K=32 early-sensing AE per SP (Table B protocol).

    MIT uses 8 hold-out test cells; AE averaged across test cells.
    """
    caps, train_cells, test_cell, W, sps, eol_ah = load_series(ds)
    all_tr = np.concatenate([caps[c] for c in train_cells])
    lo, hi = all_tr.min(), all_tr.max()
    threshold_n = (eol_ah - lo) / (hi - lo + 1e-8)

    if ds == "mit":
        from load_datasets import MIT_TEST_CELLS
        test_cells = MIT_TEST_CELLS
    else:
        test_cells = [test_cell]

    model, W = load_model(ds, 32, seed)
    if model is None:
        return None

    per_cell = []
    for tcell in test_cells:
        tc = (caps[tcell] - lo) / (hi - lo + 1e-8)
        te = int(np.argmax(caps[tcell] < eol_ah)) if (caps[tcell] < eol_ah).any() else len(tc)
        aes = []
        with torch.no_grad():
            for sp in sps:
                if sp >= te:
                    continue
                tru = te - sp
                pe = -1
                for t in range(sp, te):
                    if t - W < 0:
                        break
                    win = tc[t - W:t]
                    wmean = float(win.mean())
                    wstd = float(win.std()) + 1e-6
                    cin = torch.tensor(win, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEV)
                    pred = model(cin).squeeze(0).cpu().numpy() * wstd + wmean
                    if pred[-1] < threshold_n:
                        for j in range(len(pred) - 1):
                            if pred[j] >= threshold_n > pred[j + 1]:
                                frac = (threshold_n - pred[j]) / (pred[j + 1] - pred[j] + 1e-8)
                                pe = t + j + frac
                                break
                        else:
                            pe = t + len(pred) - 1
                        break
                aes.append(abs(tru - (pe - sp)) if pe >= 0 else np.nan)
        per_cell.append(aes)
    # average across test cells, per SP
    if not per_cell:
        return None
    return [float(np.nanmean([pc[i] for pc in per_cell])) for i in range(len(sps))]


def fmt(vals):
    vals = np.array([v for v in vals if v is not None and not np.isnan(v)], dtype=float)
    if len(vals) == 0:
        return "n/a"
    if vals.std() < 1e-6:
        return f"{vals.mean():.4f}"
    return f"{vals.mean():.4f}±{vals.std():.4f}"


def main():
    print("=== Table A: K=1 per-SP (MAE, RMSE, R2, AE) mean±std over seeds ===")
    for ds in ["calce", "nasa", "mit", "panasonic", "tju"]:
        per_seed = [eval_table_a(ds, s) for s in SEEDS]
        if any(x is None for x in per_seed):
            print(f"{ds}: missing checkpoints")
            continue
        print(f"\n{ds} (SPs {SPS[ds]}):")
        for i, sp in enumerate(SPS[ds]):
            print(f"  SP{sp}: MAE={fmt([x[i][0] for x in per_seed])} "
                  f"RMSE={fmt([x[i][1] for x in per_seed])} "
                  f"R2={fmt([x[i][2] for x in per_seed])} "
                  f"AE={fmt([x[i][3] for x in per_seed])}")

    print("\n=== Table B: K=32 AE per SP mean±std over seeds ===")
    for ds in ["calce", "nasa", "mit", "panasonic", "tju"]:
        per_seed = [eval_table_b(ds, s) for s in SEEDS]
        if any(x is None for x in per_seed):
            print(f"{ds}: missing checkpoints")
            continue
        print(f"{ds}: " + " | ".join(
            f"SP{sp}={fmt([x[i] for x in per_seed])}"
            for i, sp in enumerate(SPS[ds])))


if __name__ == "__main__":
    main()
