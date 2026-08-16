"""Verify: does test-sequence truncation (PatchFormer style, test
starts at SP-W) change per-SP AE for our globally-normalized model?

Compares:
  (a) full-sequence non-recursive eval (current protocol) — slice pv[sp-W:]
  (b) truncated eval — only run windows starting at SP-W, same slices
Expected: identical AE (truncation only skips earlier windows), but
verify empirically per user request.
"""
import sys
import numpy as np
import torch

sys.path.insert(0, '.')
from gdn_model import build_gdn_model
from make_figures import load_series
from eval_multiseed import true_rul, SEEDS

CKPT = "D:/research/degradation_prognostics/Transformer_and_Multi_Scale_Models/checkpoints"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_full(model, tc, W, sps, th):
    """Current protocol: predict ALL windows, slice per SP."""
    wins = np.stack([tc[i - W:i] for i in range(W, len(tc))])
    cin = torch.tensor(wins, dtype=torch.float32).unsqueeze(-1).to(DEV)
    with torch.no_grad():
        pv = model(cin).cpu().numpy()[:, 0]
    pv = pv[: len(tc) - W]
    out = []
    for sp in sps:
        seg_p, seg_t = pv[sp - W:], tc[sp:]
        n = min(len(seg_p), len(seg_t))
        ae = abs(true_rul(seg_t[:n], th) - true_rul(seg_p[:n], th))
        out.append(ae)
    return out


def eval_truncated(model, tc, W, sps, th):
    """PatchFormer style: for each SP, only run windows from SP-W."""
    out = []
    for sp in sps:
        n = len(tc) - sp
        wins = np.stack([tc[sp - W + i: sp + i] for i in range(n)])
        cin = torch.tensor(wins, dtype=torch.float32).unsqueeze(-1).to(DEV)
        with torch.no_grad():
            seg_p = model(cin).cpu().numpy()[:, 0]
        seg_t = tc[sp:]
        ae = abs(true_rul(seg_t, th) - true_rul(seg_p, th))
        out.append(ae)
    return out


if __name__ == "__main__":
    caps, train_cells, test_cell, W, sps, eol_ah = load_series("calce")
    all_tr = np.concatenate([caps[c] for c in train_cells])
    lo, hi = all_tr.min(), all_tr.max()
    tc = (caps[test_cell] - lo) / (hi - lo + 1e-8)
    th = (eol_ah - lo) / (hi - lo + 1e-8)

    for seed in SEEDS:
        path = f"{CKPT}/unified_calce_K1_seed{seed}.pt"
        model = build_gdn_model(multiscale=True, cross_exchange=True,
                                input_dim=1, window_size=W, output_len=1,
                                readout="last").to(DEV)
        model.load_state_dict(torch.load(path, map_location=DEV,
                                         weights_only=True))
        model.eval()
        full = eval_full(model, tc, W, sps, th)
        trunc = eval_truncated(model, tc, W, sps, th)
        print(f"seed {seed}: full={full} truncated={trunc} "
              f"{'IDENTICAL' if full == trunc else 'DIFFERENT!'}")
