"""NASA ablation under both std conventions, from the stored checkpoints.

eval_fullseq() decodes with numpy's .std() (biased, ddof=0) while training
used torch's .std (unbiased, ddof=1).  The decode is y = z * wstd + wmean,
so the convention enters every reported number.  At W=30 the two differ by
a factor sqrt(30/29) = 1.0171.

This script re-runs the same decode both ways from the same checkpoints so
the convention can be ruled in or out as the reason the paper's NASA row
(single 0.0087+/-0.0002, multi 0.0098+/-0.0014, main 0.0084+/-0.0004) does
not match any stored result.

The decode is written out here rather than imported because eval_fullseq
computes its window sd inline, so the convention cannot be swapped from
outside.  The sd is computed by hand (sum of squares over n or n-1) rather
than through torch.std, so the two conventions are explicit and the script
does not depend on whether this torch build spells the flag
`unbiased=` or `correction=`.

Reads only: checkpoints/abl_seed/nasa/{single,multi,xchg}/seed{1..10}.pt
Writes: nothing.

Usage: cd src && python eval_abl_nasa_std_convention.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import train_ablation_fullseq as T          # noqa: E402
from eval_multiseed import true_rul         # noqa: E402

DS = "nasa"
SEEDS = tuple(range(1, 11))
CONFIGS = {
    "single": {"multiscale": False, "stage_query": False},
    "multi": {"multiscale": True, "stage_query": False},
    "xchg": {"multiscale": True, "stage_query": True},
}
ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "checkpoints", "abl_seed", DS,
)
EPS = 1e-6


def window_sd(win, unbiased):
    """sd of a 1-D float tensor; n-1 denominator if unbiased, n otherwise."""
    n = win.numel()
    ss = float(((win - win.mean()) ** 2).sum())
    return float(np.sqrt(ss / (n - 1 if unbiased else n)))


def decode(model, caps, test_cell, W, lo, hi, eol_ah, unbiased):
    """Full-sequence single-value metrics, with a selectable window sd."""
    tc = (np.asarray(caps[test_cell], dtype=np.float64) - lo) / (hi - lo + EPS)
    seg_p = []
    with torch.no_grad():
        for i in range(W, len(tc)):
            win = torch.tensor(tc[i - W:i], dtype=torch.float32)
            cin = win.view(1, W, 1).to(T.DEV)
            wmean = float(win.mean())
            wstd = window_sd(win, unbiased) + EPS
            seg_p.append(float(model(cin).item()) * wstd + wmean)
    seg_p = np.array(seg_p)
    tv = tc[W:]
    mae = float(np.mean(np.abs(seg_p - tv)))
    r2 = 1 - np.sum((tv - seg_p) ** 2) / (np.sum((tv - tv.mean()) ** 2) + EPS)
    th = (eol_ah - lo) / (hi - lo + EPS)
    ae = abs(true_rul(tv, th) - true_rul(seg_p, th))
    return mae, r2, ae


def main() -> None:
    caps, train_cells, test_cell, W, sps, eol_ah = T.load_series(DS)
    all_tr = np.concatenate([np.asarray(caps[c], dtype=np.float64) for c in train_cells])
    lo, hi = float(all_tr.min()), float(all_tr.max())
    print(f"{DS}: W={W} test_cell={test_cell}  "
          f"unbiased/biased sd ratio = sqrt({W}/{W - 1}) = {np.sqrt(W / (W - 1)):.4f}\n")
    print(f"{'arm':>8} {'':>10} | {'MAE':>18} {'R2':>8} {'AE':>5}")

    for name, cfg in CONFIGS.items():
        for unbiased in (True, False):
            ms = []
            for seed in SEEDS:
                p = os.path.join(ROOT, name, f"seed{seed}.pt")
                if not os.path.exists(p):
                    continue
                ck = torch.load(p, map_location="cpu", weights_only=False)
                sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
                model = T.build_gdn_model(
                    multiscale=cfg["multiscale"], stage_query=cfg["stage_query"],
                    input_dim=1, window_size=W, output_len=1, readout="last",
                ).to(T.DEV)
                model.load_state_dict(sd)
                ms.append(decode(model, caps, test_cell, W, lo, hi, eol_ah, unbiased))
            a = np.array(ms)
            tag = "unbiased" if unbiased else "biased"
            print(f"{name:>8} {tag:>10} | "
                  f"{a[:, 0].mean():.4f}+/-{a[:, 0].std(ddof=0):.4f} "
                  f"{a[:, 1].mean():>8.4f} {a[:, 2].mean():>5.2f}")


if __name__ == "__main__":
    main()
