"""C3 analysis: K=32 trajectory error vs prediction horizon k.

For every real window of the test cell, run a K=32 forward pass and
compare step k's prediction against the true capacity k steps later.
Reports MAE(k) and mean bias(k) for k=1..32, on CALCE and NASA.
Outputs fig_horizon.pdf + prints the numbers for the paper text.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from make_figures import predict_series, load_series
from gdn_model import build_gdn_model

CKPT = "D:/research/degradation_prognostics/Transformer_and_Multi_Scale_Models/checkpoints"
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper", "figures")
os.makedirs(FIG, exist_ok=True)

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

plt.rcParams.update(
    {
        "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
        "legend.fontsize": 7, "figure.dpi": 300, "savefig.dpi": 300,
        "font.family": "serif", "mathtext.fontset": "dejavuserif",
        "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.bbox": None,  # matplotlib 3.11 tight-bbox bug workaround
    }
)


def horizon_errors(ds, seeds=(42, 43, 44)):
    """Return (k, mae, bias, n) for dataset ds, averaged over seeds."""
    caps, train_cells, test_cell, W, sps, eol_ah = load_series(ds)
    all_tr = np.concatenate([caps[c] for c in train_cells])
    lo, hi = all_tr.min(), all_tr.max()
    full = caps[test_cell]
    tc = (full - lo) / (hi - lo + 1e-8)
    windows = np.stack([tc[i - W:i] for i in range(W, len(tc))])
    cin = torch.tensor(windows, dtype=torch.float32).unsqueeze(-1)

    preds = []
    for seed in seeds:
        path = f"{CKPT}/unified_{ds}_K32_seed{seed}.pt"
        if not os.path.exists(path) and seed == 42:
            path = f"{CKPT}/unified_{ds}_K32.pt"
        model = build_gdn_model(multiscale=True, cross_exchange=True,
                                input_dim=1, window_size=W, output_len=32,
                                readout="last")
        model.load_state_dict(torch.load(path, map_location="cpu",
                                         weights_only=True))
        model.eval()
        with torch.no_grad():
            preds.append(model(cin).numpy())  # (N, 32) normalized
    pred = np.mean(preds, axis=0)

    K = 32
    k_mae, k_bias, k_n = [], [], []
    for k in range(1, K + 1):
        nv = max(len(tc) - W - k, 0)
        err = pred[:nv, k - 1] - tc[W + k: W + k + nv]
        k_mae.append(np.mean(np.abs(err)))
        k_bias.append(np.mean(err))
        k_n.append(nv)
    return np.arange(1, K + 1), np.array(k_mae), np.array(k_bias), np.array(k_n)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.0), sharey=True)
    for ax, ds, title in zip(axes, ["calce", "nasa"], ["CALCE (CS2-35)", "NASA (B0005)"]):
        k, mae, bias, n = horizon_errors(ds)
        ax.plot(k, mae, color=COLORS[0], lw=1.3, label="MAE")
        ax.plot(k, bias, color="#d62728", lw=1.1, ls="--",
                label="mean bias (pred - true)")
        ax.axhline(0, color="0.4", lw=0.7, ls=":")
        ax.set_xlabel("horizon $k$ (steps ahead)")
        ax.set_ylabel("normalized capacity error")
        ax.set_title(f"{title}  (n={n[0]} windows)")
        ax.legend(frameon=False)
        print(f"{ds}: k=1 MAE={mae[0]:.4f} bias={bias[0]:+.4f} | "
              f"k=8 MAE={mae[7]:.4f} bias={bias[7]:+.4f} | "
              f"k=32 MAE={mae[-1]:.4f} bias={bias[-1]:+.4f}")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_horizon.pdf"))
    plt.close(fig)
    print("fig_horizon.pdf done")


if __name__ == "__main__":
    main()
