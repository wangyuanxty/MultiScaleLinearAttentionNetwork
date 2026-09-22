"""Redraw fig_robust from the ten-seed checkpoints.

make_figures_phys.fig_robust() trains its own models with a fixed seed of 42 and
corrupts the input with a fixed RNG seed.  tab:robust, however, reports seeds 1-10
under corrupt_capacity(tc, mode, seed), so the figure and the table quote
different runs -- the figure's caption said "AE 23 / AE 17" while the table said
59.6 / 3.4, and neither number appears in the other.

This script rebuilds the same figure from the checkpoints that tab:robust was
computed from, so the two finally show one run:

  checkpoints/phys_ir_seeds/seed{n}_std.pt   rate head  (phys_ir readout)
  checkpoints/abl_seed/single/seed{n}.pt     free head  (z-score readout)

No training: both are loaded and run forward.

Usage: python src/make_figures_robust_10seed.py [seed]     (default 4)
Writes: paper/fig_robust.pdf, figures_export/fig_robust.png
"""

import os
import sys

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from gdn_model import build_gdn_model                          # noqa: E402
from eval_multiseed import true_rul                            # noqa: E402
from train_per_sp import window_std                            # noqa: E402
from test_physics_ir_seeds import (                            # noqa: E402
    load_data, corrupt_capacity, EPS, DEV, IR_IDX,
)

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 4
PAPER = os.path.join(ROOT, "paper")
EXPORT = os.path.join(ROOT, "figures_export")
CK_RATE = os.path.join(ROOT, "checkpoints", "phys_ir_seeds", f"seed{SEED}_std.pt")
CK_FREE = os.path.join(ROOT, "checkpoints", "abl_seed", "single", f"seed{SEED}.pt")

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
    "legend.fontsize": 6, "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": None, "font.family": "serif",
    "mathtext.fontset": "dejavuserif", "axes.grid": True, "grid.alpha": 0.3,
    "grid.linewidth": 0.4, "axes.spines.top": False, "axes.spines.right": False,
})


def predict_free(zck, caps, test_cell, lo, hi, W, mode):
    """z-score free head on the corrupted capacity channel."""
    model = build_gdn_model(multiscale=False, input_dim=1, window_size=W,
                            output_len=1, readout="last").to(DEV)
    model.load_state_dict(zck["state_dict"])
    model.eval()
    tc_clean = (caps[test_cell] - lo) / (hi - lo + EPS)
    tc = corrupt_capacity(tc_clean, mode, SEED)
    out = []
    with torch.no_grad():
        for i in range(W, len(tc)):
            win = tc[i - W:i]
            wmean = float(win.mean())
            wstd = window_std(win) + EPS
            cin = torch.tensor(win, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEV)
            out.append(model(cin).item() * wstd + wmean)
    return np.array(out), tc_clean


def predict_rate(caps, feats, test_cell, lo_c, hi_c, lo_i, hi_i, W, mode):
    """physics rate head: capacity + IR, absolute normalized space."""
    m = build_gdn_model(multiscale=False, input_dim=2, window_size=W,
                        output_len=1, readout="phys_ir").to(DEV)
    m.load_state_dict(torch.load(CK_RATE, map_location=DEV)["state_dict"])
    m.eval()
    tc_clean = (caps[test_cell] - lo_c) / (hi_c - lo_c + EPS)
    ti = (feats[test_cell][:, IR_IDX] - lo_i) / (hi_i - lo_i + EPS)
    tc = corrupt_capacity(tc_clean, mode, SEED)
    out = []
    with torch.no_grad():
        for i in range(W, len(tc)):
            win = np.stack([tc[i - W:i], ti[i - W:i]], axis=1)
            cin = torch.tensor(win, dtype=torch.float32).unsqueeze(0).to(DEV)
            out.append(m(cin).item())
    return np.array(out), tc


def main():
    for p in (CK_RATE, CK_FREE):
        if not os.path.exists(p):
            raise SystemExit(f"missing checkpoint: {p}")

    caps, feats, train_cells, test_cell, W, sps, eol_ah, \
        lo_c, hi_c, lo_i, hi_i = load_data()

    free, tc_clean = predict_free(torch.load(CK_FREE, map_location=DEV),
                                  caps, test_cell, lo_c, hi_c, W, "drop30")
    rate, tc = predict_rate(caps, feats, test_cell, lo_c, hi_c, lo_i, hi_i,
                            W, "drop30")

    n = min(len(tc_clean) - W, len(free), len(rate))
    truth = tc_clean[W:W + n]
    free, rate = free[:n], rate[:n]
    corr = tc[W:W + n]
    x = np.arange(W, W + n)

    th = (eol_ah - lo_c) / (hi_c - lo_c + EPS)
    ae_free = abs(true_rul(truth, th) - true_rul(free, th))
    ae_rate = abs(true_rul(truth, th) - true_rul(rate, th))
    print(f"seed {SEED}: drop30 AE  free={ae_free:.0f}  rate={ae_rate:.0f}")

    cross = int(np.argmax(truth < th))
    z0, z1 = max(0, cross - 60), min(n, cross + 60)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].plot(x, truth, color="black", lw=1.2, label="true")
    axes[0].plot(x, corr, color="gray", lw=0.7, alpha=0.7,
                 label="corrupted input (drop30)")
    axes[0].plot(x, free, color="tab:red", lw=1.0, alpha=0.9,
                 label=f"free head (AE {ae_free:.0f})")
    axes[0].plot(x, rate, color="tab:green", lw=1.2,
                 label=f"rate head + IR (AE {ae_rate:.0f})")
    axes[0].axhline(th, color="red", ls=":", lw=0.7)
    axes[0].set_title(f"drop30: full trajectory (seed {SEED})")
    axes[0].set_xlabel("cycle")
    axes[0].set_ylabel("normalized capacity")
    axes[0].legend(fontsize=6)

    axes[1].plot(x[z0:z1], truth[z0:z1], color="black", lw=1.3, marker="o",
                 ms=2, label="true")
    axes[1].plot(x[z0:z1], corr[z0:z1], color="gray", lw=0.7, alpha=0.7,
                 label="corrupted input")
    axes[1].plot(x[z0:z1], free[z0:z1], color="tab:red", lw=1.0, label="free head")
    axes[1].plot(x[z0:z1], rate[z0:z1], color="tab:green", lw=1.2,
                 label="rate head + IR")
    axes[1].axhline(th, color="red", ls=":", lw=0.7)
    axes[1].set_title("drop30: EOL-crossing zoom")
    axes[1].set_xlabel("cycle")
    axes[1].legend(fontsize=6)

    fig.tight_layout()
    fig.savefig(os.path.join(PAPER, "fig_robust.pdf"))
    fig.savefig(os.path.join(EXPORT, "fig_robust.png"), dpi=150)
    plt.close(fig)
    print("saved fig_robust.pdf / .png")


if __name__ == "__main__":
    main()
