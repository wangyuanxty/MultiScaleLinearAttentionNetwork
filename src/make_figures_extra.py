"""Extra paper figures (matplotlib data figures only):
  - fig_regen:   PANASONIC regeneration-zone zoom (I2 evidence)
  - fig_compare: tab:lit data as grouped bars (AMAE per method)
  - fig_pf_stall: PatchFormer AR-32 stall vs Ours (CALCE SP300)

The deployment-verification chain figure is generated separately
with baoyu-image-gen (fig_deploy_chain.png).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from make_figures import predict_series, load_series

CKPT = "D:/research/degradation_prognostics/Transformer_and_Multi_Scale_Models/checkpoints"
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper", "figures")
os.makedirs(FIG, exist_ok=True)

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

plt.rcParams.update(
    {
        "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
        "legend.fontsize": 7, "figure.dpi": 300, "savefig.dpi": 300,
        "savefig.bbox": "tight", "font.family": "serif",
        "mathtext.fontset": "dejavuserif", "axes.grid": True,
        "grid.alpha": 0.3, "grid.linewidth": 0.4,
        "axes.spines.top": False, "axes.spines.right": False,
    }
)


def fig_regen():
    """PANASONIC: full trajectory + zoom on a regeneration zone."""
    pv, tv, lo, hi, W, sps, eol_ah = predict_series("panasonic", K=1)
    caps, _, test_cell, _, _, _ = load_series("panasonic")
    full = caps[test_cell]
    x_full = np.arange(len(full))
    x_pred = np.arange(W, W + len(tv))
    pred = pv[:, 0] * (hi - lo) + lo

    # locate regeneration zones: local rises in true capacity
    d = np.diff(full)
    rises = np.where(d > 0.004)[0]
    # pick the densest cluster of rises within the prediction window
    in_win = rises[(rises >= W + 10) & (rises < len(full) - 10)]
    if len(in_win) > 3:
        best, best_n = in_win[0], 0
        for c in in_win:
            n = np.sum(np.abs(in_win - c) < 40)
            if n > best_n:
                best, best_n = c, n
        z0, z1 = max(best - 45, W), min(best + 45, len(full))
    else:
        z0, z1 = W, len(full)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2))
    ax = axes[0]
    ax.plot(x_full, full, color="0.35", lw=1.1, label="true capacity")
    ax.plot(x_pred, pred, color=COLORS[0], lw=0.9, ls="--", label="prediction (K=1)")
    ax.axhline(eol_ah, color="#d62728", lw=0.9, ls=":", label="EOL 2.12 Ah")
    ax.set_xlabel("cycle")
    ax.set_ylabel("capacity (Ah)")
    ax.set_title("PANASONIC test cell")
    ax.legend(frameon=False, loc="best", fontsize=6.5)

    ax = axes[1]
    ax.plot(x_full[z0:z1], full[z0:z1], color="0.35", lw=1.3, label="true capacity")
    ax.plot(x_pred[z0 - W:z1 - W], pred[z0:z1], color=COLORS[0], lw=1.1,
            ls="--", label="prediction (K=1)")
    for r in in_win:
        if z0 <= r < z1:
            ax.axvline(r, color="#2ca02c", lw=0.5, alpha=0.5)
    ax.set_xlabel("cycle")
    ax.set_ylabel("capacity (Ah)")
    ax.set_title("Zoom: capacity-regeneration zone")
    ax.legend(frameon=False, loc="best", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_regen.pdf"))
    plt.close(fig)
    print("fig_regen.pdf done")


def fig_compare():
    """tab:lit AMAE as grouped bars (log scale), Ours highlighted."""
    data = {
        "NASA": {
            "TimeMixer": 0.0228, "TimesNet": 0.0292, "PatchTST": 0.0204,
            "MambaSimple": 0.0246, "ModernTCN": 0.0130, "Autoformer": 0.0226,
            "FEDformer": 0.0195, "iTransformer": 0.0080, "PathFormer": 0.0225,
            "PatchFormer": 0.0062, "RUL-Mamba": 0.0089, "Ours": 0.0108,
        },
        "TJU": {
            "TimeMixer": 0.0093, "TimesNet": 0.0146, "PatchTST": 0.0071,
            "MambaSimple": 0.0107, "ModernTCN": 0.0015, "Autoformer": 0.0049,
            "FEDformer": 0.0056, "iTransformer": 0.0018, "PathFormer": 0.0121,
            "PatchFormer": 0.0014, "RUL-Mamba": 0.0030, "Ours": 0.0064,
        },
        "CALCE": {
            "TimeMixer": 0.0291, "TimesNet": 0.0461, "PatchTST": 0.0235,
            "MambaSimple": 0.0284, "ModernTCN": 0.0127, "Autoformer": 0.0246,
            "FEDformer": 0.0227, "iTransformer": 0.0168, "PathFormer": 0.0541,
            "PatchFormer": 0.0063, "Ours": 0.0139,
        },
    }
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1), sharey=True)
    for ax, (ds, d) in zip(axes, data.items()):
        names = list(d.keys())
        vals = [d[n] for n in names]
        colors = []
        for n in names:
            if n == "Ours":
                colors.append("#2ca02c")
            elif n in ("PatchFormer", "RUL-Mamba"):
                colors.append("#1f77b4")
            else:
                colors.append("#b0b0b0")
        ax.bar(range(len(names)), vals, color=colors, width=0.62)
        ax.set_yscale("log")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=6.5)
        ax.set_title(ds)
        if ds == "NASA":
            ax.set_ylabel("average MAE (log)")
    axes[0].legend(
        handles=[plt.Rectangle((0, 0), 1, 1, fc="#2ca02c"),
                 plt.Rectangle((0, 0), 1, 1, fc="#1f77b4"),
                 plt.Rectangle((0, 0), 1, 1, fc="#b0b0b0")],
        labels=["Ours", "PatchFormer / RUL-Mamba", "other baselines"],
        frameon=False, loc="upper left", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_compare.pdf"))
    plt.close(fig)
    print("fig_compare.pdf done")


def fig_pf_stall():
    """PatchFormer AR-32 stall vs Ours K=32 window (CALCE).

    Shows: (a) PatchFormer's true AR rollout never crosses the EOL
    threshold; (b) our K=32 window launched 32 cycles before the true
    EOL crosses the threshold ~16.9 cycles early.
    """
    import torch
    from gdn_model import build_gdn_model

    npz = os.path.join(CKPT, "pf_ar_sp300.npz")
    if not os.path.exists(npz):
        print("fig_pf_stall skipped (no pf_ar_sp300.npz)")
        return
    z = np.load(npz)
    preds_n = z["preds_norm"]
    true_eol = int(z["true_eol"])
    x_min, x_max = float(z["x_min"]), float(z["x_max"])
    sp = int(z["sp"])

    caps, _, test_cell, _, _, _ = load_series("calce")
    full = caps[test_cell]
    x_full = np.arange(len(full))

    pf_pred = preds_n * (x_max - x_min) + x_min
    x_pf = sp + np.arange(len(pf_pred))

    # Ours K=32 early-sensing window: find the first window (stride-1
    # scan, Table B protocol) whose last predicted value falls below
    # the EOL threshold, then plot that window's one-shot trajectory.
    # Normalization must match the model's training (load_series
    # train-cells min/max), NOT the PatchFormer repo's x_min/x_max.
    pv, tv, lo, hi, W, sps, eol = predict_series("calce", K=1)
    tc = (full - lo) / (hi - lo + 1e-8)
    model = build_gdn_model(multiscale=True, cross_exchange=True,
                            input_dim=1, window_size=W, output_len=32,
                            readout="last")
    model.load_state_dict(torch.load(f"{CKPT}/unified_calce_K32.pt",
                                     map_location="cpu", weights_only=True))
    model.eval()
    t0, p32, cross = None, None, None
    with torch.no_grad():
        for t in range(W, true_eol):
            cin = torch.tensor(tc[t - W:t], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            p = model(cin).squeeze(0).numpy()
            if p[-1] < (0.77 - lo) / (hi - lo + 1e-8):
                t0 = t
                p32 = p * (hi - lo) + lo
                for j in range(len(p32) - 1):
                    if p32[j] >= 0.77 > p32[j + 1]:
                        frac = (0.77 - p32[j]) / (p32[j + 1] - p32[j] + 1e-8)
                        cross = t0 + j + frac
                        break
                break
    if t0 is None:
        print("fig_pf_stall skipped (no crossing window found)")
        return
    x32 = t0 + np.arange(32)

    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    ax.plot(x_full, full, color="0.35", lw=1.2, label="true capacity")
    ax.plot(x_pf, pf_pred, color="#d62728", lw=1.0, ls=":",
            label="PatchFormer AR rollout (never crosses EOL)")
    ax.plot(x32, p32, color=COLORS[0], lw=1.4, ls="--",
            label="Ours K=32 window (one-shot)")
    ax.axhline(0.77, color="#8e44ad", lw=1.0, ls="-.",
               label="EOL 0.77 Ah")
    ax.axvline(true_eol, color="#2ca02c", lw=1.0, ls="--",
               label=f"true EOL ({true_eol})")
    ax.axvline(cross, color=COLORS[0], lw=1.0, ls="--",
               label=f"predicted EOL ({cross:.1f}, {true_eol - cross:.1f} cyc early)")
    ax.annotate("AR rollout stalls above EOL\n(AE = N/A)",
                xy=(x_pf[-1], pf_pred[-1]), xytext=(sp + 45, 0.92),
                fontsize=6.5, arrowprops=dict(arrowstyle="->", lw=0.7),
                color="#d62728")
    ax.annotate("K=32 window starts here,\none-shot to cycle " + str(t0 + 31),
                xy=(t0, p32[0]), xytext=(t0 - 78, 0.92),
                fontsize=6.5, arrowprops=dict(arrowstyle="->", lw=0.7),
                color=COLORS[0])
    ax.set_xlabel("cycle")
    ax.set_ylabel("capacity (Ah)")
    ax.set_title("K=32 early sensing vs PatchFormer AR rollout (CALCE)")
    ax.legend(frameon=False, loc="lower left", fontsize=6.5)
    ax.set_xlim(max(sp - 30, 0), len(full) + 5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_pf_stall.pdf"))
    plt.close(fig)
    print(f"fig_pf_stall.pdf done (pred crossing {cross:.1f}, true {true_eol})")


if __name__ == "__main__":
    fig_regen()
    fig_compare()
    fig_pf_stall()
