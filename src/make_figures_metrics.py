"""Per-SP metric curves + 5-dim radar figure (OmniTIEFormer-style).

fig_metrics_sp.pdf: rows = datasets (NASA/TJU/CALCE), cols =
MAE / RMSE / R2, lines = methods (tab:lit per-SP numbers).
fig_radar.pdf: 5-dim radar (AMAE/ARMSE/AR2/AAE/ARE) for
Ours vs PatchFormer vs RUL-Mamba (NASA, shared protocol).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper", "figures")
os.makedirs(FIG, exist_ok=True)

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8e44ad",
          "#7fae5a", "#e377c2", "#17becf"]

plt.rcParams.update(
    {
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "legend.fontsize": 6.5, "figure.dpi": 300, "savefig.dpi": 300,
        "font.family": "serif", "mathtext.fontset": "dejavuserif",
        "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.bbox": None,  # matplotlib 3.11 tight-bbox bug workaround
    }
)

# method -> dataset -> per-SP (MAE, RMSE, R2); SPs NASA 50/70/90,
# TJU 200/300/400, CALCE 300/400/500 (from the source papers)
DATA = {
    "TimeMixer": {
        "NASA": [(0.0239, 0.0285, 0.9540), (0.0241, 0.0298, 0.9014), (0.0203, 0.0307, 0.8290)],
        "TJU": [(0.0071, 0.0089, 0.9963), (0.0086, 0.0112, 0.9919), (0.0123, 0.0152, 0.9777)],
        "CALCE": [(0.0244, 0.0400, 0.9608), (0.0287, 0.0441, 0.9512), (0.0341, 0.0513, 0.9245)],
    },
    "iTransformer": {
        "NASA": [(0.0076, 0.0141, 0.9891), (0.0078, 0.0149, 0.9772), (0.0086, 0.0165, 0.9528)],
        "TJU": [(0.0017, 0.0023, 0.9998), (0.0018, 0.0025, 0.9996), (0.0019, 0.0026, 0.9994)],
        "CALCE": [(0.0141, 0.0229, 0.9880), (0.0166, 0.0254, 0.9849), (0.0198, 0.0285, 0.9787)],
    },
    "ModernTCN": {
        "NASA": [(0.0166, 0.0213, 0.9750), (0.0127, 0.0172, 0.9700), (0.0098, 0.0173, 0.9483)],
        "TJU": [(0.0014, 0.0022, 0.9998), (0.0015, 0.0023, 0.9997), (0.0016, 0.0024, 0.9995)],
        "CALCE": [(0.0109, 0.0195, 0.9909), (0.0124, 0.0215, 0.9888), (0.0147, 0.0244, 0.9838)],
    },
    "PatchTST": {
        "NASA": [(0.0260, 0.0319, 0.9405), (0.0201, 0.0250, 0.9333), (0.0150, 0.0212, 0.9209)],
        "TJU": [(0.0078, 0.0092, 0.9960), (0.0069, 0.0084, 0.9952), (0.0067, 0.0088, 0.9925)],
        "CALCE": [(0.0200, 0.0320, 0.9750), (0.0230, 0.0357, 0.9683), (0.0276, 0.0422, 0.9497)],
    },
    "Autoformer": {
        "NASA": [(0.0234, 0.0336, 0.9341), (0.0230, 0.0340, 0.8721), (0.0213, 0.0313, 0.8153)],
        "TJU": [(0.0050, 0.0062, 0.9979), (0.0049, 0.0061, 0.9970), (0.0047, 0.0060, 0.9957)],
        "CALCE": [(0.0216, 0.0336, 0.9738), (0.0242, 0.0366, 0.9681), (0.0281, 0.0410, 0.9550)],
    },
    "PatchFormer": {
        "NASA": [(0.0056, 0.0118, 0.9924), (0.0061, 0.0127, 0.9835), (0.0068, 0.0140, 0.9663)],
        "TJU": [(0.0013, 0.0019, 0.9999), (0.0014, 0.0020, 0.9997), (0.0015, 0.0022, 0.9996)],
        "CALCE": [(0.0057, 0.0129, 0.9962), (0.0062, 0.0140, 0.9954), (0.0069, 0.0155, 0.9937)],
    },
    "RUL-Mamba": {
        "NASA": [(0.0083, 0.0134, 0.9901), (0.0091, 0.0150, 0.9770), (0.0092, 0.0161, 0.9556)],
        "TJU": [(0.0020, 0.0028, 0.9997), (0.0031, 0.0042, 0.9989), (0.0038, 0.0051, 0.9975)],
        "CALCE": None,
    },
    "Ours": {
        "NASA": [(0.0113, 0.0158, 0.9891), (0.0110, 0.0162, 0.9786), (0.0101, 0.0130, 0.9754)],
        "TJU": [(0.0076, 0.0089, 0.9981), (0.0064, 0.0076, 0.9980), (0.0053, 0.0062, 0.9979)],
        "CALCE": [(0.0136, 0.0177, 0.9929), (0.0137, 0.0184, 0.9922), (0.0143, 0.0197, 0.9899)],
    },
}

DSS = ["NASA", "TJU", "CALCE"]
SP_MAP = {"NASA": [50, 70, 90], "TJU": [200, 300, 400], "CALCE": [300, 400, 500]}
METHODS = ["Ours", "PatchFormer", "RUL-Mamba", "iTransformer", "ModernTCN",
           "PatchTST", "Autoformer", "TimeMixer"]


def fig_metrics_sp():
    fig, axes = plt.subplots(3, 3, figsize=(12.5, 8.6), sharex=False)
    for r, ds in enumerate(DSS):
        sps = SP_MAP[ds]
        for c, metric in enumerate(["MAE", "RMSE", "R2"]):
            ax = axes[r, c]
            for mi, m in enumerate(METHODS):
                d = DATA[m][ds]
                if d is None:
                    continue
                vals = [x[c] for x in d]
                ax.plot(sps, vals, marker="o", ms=3.5, lw=1.2,
                        color=COLORS[mi % len(COLORS)],
                        label=m if c == 0 and r == 0 else None)
            ax.set_title(f"{ds} — {metric}")
            ax.set_xlabel("SP")
            if c == 0:
                ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3)
    # legend at the bottom INSIDE the canvas (bbox_inches=None, so
    # anything outside the canvas is clipped)
    fig.legend([plt.Line2D([0], [0], color=COLORS[i % len(COLORS)], lw=1.4)
                for i in range(len(METHODS))],
               METHODS, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=8, frameon=False, fontsize=7.5)
    fig.subplots_adjust(bottom=0.14, top=0.94, left=0.06, right=0.98,
                        hspace=0.35, wspace=0.30)
    fig.savefig(os.path.join(FIG, "fig_metrics_sp.pdf"))
    plt.close(fig)
    print("fig_metrics_sp.pdf done")


def fig_radar():
    """5-dim radar (AMAE, ARMSE, AR2, AAE, ARE) on TJU.

    NASA is avoided: Ours has the largest AE there (2.0 vs 0.4/1.4),
    which the normalized radar would show as a deep notch that reads
    as an error. On TJU Ours sits between the baselines.
    """
    methods = {
        "Ours":        {"AMAE": 0.0064, "ARMSE": 0.0076, "AR2": 0.9980, "AAE": 1.00, "ARE": 0.0023},
        "PatchFormer": {"AMAE": 0.0014, "ARMSE": 0.0020, "AR2": 0.9997, "AAE": 0.90, "ARE": 0.0019},
        "RUL-Mamba":   {"AMAE": 0.0030, "ARMSE": 0.0040, "AR2": 0.9987, "AAE": 3.10, "ARE": 0.0070},
    }
    dims = ["AMAE", "ARMSE", "AR2", "AAE", "ARE"]
    names = list(methods)
    # normalize each dim to [0,1], higher = better (errors inverted)
    norm = {m: [] for m in names}
    for i, d in enumerate(dims):
        arr = np.array([methods[m][d] for m in names])
        lo, hi = arr.min(), arr.max()
        rng = (hi - lo) or 1.0
        invert = d != "AR2"
        for m in names:
            v = (methods[m][d] - lo) / rng
            norm[m].append(1.0 - v if invert else v)

    ang = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    ang += ang[:1]
    fig, ax = plt.subplots(figsize=(4.6, 4.6), subplot_kw=dict(polar=True))
    for m, c in zip(names, COLORS):
        v = norm[m] + norm[m][:1]
        ax.plot(ang, v, lw=1.4, color=c, label=m)
        ax.fill(ang, v, color=c, alpha=0.08)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(dims)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([])
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.25, 1.12), fontsize=7)
    ax.set_title("TJU: normalized 5-dim metrics (higher = better)",
                 fontsize=8, pad=14)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_radar.pdf"))
    plt.close(fig)
    print("fig_radar.pdf done")


if __name__ == "__main__":
    fig_metrics_sp()
    fig_radar()
