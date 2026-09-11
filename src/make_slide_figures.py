"""Render slide-only variants of paper figures (never touches paper/figures).

Writes wide-layout PNGs into slides_assets/ for the group-meeting deck.
Currently: fig_compare in a 2x3 layout — the paper's 3x2 version is
1318x1453 (portrait) and cannot be shown legibly on a 16:9 slide.

Data is identical to make_figures_extra.fig_compare (mean MAE over the
three SPs, taken from the tab:lit tables).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "slides_assets")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    # savefig.bbox must be None: importing make_figures leaves its
    # 'tight' setting active and breaks mathtext canvases (see CLAUDE.md).
    "savefig.bbox": None,
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Same numbers as make_figures_extra.fig_compare.
DATA = {
    "NASA": {
        "TimeMixer": 0.0228, "TimesNet": 0.0292, "PatchTST": 0.0204,
        "MambaSimple": 0.0246, "ModernTCN": 0.0130, "Autoformer": 0.0226,
        "FEDformer": 0.0195, "iTransformer": 0.0080, "PathFormer": 0.0225,
        "PatchFormer": 0.0068, "RUL-Mamba": 0.0090, "Ours": 0.0089,
    },
    "TJU": {
        "TimeMixer": 0.0093, "TimesNet": 0.0146, "PatchTST": 0.0071,
        "MambaSimple": 0.0107, "ModernTCN": 0.0015, "Autoformer": 0.0049,
        "FEDformer": 0.0056, "iTransformer": 0.0018, "PathFormer": 0.0121,
        "PatchFormer": 0.0016, "RUL-Mamba": 0.0015, "Ours": 0.0015,
    },
    "CALCE": {
        "TimeMixer": 0.0291, "TimesNet": 0.0461, "PatchTST": 0.0235,
        "MambaSimple": 0.0284, "ModernTCN": 0.0127, "Autoformer": 0.0246,
        "FEDformer": 0.0227, "iTransformer": 0.0168, "PathFormer": 0.0541,
        "PatchFormer": 0.0069, "RUL-Mamba": 0.0207, "Ours": 0.0079,
    },
    "PANASONIC": {
        "iTransformer": 0.0180, "Autoformer": 0.0245, "FEDformer": 0.0274,
        "PatchFormer": 0.0079, "RUL-Mamba": 0.0189, "Ours": 0.0038,
    },
    "MIT": {
        "PatchFormer": 0.0008, "RUL-Mamba": 0.0016, "Ours": 0.0025,
    },
    "GOTION": {
        "ModernTCN": 0.0581, "Autoformer": 0.0731, "FEDformer": 0.0749,
        "iTransformer": 0.0597, "PathFormer": 0.1965,
        "PatchFormer": 0.0568, "RUL-Mamba": 0.0558, "Ours": 0.0080,
    },
}

C_OURS, C_BASE, C_LIT = "#2ca02c", "#1f77b4", "#b0b0b0"


def color_of(name):
    if name == "Ours":
        return C_OURS
    if name in ("PatchFormer", "RUL-Mamba"):
        return C_BASE
    return C_LIT


def fig_compare_wide():
    """2 rows x 3 cols so the slide can show it at full width."""
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 4.9), sharey=True)
    for ax, (ds, d) in zip(axes.ravel(), DATA.items()):
        names = list(d.keys())
        vals = [d[n] for n in names]
        ax.bar(range(len(names)), vals,
               color=[color_of(n) for n in names], width=0.62)
        ax.set_yscale("log")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_title(ds, fontsize=10)
        ax.grid(axis="x", visible=False)
    axes[0, 0].set_ylabel("average MAE (log)", fontsize=8.5)
    axes[1, 0].set_ylabel("average MAE (log)", fontsize=8.5)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=C_OURS),
               plt.Rectangle((0, 0), 1, 1, fc=C_BASE),
               plt.Rectangle((0, 0), 1, 1, fc=C_LIT)]
    fig.legend(handles,
               ["Ours", "PatchFormer / RUL-Mamba", "other baselines"],
               loc="upper center", ncol=3, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    path = os.path.join(OUT, "fig_compare.png")
    fig.savefig(path, dpi=190)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    fig_compare_wide()
