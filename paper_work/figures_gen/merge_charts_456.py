#!/usr/bin/env python3
"""Merge the three §3.2 evidence charts into one 1x3 vector figure.

Replaces the low-resolution raster PDFs (fig_decay / fig_complexity /
fig_patch_decomp, all 456x296-embedded bitmaps) with a single
vector-quality figure drawn from the real data in data/:
  (a) per-layer mean decay weights (decay.dat, seed-42 CALCE ckpt)
  (b) per-window cost: O(L) vs O(L^2) (concept curve)
  (c) real multi-scale capacity series, W=64 window, P in {2,4,8}

Outputs paper/figures/fig_method_evidence.{pdf,png} (PNG preview at
200 dpi; PDF is pure vector).
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent
D = ROOT / "data"
FIG = (ROOT / ".." / "figures").resolve()
FIG.mkdir(exist_ok=True)

INK = "#1F262B"
MUTED = "#707A82"
RED = "#D64533"
TEAL = "#2E7D8C"
BLUE_MID = "#6B8FB0"
TERRA = "#C96A4A"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "cm", "savefig.bbox": None,
    "axes.edgecolor": INK, "text.color": INK,
})


def panel_decay(ax):
    d = np.loadtxt(D / "decay.dat")
    ax.bar(d[:, 0], d[:, 1], color=TEAL, width=0.62,
           edgecolor="#1F262B", linewidth=0.6)
    for x, y in zip(d[:, 0], d[:, 1]):
        ax.text(x, y + 0.0004, f"{y:.4f}", ha="center", fontsize=8.0,
                fontweight="bold", color=INK)
    ax.set_xticks(d[:, 0])
    ax.set_xlabel("GDN-2 layer index", fontsize=8.5)
    ax.set_ylabel("mean decay", fontsize=8.5)
    ax.set_ylim(0, 0.0285)
    ax.tick_params(labelsize=8, length=3, width=0.7)
    for s in ax.spines.values():
        s.set_linewidth(0.7)
    return ax


def panel_complexity(ax):
    L = np.linspace(1, 100, 200)
    ax.plot(L, L, color=TEAL, lw=1.5, label="linear attention · O(L)")
    ax.plot(L, L ** 2 / 60, color=RED, lw=1.3, ls="--",
            label="self-attention · O(L$^2$)")
    ax.set_xlim(-14, 108); ax.set_ylim(-14, 190)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.annotate("", xy=(104, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
    ax.annotate("", xy=(0, 182), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
    ax.text(52, -11, "window length $L$", fontsize=7.5, color=INK,
            ha="center", fontstyle="italic", clip_on=False)
    ax.text(-12, 92, "compute", fontsize=7.5, color=INK, va="center",
            rotation=90, fontstyle="italic", clip_on=False)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right",
              bbox_to_anchor=(1.0, 0.05))
    return ax


def panel_patch(ax):
    for P, c in [("2", TEAL), ("4", BLUE_MID), ("8", TERRA)]:
        d = np.loadtxt(D / f"patch{P}.dat")
        ax.plot(np.arange(len(d)), d[:, 1], color=c, lw=1.1,
                label=f"patch {P}")
    ax.set_xlabel("patched index", fontsize=8.5)
    ax.set_ylabel("norm. $Q$", fontsize=8.5)
    ax.tick_params(labelsize=8, length=2, width=0.5)
    for s in ax.spines.values():
        s.set_linewidth(0.5)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    return ax


def one(name, panel, figsize=(3.2, 3.0)):
    """One standalone panel figure; labeling is done by LaTeX (subcaption)."""
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    panel(ax)
    fig.tight_layout(pad=0.55)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches=None,
                facecolor="white", edgecolor="none")
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches=None,
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"ok {name}")


def main():
    one("fig_evidence_decay", panel_decay)
    one("fig_evidence_complexity", panel_complexity)
    one("fig_evidence_patch", panel_patch)


if __name__ == "__main__":
    main()
