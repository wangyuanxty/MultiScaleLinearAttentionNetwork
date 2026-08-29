#!/usr/bin/env python3
"""Real-data SVGs for fig_arch_main: window+pooling, patch decomposition,
concept complexity curve (no numeric axis)."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent
D = ROOT / "data"
OUT = ROOT / "charts"
OUT.mkdir(exist_ok=True)

INK = "#1F262B"
MUTED = "#707A82"
RED = "#D64533"
TEAL = "#2E7D8C"
AMBER = "#B08A2E"
BLUEL = "#A8BDD1"


def window_pooling():
    """Real 64-cycle window + rich patch bands (regen annotation, full bands)."""
    top = np.loadtxt(D / "capacity.dat")
    w = top[:64]
    fig = plt.figure(figsize=(3.2, 11.0), facecolor="none")

    # top: real 64-cycle window
    a1 = fig.add_axes([0.075, 0.80, 0.88, 0.19])
    a1.plot(np.arange(64), w[:, 1], color=TEAL, lw=1.8)
    a1.fill_between(np.arange(64), 0.900, w[:, 1], color=TEAL, alpha=0.10, lw=0)
    a1.set_xlim(0, 63); a1.set_ylim(0.895, 1.000)
    a1.tick_params(labelsize=10, length=3, width=0.7)
    for s_ in a1.spines.values():
        s_.set_linewidth(0.6)
    a1.set_ylabel("norm. Q", fontsize=11, labelpad=2)
    a1.set_xlabel("cycles before SP", fontsize=11, labelpad=2)
    a1.tick_params(axis="x", labelbottom=False)
    # regeneration bump: local max before the declining tail (real data)
    peak_i = int(np.argmax(w[20:60, 1])) + 20
    a1.plot([peak_i], [w[peak_i, 1]], marker="o", ms=4, color="#B08A2E", zorder=6)
    a1.annotate("regeneration", xy=(peak_i, w[peak_i, 1]),
                xytext=(peak_i - 34, w[peak_i, 1] + 0.012),
                fontsize=12, color="#B08A2E", fontstyle="italic",
                arrowprops=dict(arrowstyle="->", color="#B08A2E", lw=0.9))
    a1.plot([63], [w[-1, 1]], marker="v", ms=6, color="#D64533", zorder=7)
    a1.text(59.5, w[-1, 1] + 0.010, "SP", fontsize=10.5, color="#D64533",
            ha="right", fontweight="bold")


    # bottom: three full-height patch bands + alignment guides
    a2 = fig.add_axes([0.0, 0.0, 1.0, 0.74])
    a2.set_xlim(0, 64); a2.set_ylim(-0.35, 5.55); a2.axis("off")
    bands = [(2, "#2E7D8C", 4.75, "patch 2 → 32 tokens · stride 2"),
             (4, "#7A9BB5", 2.60, "patch 4 → 16 tokens · stride 4"),
             (8, "#D98B6A", 0.45, "patch 8 → 8 tokens · stride 8")]
    for ps, col, ylab, lab in bands:
        n = 64 // ps
        for i in range(n):
            x0 = i * ps
            a2.add_patch(plt.Rectangle((x0 + 0.10, ylab - 0.46), ps - 0.20, 0.92,
                                       facecolor=col, edgecolor="none", alpha=0.85))
        a2.text(66.0, ylab, lab, fontsize=12, color=INK, va="center")
        a2.text(ps / 2, ylab, f"0–{ps - 1}", fontsize=9, color="white",
                ha="center", va="center", fontweight="bold")
    for c in range(8, 64, 8):
        a2.plot([c, c], [-0.25, 4.65], color="#C9D1D8", lw=0.6, ls=":")
    # real pooled series overlaid on each band (true values, normalized to band)
    for ps, col, ylab in [(2, "#0F4457", 4.75), (4, "#4F7590", 2.60), (8, "#A5543C", 0.45)]:
        d = np.loadtxt(D / f"patch{ps}.dat")[:, 1]
        lo, hi = d.min(), d.max()
        xs = [(i + 0.5) * ps for i in range(len(d))]
        ys = [ylab - 0.46 + (v - lo) / (hi - lo) * 0.92 for v in d]
        a2.plot(xs, ys, color=col, lw=1.0, alpha=0.95)
        a2.plot(xs, ys, marker="o", ms=2.6, color="#FFFFFF", mec=col,
                mew=0.7, ls="none", zorder=5)
    for ylab in (3.675, 1.525):
        a2.text(63.0, ylab, "pool", fontsize=10, color=MUTED,
                fontstyle="italic", va="center", ha="right")

    fig.savefig(OUT / "arch_window_pool.svg", format="svg", bbox_inches="tight",
                facecolor="none", edgecolor="none")
    plt.close(fig)
    print("ok arch_window_pool.svg")


def x_label(i):
    """Format cycle label compactly."""
    return f"{i}"


def patch_decomp():
    fig, ax = plt.subplots(figsize=(4.6, 2.6), facecolor="none")
    for P, c in [("2", "#2E7D8C"), ("4", "#6B8FB0"), ("8", "#C96A4A")]:
        d = np.loadtxt(D / f"patch{P}.dat")
        ax.plot(np.arange(len(d)), d[:, 1], color=c, lw=1.1, label=f"patch {P}")
    ax.set_xlabel("patched index", fontsize=7)
    ax.set_ylabel("norm. Q", fontsize=7)
    ax.tick_params(labelsize=6, length=2, width=0.5)
    for s in ax.spines.values():
        s.set_linewidth(0.5)
    ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    ax.set_title("real multi-scale patch series (W=64 window)", fontsize=7,
                 color=MUTED, pad=4, fontstyle="italic")
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "arch_patch_decomp.svg", format="svg", bbox_inches="tight",
                facecolor="none", edgecolor="none")
    plt.close(fig)
    print("ok arch_patch_decomp.svg")


def complexity():
    """Concept: O(L) vs O(L^2), qualitative, axes without tick numbers."""
    fig, ax = plt.subplots(figsize=(4.6, 2.6), facecolor="none")
    L = np.linspace(1, 100, 200)
    ax.plot(L, L, color=TEAL, lw=1.5, label="linear attention · O(L)")
    ax.plot(L, L ** 2 / 60, color=RED, lw=1.3, ls="--", label="self-attention · O(L²)")
    ax.set_xlim(-14, 108); ax.set_ylim(-14, 190)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.annotate("", xy=(104, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
    ax.annotate("", xy=(0, 182), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
    ax.text(52, -14, "window length L", fontsize=6.5, color=INK, ha="center",
            fontstyle="italic", clip_on=False)
    ax.text(-10, 92, "compute", fontsize=6.5, color=INK, va="center",
            rotation=90, fontstyle="italic", clip_on=False)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "arch_complexity.svg", format="svg", bbox_inches="tight",
                facecolor="none", edgecolor="none")
    plt.close(fig)
    print("ok arch_complexity.svg")


if __name__ == "__main__":
    window_pooling()
    patch_decomp()
    complexity()
