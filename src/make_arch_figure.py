"""Hand-drawn vector architecture figure (fig_arch.pdf) for the paper.

Replaces the gpt-image-2 version: precise labels, journal style,
no deployment-related content. Layout (left to right):
  input [C] -> 3 branches (patch 2/4/8), each: patch embed + 2x GDN-2
  per-layer gated cross-scale exchange between branches
  -> concat last token -> readout -> K=1 SOH / K=32 trajectory
  physics regularizer (training only) shown as dashed box below.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper", "figures")
os.makedirs(FIG, exist_ok=True)

BLUE = "#1f77b4"
GREEN = "#2ca02c"
RED = "#d62728"
GRAY = "#555555"
LGRAY = "#f2f4f7"

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "xtick.bottom": False,
        "ytick.left": False,
    }
)

fig, ax = plt.subplots(figsize=(12.0, 6.2))
ax.set_xlim(0, 12)
ax.set_ylim(0, 6.2)
ax.axis("off")


def box(x, y, w, h, text, fc="white", ec=BLUE, fs=8, lw=1.1, tc="black",
        ls="-", bold=False, z=2):
    """Draw a rounded box with centered text; returns center x, y."""
    b = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
        fc=fc, ec=ec, lw=lw, linestyle=ls, zorder=z,
    )
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, zorder=z + 1,
            fontweight="bold" if bold else "normal")
    return x + w / 2, y + h / 2


def arrow(x1, y1, x2, y2, color=GRAY, lw=1.2, style="-|>", ls="-", z=1, ms=11):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=ms,
        color=color, lw=lw, linestyle=ls, zorder=z,
        shrinkA=0, shrinkB=0,
    )
    ax.add_patch(a)


# ---------------------------------------------------------------- input
box(0.15, 2.55, 1.05, 0.9,
    "Capacity\nsequence $[C]$\nwindow $W{=}64$", ec=GRAY, fs=8.5, bold=True)
arrow(1.25, 3.0, 2.0, 4.95, lw=1.1)   # to fine
arrow(1.25, 3.0, 2.0, 3.15, lw=1.1)  # to mid
arrow(1.25, 3.0, 2.0, 1.35, lw=1.1)  # to coarse

# ---------------------------------------------------------------- branches
branch_y = {2: 4.35, 4: 2.55, 8: 0.75}   # bottom y of each branch lane
branch_lab = {2: "fine branch  (patch 2)", 4: "mid branch  (patch 4)",
              8: "coarse branch  (patch 8)"}

for ps, y0 in branch_y.items():
    # patch embed
    box(2.0, y0 + 0.62, 1.15, 0.5, f"patch {ps}\nembed", fs=7.5, ec=BLUE)
    # two GDN-2 blocks
    for i, (bx, bl) in enumerate([(3.35, "GDN-2\nblock"), (4.85, "GDN-2\nblock")]):
        box(bx, y0 + 0.42, 1.15, 0.9, bl, fs=8, bold=True, fc="#eef4fb")
        arrow(bx + 1.15, y0 + 0.87, bx + 1.52, y0 + 0.87, lw=1.0)
    arrow(2.0 + 1.15, y0 + 0.87, 3.35, y0 + 0.87, lw=1.0)
    # branch label
    ax.text(2.0, y0 + 1.28, branch_lab[ps], fontsize=7.5, color=GRAY,
            style="italic")

# GDN-2 detail callout (small box under coarse branch)
box(3.35, 0.02, 2.65, 0.42,
    "q,k,v proj $\\to$ conv1d $\\to$ $S{=}(I{-}k(b{\\odot}k)^{\\top})$diag$(\\alpha)S{+}k(w{\\odot}v)^{\\top}$, $o{=}S^{\\top}q$",
    fs=6.2, ec=GRAY, fc=LGRAY)
ax.annotate("", xy=(4.4, 0.44), xytext=(4.4, 0.75),
            arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.8))

# ------------------------------------------------- cross-scale exchange
for gx, gy in [(3.7, 3.9), (3.7, 2.1), (5.2, 3.9), (5.2, 2.1)]:
    # fine<->mid at layer1/2 ; mid<->coarse at layer1/2
    ax.add_patch(plt.Circle((gx, gy), 0.10, fc=GREEN, ec="none", zorder=3))
    ax.text(gx, gy, "$\\sigma$", ha="center", va="center", fontsize=7,
            color="white", zorder=4, fontweight="bold")
arrow(3.7, 4.35 + 0.42 + 0.1, 3.7, 3.9 + 0.1 + 0.05, color=GREEN, lw=0.9)
arrow(3.7, 3.9 - 0.1 - 0.05, 3.7, 2.55 + 0.1, color=GREEN, lw=0.9)
arrow(5.2, 4.35 + 0.1, 5.2, 3.9 + 0.1 + 0.05, color=GREEN, lw=0.9)
arrow(5.2, 3.9 - 0.1 - 0.05, 5.2, 2.55 + 0.1, color=GREEN, lw=0.9)
ax.text(6.0, 3.75, "per-layer gated cross-scale exchange", fontsize=7.5,
        color=GREEN, style="italic", rotation=90, va="center")

# ------------------------------------------------- fusion + readout
box(6.55, 2.55, 1.0, 0.9, "concat\nlast token", fs=8, bold=True, ec=BLUE)
for bx, y0 in [(4.85, 4.35), (4.85, 2.55), (4.85, 0.75)]:
    arrow(bx + 1.15, y0 + 0.87, 6.55, 3.0, lw=1.0)

box(7.75, 2.55, 1.35, 0.9, "readout\nRMSNorm $\\to$ Linear(128)\n$\\to$ GELU $\\to$ Linear($K$)",
    fs=7.5, bold=True, ec=BLUE)
arrow(7.55, 3.0, 7.75, 3.0, lw=1.2)

# outputs
box(9.35, 3.2, 1.5, 0.62, "SOH estimate\n$K{=}1$ (single step)", fs=7.5,
    ec=GREEN)
box(9.35, 2.0, 1.5, 0.62, "K=32 trajectory\n31-cycle lookahead", fs=7.5,
    ec=GREEN)
arrow(9.1, 3.3, 9.35, 3.42, lw=1.1)
arrow(9.1, 2.7, 9.35, 2.36, lw=1.1)

# ------------------------------------------------- physics regularizer
phy = FancyBboxPatch((7.75, 0.02), 3.1, 1.0, boxstyle="round,pad=0.04,rounding_size=0.08",
                     fc="#fdf3f4", ec=RED, lw=1.1, ls="--", zorder=2)
ax.add_patch(phy)
box(7.9, 0.28, 1.0, 0.5, "IR, T\n(training only)", fs=7, ec=RED, fc="#fdf3f4")
box(9.05, 0.28, 1.65, 0.5, "Arrhenius decay\n$r_{\\mathrm{phys}}{=}\\beta{+}\\gamma_{\\mathrm{ir}}{\\cdot}$IR${+}\\gamma_t{\\cdot}e^{{-}E_a/RT}$",
    fs=6.4, ec=RED, fc="#fdf3f4")
arrow(8.9, 0.53, 9.05, 0.53, color=RED, lw=1.0)
# dashed link from physics box up to the loss
arrow(9.88, 1.02, 9.88, 1.9, color=RED, lw=0.9, ls="--")
ax.text(9.95, 1.45, "$\\mathcal{L}{=}\\mathcal{L}_{\\mathrm{MAE}}{+}\\lambda{\\cdot}|r_{\\mathrm{pred}}{-}r_{\\mathrm{phys}}|$",
        fontsize=6.8, color=RED, rotation=90, va="center")
ax.text(7.75, 1.14, "physics-consistent regularization (training objective only)",
        fontsize=7, color=RED, style="italic")

fig.tight_layout(pad=0.2)
fig.savefig(os.path.join(FIG, "fig_arch.pdf"))
print("fig_arch.pdf done")
