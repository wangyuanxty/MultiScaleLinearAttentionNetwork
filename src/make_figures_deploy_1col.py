"""Single-column renders of the deployment figures.

fig_dep_memctx and fig_dep_flash are drawn full-width (6.9 and 6.8 in) in
make_figures_deploy.py, so as full-width floats they eat a lot of page.
This script re-renders them on a 3.45 in canvas -- the column width of the
elsarticle 5p two-column layout -- so they can be placed as ordinary
single-column floats.  Font sizes are unchanged, so text renders at its
nominal size instead of being scaled down.

make_figures_deploy.py is left untouched.

Usage: python src/make_figures_deploy_1col.py
Writes: paper/figures/fig_dep_memctx.pdf/.png, fig_dep_flash.pdf/.png
"""
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

FIG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "paper", "figures"
)
os.makedirs(FIG, exist_ok=True)

COL_WIDTH = 3.45            # elsarticle 5p column width, inches
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
GREY = "#b0b0b0"
EDGE = "#2b2b2b"

plt.rcParams.update(
    {
        "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
        "legend.fontsize": 6.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "figure.dpi": 300, "savefig.dpi": 300,
        "font.family": "serif", "mathtext.fontset": "dejavuserif",
        "axes.grid": True, "grid.color": "#d9d9d9", "grid.linewidth": 0.4,
        "axes.spines.top": False, "axes.spines.right": False,
        # matplotlib 3.11: a 'tight' bbox plus mathtext yields a broken
        # portrait canvas, so this must stay None.
        "savefig.bbox": None,
    }
)

# --- constants, copied from make_figures_deploy.py -------------------------
ATTN_H = 4
MODEL_D = 64
STATE_MULTI_KB = 48.0
STATE_PER_LAYER_KB = 8.0
ATTN_LAYERS = int(round(STATE_MULTI_KB / STATE_PER_LAYER_KB))     # 6

TOKENS = np.array([32, 64, 128, 256, 512, 1024])
ATTN_KB = ATTN_H * TOKENS.astype(float) ** 2 * 4.0 / 1024.0
KV_KB = ATTN_LAYERS * 2.0 * TOKENS * MODEL_D * 4.0 / 1024.0
CROSSOVER_L = float(np.sqrt(STATE_MULTI_KB * 1024.0 / (ATTN_H * 4.0)))
CROSSOVER_KV = STATE_MULTI_KB * 1024.0 / (ATTN_LAYERS * 2.0 * MODEL_D * 4.0)

W_MULTI = {"fp32": 1.85 * 1024.0, "int8": 504.0, "int4": 268.0}
W_SINGLE = {"fp32": 444.0, "int8": 122.0, "int4": 67.0}


def _kb_fmt(v: float) -> str:
    return f"{v / 1024.0:g} MB" if v >= 1024.0 else f"{v:g} KB"


def _save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=300)
    plt.close(fig)
    print(f"{name}.pdf/png done (single column)")


def fig_dep_memctx_1col():
    """Same three curves, redesigned for a 3.45 in canvas.

    The full-width version carries a three-line annotation box and a
    three-entry legend; at column width they collide, so the annotation
    is reduced to two short lines placed bottom-left and the legend moves
    under the axis.
    """
    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.75))

    ax.plot(TOKENS, ATTN_KB, "o-", color=COLORS[3], lw=1.2, ms=3.4,
            label="attention scores ($HL^2{\\times}4$ B)")
    ax.plot(TOKENS, KV_KB, "s-", color=COLORS[1], lw=1.2, ms=3.2,
            label=f"KV cache ($2Ld_{{\\mathrm{{model}}}}{{\\times}}4$ B $\\times${ATTN_LAYERS})")
    ax.plot(TOKENS, np.full_like(TOKENS, STATE_MULTI_KB, dtype=float), "^--",
            color=COLORS[0], lw=1.3, ms=3.8,
            label=f"ours ({ATTN_LAYERS}-layer state)")

    for xc, lab, col in ((CROSSOVER_KV, "KV", COLORS[1]),
                         (CROSSOVER_L, "scores", COLORS[3])):
        ax.axvline(xc, color="0.45", ls=":", lw=0.9, zorder=0)
        ax.plot([xc], [STATE_MULTI_KB], "o", ms=4.4, mfc="white",
                mec="0.2", mew=1.0, zorder=5)
        ax.text(xc * 1.12, 2.2e4, f"{lab} $L{{=}}{xc:.0f}$", fontsize=6.0,
                color=col, ha="left", va="top")

    ax.axvline(32.0, color="0.55", ls="--", lw=0.8, zorder=0)
    for yv, col, mk in ((ATTN_KB[0], COLORS[3], "o"), (KV_KB[0], COLORS[1], "s")):
        ax.plot([32.0], [yv], mk, ms=5.0, mfc="white", mec=col, mew=1.3, zorder=6)

    ax.annotate(
        f"32 tokens: KV cache {KV_KB[0]:.0f} KB,\ntwice our whole state",
        xy=(32.0, KV_KB[0]), xytext=(48.0, 5.0),
        fontsize=6.0, color="0.12", ha="left", va="bottom",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.65", lw=0.5),
        arrowprops=dict(arrowstyle="->", lw=0.7, color="0.35",
                        shrinkA=1, shrinkB=2),
    )

    ax.text(1250.0, STATE_MULTI_KB * 1.18, "48 KB, flat in $L$",
            fontsize=6.8, color=COLORS[0], ha="right", va="bottom")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(TOKENS)
    ax.set_xticklabels([str(t) for t in TOKENS])
    ax.set_yticks([16, 64, 256, 1024, 4096, 16384])
    ax.set_yticklabels([_kb_fmt(v) for v in [16, 64, 256, 1024, 4096, 16384]])
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlim(13.0, 2600.0)
    ax.set_ylim(4.2, 1.4e5)

    ax.set_xlabel("context length $L$ (tokens)")
    ax.set_ylabel("working memory (KB)")
    leg = ax.legend(frameon=False, loc="upper left", fontsize=6.0,
                    handlelength=1.8, borderaxespad=0.3, labelspacing=0.25)
    leg.set_zorder(10)

    fig.tight_layout(pad=0.4)
    _save(fig, "fig_dep_memctx")


def fig_dep_flash_1col():
    """Same footprint bars on a column-width canvas."""
    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.5))

    rows = [
        ("multi-scale fp32", W_MULTI["fp32"], True),
        ("multi-scale INT8", W_MULTI["int8"], True),
        ("multi-scale INT4", W_MULTI["int4"], True),
        ("single-branch fp32", W_SINGLE["fp32"], False),
        ("single-branch INT8", W_SINGLE["int8"], False),
        ("single-branch INT4", W_SINGLE["int4"], False),
    ]
    y = np.arange(len(rows))
    vals = [r[1] for r in rows]
    cols = [COLORS[0] if r[2] else GREY for r in rows]

    X_MIN = 52.0
    ax.set_xscale("log", base=2)
    bars = ax.barh(y, [v - X_MIN for v in vals], left=X_MIN, color=cols,
                   height=0.60, edgecolor=EDGE, linewidth=0.5)
    for yy, v in zip(y, vals):
        ax.text(v * 1.08, yy, _kb_fmt(v), va="center", ha="left", fontsize=6.5)

    # The dashed tiers sit at 128/256/512/1024/2048 -- the same values as the
    # x ticks -- so labelling both duplicates every number and the two rows
    # collide at column width.  The caption already states that the dashed
    # lines are common MCU flash sizes, so only the axis carries numbers.
    for tier in (128.0, 256.0, 512.0, 1024.0, 2048.0):
        ax.axvline(tier, color="0.35", ls="--", lw=0.8, zorder=0)

    ax.set_xlim(X_MIN, 4200.0)
    ax.set_xticks([64, 128, 256, 512, 1024, 2048])
    ax.set_xticklabels([_kb_fmt(v) for v in [64, 128, 256, 512, 1024, 2048]],
                       fontsize=6.0)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.5)
    ax.set_ylim(-0.75, len(rows) - 0.3)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("weight footprint (KB)")

    handles = [
        Patch(fc=COLORS[0], ec=EDGE, lw=0.5, label="multi-scale (paper model)"),
        Patch(fc=GREY, ec=EDGE, lw=0.5, label="single-branch encoder"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=6.0)

    fig.tight_layout(pad=0.4)
    _save(fig, "fig_dep_flash")


if __name__ == "__main__":
    fig_dep_memctx_1col()
    fig_dep_flash_1col()
