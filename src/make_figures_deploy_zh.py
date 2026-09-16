"""Chinese (中文) versions of the edge-deployment figures -> paper/figures_zh/.

Regenerated rather than patched: every number in `make_figures_deploy.py` comes
from hardcoded constants sourced from paper/sections/05_deployment.tex
(tab:deploy), so no checkpoint or dataset is involved and the Chinese version
can be a real vector figure instead of a PDF with substituted glyphs.

Fonts: `font.serif = [DejaVu Serif, SimHei]`.  matplotlib 3.6+ walks the list
per glyph, so Latin letters and digits keep the paper's DejaVu Serif while CJK
falls through to SimHei -- the same pairing used by figures_zh/fig_traj.pdf.

Usage: python src/make_figures_deploy_zh.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Reuse the paper's own constants and helpers so the two versions cannot drift.
import make_figures_deploy as D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_ZH = os.path.join(ROOT, "paper", "figures_zh")
os.makedirs(FIG_ZH, exist_ok=True)

plt.rcParams.update(D.plt.rcParams)  # same look as the English figures
plt.rcParams.update(
    {
        # A *list* on font.family is what enables matplotlib's per-glyph font
        # fallback. Putting the pair in font.serif instead does not: matplotlib
        # resolves that to the first entry only and silently drops CJK glyphs.
        "font.family": ["DejaVu Serif", "SimHei"],
        "axes.unicode_minus": False,  # SimHei has no U+2212
    }
)


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_ZH, f"{name}.{ext}"), dpi=300)
    plt.close(fig)
    print(f"figures_zh/{name}.pdf/png done")


# ---------------------------------------------------------------------------
# C -- which precision fits which flash tier
# ---------------------------------------------------------------------------
def fig_dep_flash_zh():
    fig, ax = plt.subplots(figsize=(6.8, 3.5))

    # (label, KB, is_multi) -- ascending by size, same order as the English one
    rows = [
        ("单分支 INT4", D.W_SINGLE["int4"], False),
        ("单分支 INT8", D.W_SINGLE["int8"], False),
        ("多尺度 INT4", D.W_MULTI["int4"], True),
        ("单分支 fp32", D.W_SINGLE["fp32"], False),
        ("多尺度 INT8", D.W_MULTI["int8"], True),
        ("多尺度 fp32", D.W_MULTI["fp32"], True),
    ]
    y = np.arange(len(rows))
    vals = [r[1] for r in rows]
    cols = [D.COLORS[0] if r[2] else D.GREY for r in rows]

    # Bars are anchored at the axis minimum rather than 0 (see the English
    # figure): on a log axis a zero left edge is clipped to the axis minimum.
    X_MIN = 52.0
    ax.set_xscale("log", base=2)
    # Deliberately opaque: alpha writes an ExtGState into the PDF that some
    # viewers drop, rendering the figure as bare labels.
    D._edged(ax.barh(y, [v - X_MIN for v in vals], left=X_MIN, color=cols,
                     height=0.62))
    for yy, v in zip(y, vals):
        ax.text(v * 1.07, yy, D._kb_fmt(v), va="center", ha="left", fontsize=7.2)

    for tier in (128.0, 256.0, 512.0, 1024.0, 2048.0):
        ax.axvline(tier, color="0.35", ls="--", lw=0.9, zorder=0)
        ax.text(tier, -0.50, D._kb_fmt(tier), fontsize=6.8, color="0.3",
                ha="center", va="bottom")

    ax.set_xlim(X_MIN, 3600.0)
    ax.set_xticks([64, 128, 256, 512, 1024, 2048])
    ax.set_xticklabels([D._kb_fmt(v) for v in [64, 128, 256, 512, 1024, 2048]])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
    ax.set_ylim(-0.62, len(rows) - 0.35)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("权重占用 (KB)")

    handles = [
        Patch(fc=D.COLORS[0], ec=D.EDGE, lw=0.6, label="多尺度（本文模型）"),
        Patch(fc=D.GREY, ec=D.EDGE, lw=0.6, label="单分支编码器"),
    ]
    ax.legend(handles=handles, frameon=False, loc="center right", fontsize=7.0)
    ax.text(0.0, -0.27, "虚线：常见 MCU Flash 容量",
            transform=ax.transAxes, fontsize=6.8, color="0.35",
            style="italic", ha="left", va="top")

    fig.tight_layout()
    _save(fig, "fig_dep_flash")


if __name__ == "__main__":
    fig_dep_flash_zh()
