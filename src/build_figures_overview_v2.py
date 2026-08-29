"""Fig.1 system overview — design-system pass (v3, frontend-design principles).

Design tokens: paper-white stage, graphite ink, semantic accents only:
  DATA=graphite-teal  PHYSICS=coral  UNCERTAINTY=steel blue  DEPLOYMENT=sand
Signature: the five real dataset curves lane. Numbers appear only as data tiles.

Usage: python src/build_figures_overview_v2.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, BoxStyle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "paper", "figures_gen", "data")
OUT = os.path.join(ROOT, "paper", "figures")

INK = "#1E2A32"          # graphite ink
PAPER = "#FFFFFF"
LANE = "#F4F6F7"         # lane gray stage
HARDLINE = "#D8DEE2"
DATA = "#2E7D8C"         # data / model curves
PHYSICS = "#C96A4A"      # physics accent (EOL, rate head)
UNCERT = "#6B8FB0"       # uncertainty / CQR
DEPLOY = "#D9BE8C"       # deployment / hardware

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.linewidth": 0.6,
    "mathtext.fontset": "cm", "savefig.bbox": None,
    "figure.facecolor": PAPER,
})

DS = [("calce", "CALCE", "SP 300/400/500"),
      ("nasa", "NASA", "SP 50/70/90"),
      ("mit", "MIT", "SP 200/300/400"),
      ("panasonic", "PANASONIC", "SP 300/500/700"),
      ("tju", "TJU", "SP 200/300/400")]


def box(ax, x, y, w, h, text=None, fc=PAPER, ec=HARDLINE, lw=0.9, fs=9,
        bold=False, tc=INK, r=0.008, ha="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, mutation_aspect=0.5,
                                transform=ax.transAxes))
    if text:
        ax.text(x + (w / 2 if ha == "center" else 0.02), y + h / 2, text,
                ha=ha, va="center", fontsize=fs, color=tc,
                fontweight="bold" if bold else "normal",
                transform=ax.transAxes)


def arrow(ax, x1, y1, x2, y2, lw=0.9):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", lw=lw,
                                 color=INK, mutation_scale=8,
                                 transform=ax.transAxes))


def eyebrow(ax, x, y, text, color=INK):
    ax.text(x, y, text.upper(), fontsize=8.5, fontweight="bold", color=color,
            transform=ax.transAxes)


def dot(ax, x, y, color, r=0.006):
    ax.add_patch(plt.Circle((x, y), r, color=color, transform=ax.transAxes,
                            clip_on=False))


def photo(ax, x, y, w, h, name):
    path = os.path.join(ROOT, "paper", "figures_gen", "photos", name + ".png")
    if not os.path.exists(path):
        return False
    img = plt.imread(path)
    clip = FancyBboxPatch((x, y), w, h,
                          boxstyle=BoxStyle.Round(pad=0, rounding_size=0.012),
                          transform=ax.transAxes)
    im = ax.imshow(img, extent=(x, x + w, y, y + h), transform=ax.transAxes,
                   aspect="auto", zorder=1)
    im.set_clip_path(clip)
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=BoxStyle.Round(pad=0, rounding_size=0.012),
                                fc="none", ec=HARDLINE, lw=0.9,
                                transform=ax.transAxes, zorder=3))
    return True


def main():
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # ---- masthead: ink title + hairline rule ----
    ax.text(0.025, 0.95, "DeltaCycle", fontsize=19, fontweight="bold",
            color=INK, transform=ax.transAxes)
    ax.text(0.025, 0.915,
            "Capacity-only monitoring: five datasets, one linear-attention loop, "
            "calibrated decisions on an edge MCU",
            fontsize=9.5, color=INK, transform=ax.transAxes)
    ax.plot([0.025, 0.975], [0.895, 0.895], color=INK, lw=0.6,
            transform=ax.transAxes)

    # ================= LANE 1 (signature): five real datasets =================
    eyebrow(ax, 0.025, 0.865, "Five real degradation curves")
    ax.text(0.255, 0.865, "normalized capacity · EOL line (dashed) · starting points (▼)",
            fontsize=7.5, color=INK, transform=ax.transAxes)
    box(ax, 0.02, 0.575, 0.96, 0.27, None, fc=LANE, ec="none", r=0.012)
    for i, (key, name, spl) in enumerate(DS):
        x0 = 0.035 + i * 0.192
        a = fig.add_axes([x0, 0.615, 0.172, 0.175])
        dat = np.loadtxt(f"{DATA_DIR}/ds_{key}.dat")
        th = float(np.loadtxt(f"{DATA_DIR}/ds_{key}_eol.txt"))
        a.plot(dat[:, 0], dat[:, 1], color=DATA, lw=0.9)
        a.axhline(th, color=PHYSICS, ls="--", lw=0.8)
        sps = [300, 400, 500] if key == "calce" else (
            [50, 70, 90] if key == "nasa" else
            [200, 300, 400] if key == "mit" else
            [300, 500, 700] if key == "panasonic" else [200, 300, 400])
        for sp in sps:
            idx = int(np.searchsorted(dat[:, 0], sp))
            if idx < len(dat):
                a.plot(dat[idx, 0], dat[idx, 1], "v", ms=2.5, color=INK)
        a.set_ylim(0, 1.08)
        a.tick_params(labelsize=5.5, length=2, width=0.5)
        for s_ in a.spines.values():
            s_.set_linewidth(0.5)
        ax.text(x0 + 0.086, 0.594, name, fontsize=7.2, fontweight="bold",
                ha="center", transform=ax.transAxes)
        ax.text(x0 + 0.086, 0.584, spl, fontsize=6.2, ha="center",
                transform=ax.transAxes)

    # ================= LANE 2: model chain =================
    eyebrow(ax, 0.025, 0.520, "Model chain")
    chain = [("Capacity window", "W = 64, global min-max\n→ per-window z-score", DATA),
             ("Patch 2/4/8", "three branches,\nshared GDN-2 layers", DATA),
             ("GDN-2 state", "H × Dk × Dv = 8 KB,\nfixed per layer", DATA),
             ("StageQuery exchange", "coarse state as\nsingle query", PHYSICS),
             ("Rate head", "r = softplus(w·h)\n+ softplus(γ)·IR", PHYSICS),
             ("CQR outputs", "P2.5 / P50 / P97.5\n95% calibrated", UNCERT)]
    bw, bh, bx0 = 0.152, 0.145, 0.025
    for i, (title, body, acc) in enumerate(chain):
        x = bx0 + i * 0.163
        box(ax, x, 0.355, bw, bh, None, fc=PAPER, ec=HARDLINE, lw=0.9)
        dot(ax, x + 0.022, 0.483, acc)
        ax.text(x + 0.022, 0.462, title, fontsize=8.8, fontweight="bold",
                color=INK, transform=ax.transAxes)
        ax.text(x + 0.022, 0.410, body, fontsize=6.9, color=INK,
                va="top", transform=ax.transAxes)
        if i < len(chain) - 1:
            arrow(ax, x + bw, 0.4275, x + 0.166, 0.4275)
    ax.text(0.025, 0.338,
            "single-query linear attention O(L)  ·  per-layer gated cross-scale "
            "exchange  ·  last-token readout  ·  8 KB state per layer",
            fontsize=7.2, color=INK, transform=ax.transAxes)

    # ================= LANE 3: decisions + deployment =================
    eyebrow(ax, 0.025, 0.290, "Operational decisions & edge deployment")
    cards = [("Maintenance scheduling", "risk-aware, with\n95% intervals", UNCERT),
             ("Replacement planning", "RUL interval\nfrom the crossing", UNCERT),
             ("Risk tiering", "P2.5 / P97.5\nprobability bands", UNCERT)]
    for j, (title, body, acc) in enumerate(cards):
        x = 0.025 + j * 0.163
        box(ax, x, 0.115, 0.152, 0.115, None, fc=PAPER, ec=HARDLINE, lw=0.9)
        dot(ax, x + 0.022, 0.213, acc)
        ax.text(x + 0.022, 0.194, title, fontsize=8.4, fontweight="bold",
                color=INK, transform=ax.transAxes)
        ax.text(x + 0.022, 0.152, body, fontsize=6.9, color=INK,
                va="center", transform=ax.transAxes)
    # deployment cluster
    box(ax, 0.525, 0.115, 0.235, 0.115, None, fc=LANE, ec=HARDLINE, lw=0.9)
    ax.text(0.533, 0.205, "EDGE MCU", fontsize=8.6, fontweight="bold",
            color=INK, transform=ax.transAxes)
    kv = [("Recurrent state", "fixed 8 KB"), ("Weights", "INT8, 340 KB (−75%)"),
          ("Inference", "8–12 ms @ 168 MHz"), ("Fidelity", "bit-exact, 2.5e-6")]
    for j, (k, v) in enumerate(kv):
        yy = 0.181 - j * 0.024
        ax.text(0.533, yy, k, fontsize=6.8, color=INK, transform=ax.transAxes)
        ax.text(0.66, yy, v, fontsize=6.8, fontweight="bold", color=INK,
                transform=ax.transAxes)
    photo(ax, 0.775, 0.115, 0.10, 0.115, "mcu_board")

    # ================= data tiles =================
    tiles = [("MAE", "0.0049"), ("R²", "0.9945"), ("AE", "1.2 cycles"),
             ("Coverage", "0.933"), ("Scale", "5 datasets · 12 SPs · 10 seeds")]
    for j, (lab, val) in enumerate(tiles):
        x = 0.02 + j * 0.192
        ax.text(x, 0.062, lab.upper(), fontsize=6.5, color=INK,
                transform=ax.transAxes)
        ax.text(x, 0.028, val, fontsize=13, fontweight="bold", color=INK,
                transform=ax.transAxes)
    ax.plot([0.02, 0.975], [0.085, 0.085], color=HARDLINE, lw=0.6,
            transform=ax.transAxes)

    fig.savefig(os.path.join(OUT, "fig_overview.png"), dpi=200)
    fig.savefig(os.path.join(OUT, "fig_overview.pdf"))
    plt.close(fig)
    print("saved fig_overview (v3, design-system pass)")


if __name__ == "__main__":
    main()
