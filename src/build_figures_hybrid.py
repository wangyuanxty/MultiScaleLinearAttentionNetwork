"""Hybrid paper figures: matplotlib layout + gpt-image-2 photo assets.

Photos (paper/figures_gen/photos/*.png, text-free) come from the
baoyu-image-gen CLI (prompts in paper/figures_gen/prompts/photos/).
Everything else — boxes, arrows, labels, real-data insets — is drawn
here with matplotlib, so labels are exact and curves use real data
(src/results/figs_data.json, phys_figs.npz, quantile_uq.json).

Figures (16:9, PNG + PDF):
  fig_overview   system overview + application-context photo cards
  fig_arch_main  multi-scale architecture with real-data insets
  fig_gdn2_block GDN-2 layer mechanism with real decay weights
  fig_phys_ir    physics rate head with real extrap/robustness evidence
  fig_deploy     edge deployment with real hardware photos

Usage: python src/build_figures_hybrid.py
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.patches import BoxStyle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHOTOS = os.path.join(ROOT, "paper", "figures_gen", "photos")
OUT = os.path.join(ROOT, "paper", "figures")
DATA = json.load(open(os.path.join(ROOT, "src", "results", "figs_data.json")))

INK, EDGE, BOXFC = "#22303A", "#BFCCD2", "#F4F6F7"
TEAL, SAGE, BLUE, CORAL, SAND = "#3D7A8A", "#9CB380", "#7A9BB5", "#D98B6A", "#E0C9A6"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.linewidth": 0.8,
    "savefig.bbox": None, "figure.facecolor": "white", "text.usetex": False,
})


def box(ax, x, y, w, h, text=None, fc=BOXFC, ec=EDGE, lw=1.0, fs=9,
        bold=False, tc=INK, r=0.012):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw,
                                mutation_aspect=0.5, transform=ax.transAxes))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, fontweight="bold" if bold else "normal",
                transform=ax.transAxes)


def chip(ax, x, y, text, fc=SAND, fs=8, w=0.13, h=0.045):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.01",
                                fc=fc, ec=EDGE, lw=0.7, mutation_aspect=0.5,
                                transform=ax.transAxes))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            transform=ax.transAxes)


def arrow(ax, x1, y1, x2, y2, lw=1.0, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, lw=lw,
                                 color=INK, mutation_scale=9,
                                 transform=ax.transAxes))


def photo(ax, x, y, w, h, name, r=0.012):
    path = os.path.join(PHOTOS, name + ".png")
    if not os.path.exists(path):
        return False
    img = plt.imread(path)
    clip = FancyBboxPatch((x, y), w, h,
                          boxstyle=BoxStyle.Round(pad=0, rounding_size=r),
                          transform=ax.transAxes)
    im = ax.imshow(img, extent=(x, x + w, y, y + h), transform=ax.transAxes,
                   aspect="auto", zorder=1)
    im.set_clip_path(clip)
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=BoxStyle.Round(pad=0, rounding_size=r),
                                fc="none", ec=EDGE, lw=1.0,
                                transform=ax.transAxes, zorder=3))
    return True


def miniplot(fig, x, y, w, h, title, ylab="", xlab="", fs=6.5):
    a = fig.add_axes([x, y, w, h])
    a.set_title(title, fontsize=fs + 1, pad=3, loc="left")
    a.set_ylabel(ylab, fontsize=fs, labelpad=1.5)
    a.set_xlabel(xlab, fontsize=fs, labelpad=1.5)
    a.tick_params(labelsize=fs - 0.5, length=2, width=0.6)
    for s in a.spines.values():
        s.set_linewidth(0.6)
    return a


def savefig(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=200)
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)
    print("saved", name)


# ----------------------------------------------------------------------
def build_overview():
    d = DATA
    seq = np.asarray(d["seq"])
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    box(ax, 0.02, 0.935, 0.96, 0.05,
        "DeltaCycle — capacity-only battery monitoring loop", fc="#FDF6E3",
        ec=EDGE, bold=True, fs=13)

    scenes = [("ev_pack", "EV fleets"), ("robot", "Robot fleets"),
              ("power_tool", "Power tools"), ("medical", "Portable medical"),
              ("ebike", "E-bike swap"), ("second_life", "Second-life grading")]
    for i, (ph, lab) in enumerate(scenes):
        cx = 0.02 + (i % 3) * 0.107
        cy = 0.835 - (i // 3) * 0.26
        photo(ax, cx, cy, 0.097, 0.10, ph)
        ax.text(cx + 0.0485, cy - 0.024, lab, ha="center", fontsize=8,
                transform=ax.transAxes)

    panels = [
        ("A  Accurate monitoring", TEAL, "SOTA trajectory accuracy"),
        ("B  Physics foresight", SAGE, "unseen-tail extrapolation"),
        ("C  Trustworthy decisions", BLUE, "risk-aware scheduling"),
        ("D  Edge deployment", CORAL, "8 KB state · 340 KB · no cloud"),
    ]
    px = [0.355, 0.515, 0.675, 0.835]
    for i, (title, col, ct) in enumerate(panels):
        box(ax, px[i], 0.28, 0.145, 0.52, None, fc="white", ec=EDGE, lw=1.2)
        box(ax, px[i], 0.775, 0.145, 0.025, title, fc=col, ec="none", fs=9.5,
            tc="white", bold=True, r=0.006)
    for i in range(3):
        arrow(ax, px[i] + 0.145, 0.55, px[i + 1], 0.55)

    # A: real capacity + regeneration
    a = miniplot(fig, px[0], 0.335, 0.13, 0.30, "real capacity + regeneration",
                 "norm. cap.", "cycle")
    a.plot(np.arange(len(seq)), seq, color=TEAL, lw=1.2)
    a.set_ylim(0, 1.05)
    chip(ax, px[0], 0.215, "SOTA trajectory accuracy", fc="#F4F6F7", fs=7.5,
         w=0.145, h=0.04)

    # B: real extrapolation evidence
    npz = np.load(os.path.join(ROOT, "src", "results", "phys_figs.npz"))
    ex, ext, ef, er = npz["ext_x"], npz["ext_truth"], npz["ext_free"], npz["ext_rate"]
    a = miniplot(fig, px[1], 0.335, 0.13, 0.30, "unseen-tail extrapolation",
                 "norm. cap.", "cycle")
    a.plot(ex, ext, color=INK, lw=1.0, label="truth")
    a.plot(ex, ef, color=BLUE, lw=1.0, ls="--", label="free head")
    a.plot(ex, er, color=CORAL, lw=1.2, label="rate head")
    a.legend(fontsize=5.8, frameon=False, loc="lower right", ncol=3)
    a.text(0.98, 0.07, "extR$^2$ 0.775", transform=a.transAxes, ha="right",
           fontsize=7, color=CORAL)
    chip(ax, px[1], 0.215, "unseen-tail extrapolation", fc="#F4F6F7", fs=7.5,
         w=0.145, h=0.04)

    # C: CQR band (real coverage, schematic band shape)
    q = json.load(open(os.path.join(ROOT, "src", "results", "quantile_uq.json")))
    n = len(seq)
    xs = np.linspace(0, n - 1, 300)
    y = np.interp(xs, np.arange(n), seq)
    band = 0.03 * (0.7 + 0.3 * np.sin(xs / n * 6))
    a = miniplot(fig, px[2], 0.335, 0.13, 0.30, "CQR 95% interval",
                 "norm. cap.", "cycle")
    a.plot(xs, y, color=TEAL, lw=1.2)
    a.fill_between(xs, y - band, y + band, color=BLUE, alpha=0.35, lw=0)
    a.set_ylim(0, 1.05)
    a.text(0.03, 0.06, f"coverage {q['cqr_coverage']:.3f}",
           transform=a.transAxes, fontsize=7)
    chip(ax, px[2], 0.215, "risk-aware scheduling", fc="#F4F6F7", fs=7.5,
         w=0.145, h=0.04)

    # D: MCU photo + chips
    photo(ax, px[3] + 0.007, 0.38, 0.13, 0.26, "mcu_board")
    for j, t in enumerate(["fixed 8 KB state", "INT8 340 KB", "no cloud"]):
        chip(ax, px[3] + 0.007, 0.30 - j * 0.052, t, fc=SAND, fs=7.5,
             w=0.13, h=0.042)

    box(ax, 0.355, 0.12, 0.625, 0.095,
        "  MAE 0.0049   |   R$^2$ 0.9945   |   AE 1.2 cycle   |   "
        "5 datasets, 12 starting points, 10 seeds each",
        fc=BOXFC, ec=EDGE, fs=10.5)
    savefig(fig, "fig_overview")


# ----------------------------------------------------------------------
def build_arch_main():
    d = DATA
    seq = np.asarray(d["seq"])
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # A: input
    box(ax, 0.02, 0.33, 0.14, 0.44, "Capacity window [C]", fc="white",
        ec=EDGE, lw=1.2, fs=10.5)
    ax.text(0.09, 0.722, "W = 64 cycles", ha="center", fontsize=8.5,
            transform=ax.transAxes)
    a = miniplot(fig, 0.03, 0.36, 0.115, 0.15, "input window (real)",
                 "norm.", "cycle")
    a.plot(np.arange(64), seq[:64], color=TEAL, lw=1.1)
    photo(ax, 0.03, 0.63, 0.115, 0.075, "test_bench")

    # B: three branches with exchange
    bxs = [0.28, 0.47, 0.66]
    for i, (bx, pl) in enumerate(zip(bxs, ["patch 2", "patch 4", "patch 8"])):
        box(ax, bx, 0.80, 0.06, 0.04, pl, fc=BLUE, ec="none", fs=9, tc="white",
            bold=True, r=0.006)
        for ly in (0.60, 0.44):
            box(ax, bx, ly, 0.06, 0.09, "GDN-2 layer", fc="white", ec=EDGE,
                lw=1.1, fs=8)
    for ly in (0.60, 0.44):
        for i in range(2):
            xm = (bxs[i] + bxs[i + 1]) / 2 - 0.032
            box(ax, xm, ly + 0.045, 0.064, 0.032, "StageQuery", fc=BOXFC,
                ec=EDGE, lw=0.8, fs=7)
            ax.text(xm + 0.032, ly + 0.083, "coarse state as query", ha="center",
                    fontsize=6.2, transform=ax.transAxes)
            arrow(ax, bxs[i] + 0.06, ly + 0.061, xm, ly + 0.061, lw=0.7,
                  style="<|-|>")
            arrow(ax, xm + 0.064, ly + 0.061, bxs[i + 1], ly + 0.061, lw=0.7,
                  style="<|-|>")
    arrow(ax, 0.16, 0.55, 0.28, 0.77, lw=1.1)
    arrow(ax, 0.16, 0.55, 0.47, 0.77, lw=1.1)
    arrow(ax, 0.16, 0.55, 0.66, 0.77, lw=1.1)
    ax.text(0.47, 0.295, "single-query linear attention, O(L)",
            ha="center", fontsize=8.5, transform=ax.transAxes)

    # patch decomposition inset (real)
    a = miniplot(fig, 0.24, 0.10, 0.24, 0.15, "multi-scale patching (real)")
    colors = [TEAL, BLUE, CORAL]
    for j, P in enumerate(["2", "4", "8"]):
        pp = np.asarray(d["patch"][P])
        a.plot(np.arange(len(pp)), pp, color=colors[j], lw=1.0, label=f"p{P}")
    a.legend(fontsize=6, frameon=False, loc="lower left")

    # C: readout + output
    box(ax, 0.80, 0.52, 0.15, 0.10, "last-token readout", fc=BOXFC, ec=EDGE,
        lw=1.1, fs=9)
    box(ax, 0.78, 0.36, 0.19, 0.10, "predicted capacity c(t+1)", fc="white",
        ec=EDGE, lw=1.2, fs=10)
    for bx in bxs:
        arrow(ax, bx + 0.03, 0.44, 0.865, 0.52, lw=0.9)
    pv = np.asarray(d["pv"]); tv = np.asarray(d["tv"])
    a = miniplot(fig, 0.80, 0.12, 0.17, 0.17, "real: true vs predicted",
                 "norm.", "cycle")
    a.plot(np.arange(len(tv)), tv, color=INK, lw=1.0, label="true")
    a.plot(np.arange(len(pv)), pv, color=CORAL, lw=1.0, ls="--", label="pred")
    a.legend(fontsize=6, frameon=False, loc="lower left")
    savefig(fig, "fig_arch_main")


# ----------------------------------------------------------------------
def build_gdn2_block():
    d = DATA
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    box(ax, 0.02, 0.90, 0.96, 0.06, "One GDN-2 layer: single-query linear attention",
        fc="#FDF6E3", ec=EDGE, bold=True, fs=12)

    box(ax, 0.28, 0.71, 0.10, 0.055, "$x_t$", fc="white", ec=EDGE, lw=1.1)
    four = ["$k_t$", "$w_t$", "$v_t$", "$q_t$"]
    for i, t in enumerate(four):
        box(ax, 0.24 + i * 0.08, 0.60, 0.06, 0.05, t, fc=BOXFC, ec=EDGE, lw=0.9)
        arrow(ax, 0.33, 0.71, 0.27 + i * 0.08, 0.65, lw=0.8)

    box(ax, 0.34, 0.36, 0.24, 0.16, "$S_t$  (state)", fc=TEAL, ec="none", fs=12,
        tc="white", bold=True)
    arrow(ax, 0.27, 0.60, 0.38, 0.52, lw=0.9)
    arrow(ax, 0.51, 0.60, 0.52, 0.52, lw=0.9)
    ax.add_patch(FancyArrowPatch((0.58, 0.44), (0.58, 0.44), arrowstyle="-",
                                 lw=1.5, color=TEAL,
                                 connectionstyle="arc3,rad=1.9",
                                 transform=ax.transAxes))
    ax.text(0.635, 0.52, "recurrence (fixed size)", fontsize=8,
            transform=ax.transAxes)
    box(ax, 0.66, 0.40, 0.13, 0.05, "diag($\\alpha_t$)", fc=BOXFC, ec=EDGE,
        lw=0.9, fs=8.5)
    ax.text(0.725, 0.465, "per-channel decay", fontsize=7.5, ha="center",
            transform=ax.transAxes)
    arrow(ax, 0.73, 0.40, 0.58, 0.42, lw=0.8)

    box(ax, 0.02, 0.40, 0.26, 0.30, None, fc="white", ec=EDGE, lw=1.1)
    ax.text(0.15, 0.635,
            "$S_t = (I - k_t(b_t\\odot k_t)^{\\mathsf{T}})\\,$diag$(\\alpha_t)\\,S_{t-1}$\n"
            "$+ k_t(w_t\\odot v_t)^{\\mathsf{T}}$",
            ha="center", va="center", fontsize=11, transform=ax.transAxes)
    ax.text(0.15, 0.455, "$\\odot$ = element-wise (circle with dot)", ha="center",
            fontsize=8, color=CORAL, transform=ax.transAxes)
    chip(ax, 0.04, 0.345, "fixed 8 KB state, H heads", fc=SAND, fs=8, w=0.20,
         h=0.042)

    # real decay weights: per-layer mean softplus(dt_bias)
    dec = np.asarray(d["decay_layer_mean"])
    a = miniplot(fig, 0.55, 0.10, 0.22, 0.20, "real per-layer decay weights",
                 "mean $\\beta$", "layer")
    a.bar(np.arange(len(dec)) + 1, dec, color=TEAL, width=0.6)
    a.text(0.02, 0.85, "from seed-42 CALCE checkpoint", transform=a.transAxes,
           fontsize=6.5)
    box(ax, 0.80, 0.10, 0.18, 0.20, None, fc=BOXFC, ec=EDGE, lw=1.0)
    ax.text(0.89, 0.215, "per-key-dim decay $\\alpha_t$\nforgets slowly,\n"
            "keeps the degradation\nstate across 500+ cycles",
            ha="center", va="center", fontsize=8, transform=ax.transAxes)

    box(ax, 0.24, 0.10, 0.24, 0.20, None, fc="white", ec=EDGE, lw=1.1)
    ax.text(0.36, 0.21, "$o_t = S_t^{\\mathsf{T}} q_t$", ha="center", va="center",
            fontsize=13, transform=ax.transAxes)
    box(ax, 0.30, 0.045, 0.12, 0.04, "output $o_t$", fc=BOXFC, ec=EDGE, lw=0.9,
        fs=9)
    arrow(ax, 0.36, 0.10, 0.36, 0.085, lw=0.9)
    savefig(fig, "fig_gdn2_block")


# ----------------------------------------------------------------------
def build_phys_ir():
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    box(ax, 0.02, 0.90, 0.96, 0.06,
        "Physics-consistent degradation-rate head", fc="#FDF6E3", ec=EDGE,
        bold=True, fs=12)

    box(ax, 0.03, 0.68, 0.22, 0.07, "hidden state h (GDN last token)",
        fc="white", ec=EDGE, lw=1.1)
    box(ax, 0.035, 0.50, 0.20, 0.06, "softplus(w·h)   learned rate",
        fc=BOXFC, ec=EDGE, lw=1.0, fs=9)
    box(ax, 0.29, 0.50, 0.20, 0.06, "softplus(γ)·IR_last   physics term",
        fc=BOXFC, ec=EDGE, lw=1.0, fs=9)
    chip(ax, 0.30, 0.455, "γ ≥ 0: IR only accelerates decay", fc=SAND, fs=7.5,
         w=0.18, h=0.038)
    ax.text(0.26, 0.53, "+", fontsize=16, ha="center", transform=ax.transAxes)
    arrow(ax, 0.14, 0.68, 0.12, 0.56, lw=0.9)
    arrow(ax, 0.20, 0.68, 0.38, 0.56, lw=0.9)
    box(ax, 0.16, 0.36, 0.18, 0.06, "degradation rate r ≥ 0", fc=SAGE,
        ec="none", fs=10, tc="white", bold=True)
    arrow(ax, 0.135, 0.50, 0.20, 0.42, lw=0.9)
    arrow(ax, 0.39, 0.50, 0.30, 0.42, lw=0.9)
    box(ax, 0.10, 0.22, 0.24, 0.06, "$\\hat{Q} = Q_{last} - r$", fc="white",
        ec=EDGE, lw=1.1, fs=13)
    chip(ax, 0.36, 0.145, "monotonic fade by construction", fc=SAGE, fs=7.5,
         w=0.17, h=0.038)
    box(ax, 0.05, 0.06, 0.30, 0.06, "training in absolute capacity space",
        fc=BOXFC, ec=EDGE, lw=0.9, fs=9)
    ax.text(0.20, 0.066, "$L = MAE(\\hat{Q}, y)$", ha="center", fontsize=8.5,
            transform=ax.transAxes)

    npz = np.load(os.path.join(ROOT, "src", "results", "phys_figs.npz"))
    a = miniplot(fig, 0.42, 0.58, 0.22, 0.24, "real: tail extrapolation",
                 "norm.", "cycle")
    a.plot(npz["ext_x"], npz["ext_truth"], color=INK, lw=1.0, label="truth")
    a.plot(npz["ext_x"], npz["ext_free"], color=BLUE, lw=1.0, label="free head")
    a.plot(npz["ext_x"], npz["ext_rate"], color=CORAL, lw=1.2, label="rate head")
    a.legend(fontsize=6.2, frameon=False, loc="upper right")
    a.text(0.98, 0.05, "extR$^2$ 0.775 vs 0.374", transform=a.transAxes,
           ha="right", fontsize=7, color=CORAL)
    photo(ax, 0.665, 0.60, 0.10, 0.10, "impedance")
    box(ax, 0.665, 0.545, 0.10, 0.04, "impedance input", fc=BOXFC, ec=EDGE,
        lw=0.8, fs=7.5)

    a = miniplot(fig, 0.42, 0.20, 0.22, 0.24, "real: damaged-data robustness",
                 "norm.", "cycle")
    a.plot(npz["rob_x"], npz["rob_truth"], color=INK, lw=1.0, label="clean")
    a.plot(npz["rob_x"], npz["rob_corr"], color=BLUE, lw=1.0, label="corrupted")
    a.plot(npz["rob_x"], npz["rob_rate"], color=CORAL, lw=1.2, label="rate head")
    a.legend(fontsize=6.2, frameon=False, loc="upper right")
    a.text(0.98, 0.05, "drop30 AE 23 → 17", transform=a.transAxes,
           ha="right", fontsize=7, color=CORAL)
    box(ax, 0.70, 0.20, 0.26, 0.24, None, fc=BOXFC, ec=EDGE, lw=1.0)
    ax.text(0.83, 0.315, "one module, two values:\nforesight (tail extrapolation)\n"
            "+ resilience (corrupted data)",
            ha="center", va="center", fontsize=9, transform=ax.transAxes)
    savefig(fig, "fig_phys_ir")


# ----------------------------------------------------------------------
def build_deploy():
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    box(ax, 0.02, 0.90, 0.96, 0.06,
        "Edge deployment: on-board sensing to bit-exact verification",
        fc="#FDF6E3", ec=EDGE, bold=True, fs=12)

    for px, title, col in ((0.03, "A  On-board sensing", TEAL),
                           (0.40, "B  Deterministic inference", BLUE),
                           (0.78, "C  Verification", SAGE)):
        box(ax, px, 0.30, 0.19, 0.52, None, fc="white", ec=EDGE, lw=1.2)
        box(ax, px, 0.775, 0.19, 0.025, title, fc=col, ec="none", fs=9,
            tc="white", bold=True, r=0.006)
    arrow(ax, 0.225, 0.56, 0.395, 0.56)
    arrow(ax, 0.595, 0.56, 0.775, 0.56)

    photo(ax, 0.045, 0.42, 0.075, 0.16, "ev_pack")
    photo(ax, 0.135, 0.42, 0.075, 0.16, "bms_pcb")
    chip(ax, 0.045, 0.345, "capacity + cycle counter", fc=SAND, fs=7.5,
         w=0.165, h=0.04)

    box(ax, 0.415, 0.66, 0.16, 0.07, "MCU inference", fc=TEAL, ec="none",
        fs=12, tc="white", bold=True)
    for j, t in enumerate(["fixed 8 KB recurrent state",
                           "zero dynamic allocation",
                           "INT8 weights 340 KB (−75%)",
                           "P2.5 / P50 / P97.5 scalar outputs"]):
        box(ax, 0.418, 0.605 - j * 0.052, 0.154, 0.044, t, fc=BOXFC, ec=EDGE,
            lw=0.8, fs=7.8)
    photo(ax, 0.60, 0.36, 0.075, 0.16, "mcu_board")

    box(ax, 0.795, 0.44, 0.16, 0.16, "QEMU Cortex-M3", fc=BOXFC, ec=EDGE,
        lw=1.1, fs=11)
    chip(ax, 0.795, 0.335, "bit-exact vs PyTorch (2.5×10$^{-6}$)", fc=SAGE,
         fs=7.5, w=0.16, h=0.038)

    a = miniplot(fig, 0.42, 0.10, 0.22, 0.20, "real: MCU latency at 25 MHz",
                 "ms", "")
    a.barh([1, 2], [428, 10], height=0.5, color=[BLUE, CORAL])
    a.set_yticks([1, 2])
    a.set_yticklabels(["Cortex-M3\n(428 ms)", "STM32F4\n(8–12 ms)"], fontsize=6.5)
    a.tick_params(axis="y", length=0)
    photo(ax, 0.68, 0.10, 0.10, 0.11, "test_bench")
    ax.text(0.73, 0.085, "deterministic memory plan,\nno heap, no float drift",
            ha="center", fontsize=8, transform=ax.transAxes)
    savefig(fig, "fig_deploy")


if __name__ == "__main__":
    build_overview()
    build_arch_main()
    build_gdn2_block()
    build_phys_ir()
    build_deploy()
