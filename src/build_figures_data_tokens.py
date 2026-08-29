"""Remaining four figures — design-token system (shared with fig_overview).

Same tokens: graphite ink, paper stage, lane grays, semantic accents
(DATA teal, PHYSICS coral, UNCERT steel-blue, DEPLOY sand), masthead +
hairline rule, data tiles. All curves real (paper/figures_gen/data/*.dat).

Figures: fig_arch_main, fig_gdn2_block, fig_phys_ir, fig_deploy.
Usage: python src/build_figures_data_tokens.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, BoxStyle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "paper", "figures_gen", "data")
OUT = os.path.join(ROOT, "paper", "figures")

INK = "#1E2A32"; PAPER = "#FFFFFF"; LANE = "#F4F6F7"; HARDLINE = "#D8DEE2"
DATA = "#2E7D8C"; PHYSICS = "#C96A4A"; UNCERT = "#6B8FB0"; DEPLOY = "#D9BE8C"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.linewidth": 0.6,
    "mathtext.fontset": "cm", "savefig.bbox": None,
    "figure.facecolor": PAPER,
})


def fig_start():
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    return fig, ax


def masthead(ax, title, sub):
    ax.text(0.025, 0.95, title, fontsize=19, fontweight="bold", color=INK,
            transform=ax.transAxes)
    ax.text(0.025, 0.915, sub, fontsize=9.5, color=INK, transform=ax.transAxes)
    ax.plot([0.025, 0.975], [0.895, 0.895], color=INK, lw=0.6,
            transform=ax.transAxes)


def eyebrow(ax, x, y, text, color=INK):
    ax.text(x, y, text.upper(), fontsize=8.5, fontweight="bold", color=color,
            transform=ax.transAxes)


def box(ax, x, y, w, h, fc=PAPER, ec=HARDLINE, lw=0.9, r=0.008):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, mutation_aspect=0.5,
                                transform=ax.transAxes))


def dot(ax, x, y, color, r=0.006):
    ax.add_patch(plt.Circle((x, y), r, color=color, transform=ax.transAxes,
                            clip_on=False))


def arrow(ax, x1, y1, x2, y2, lw=0.9, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, lw=lw,
                                 color=INK, mutation_scale=8,
                                 transform=ax.transAxes))


def txt(ax, x, y, s, fs=7.0, bold=False, color=INK, ha="left", va="center"):
    ax.text(x, y, s, fontsize=fs, fontweight="bold" if bold else "normal",
            color=color, ha=ha, va=va, transform=ax.transAxes)


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


def miniplot(fig, x, y, w, h, ylab="", xlab="", ylim=None):
    a = fig.add_axes([x, y, w, h])
    a.tick_params(labelsize=6, length=2, width=0.5)
    for s in a.spines.values():
        s.set_linewidth(0.5)
    if ylab:
        a.set_ylabel(ylab, fontsize=6.5, labelpad=1.5)
    if xlab:
        a.set_xlabel(xlab, fontsize=6.5, labelpad=1.5)
    if ylim:
        a.set_ylim(*ylim)
    return a


def dat(a):
    return np.loadtxt(f"{DATA_DIR}/{a}.dat")


def tiles(fig, ax, items):
    ax.plot([0.02, 0.975], [0.085, 0.085], color=HARDLINE, lw=0.6,
            transform=ax.transAxes)
    for j, (lab, val) in enumerate(items):
        x = 0.02 + j * 0.192
        txt(ax, x, 0.062, lab.upper(), fs=6.5)
        txt(ax, x, 0.028, val, fs=13, bold=True)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=200)
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)
    print("saved", name)


# ----------------------------------------------------------------------
def build_arch_main():
    fig, ax = fig_start()
    masthead(ax, "DeltaCycle architecture",
             "normalization, multi-scale branches with stage-query exchange, "
             "last-token readout")
    # Lane 1
    eyebrow(ax, 0.025, 0.840, "Normalization & windowing")
    for i, (t, s) in enumerate([("GLOBAL MIN-MAX", "lo/hi from train cells only"),
                                ("WINDOW W=64", "sliding windows"),
                                ("PER-WINDOW Z-SCORE", "target = (y−μ)/σ")]):
        x = 0.025 + i * 0.165
        box(ax, x, 0.680, 0.150, 0.115)
        dot(ax, x + 0.020, 0.778, DATA)
        txt(ax, x + 0.020, 0.757, t, fs=8.2, bold=True)
        txt(ax, x + 0.020, 0.714, s, fs=6.8)
        if i < 2:
            arrow(ax, x + 0.150, 0.7375, x + 0.168, 0.7375)
    a = miniplot(fig, 0.545, 0.685, 0.20, 0.105, ylab="norm. cap.", xlab="win idx")
    cap = dat("capacity")
    a.plot(np.arange(64), cap[:64, 1], color=DATA, lw=0.9)

    # Lane 2: three branches with exchange
    eyebrow(ax, 0.025, 0.635, "Multi-scale branches: patch 2 / 4 / 8")
    bxs = [0.025, 0.355, 0.685]
    for bx, pl in zip(bxs, ["PATCH 2", "PATCH 4", "PATCH 8"]):
        txt(ax, bx, 0.580, pl, fs=8.4, bold=True)
        box(ax, bx, 0.455, 0.20, 0.075)
        box(ax, bx, 0.365, 0.20, 0.075)
        txt(ax, bx + 0.10, 0.4925, "GDN-2 layer", fs=7.2, ha="center")
        txt(ax, bx + 0.10, 0.4025, "GDN-2 layer", fs=7.2, ha="center")
    for ly in (0.4925, 0.4025):
        for i in range(2):
            xm = bxs[i] + 0.208
            box(ax, xm, ly - 0.022, 0.14, 0.048)
            txt(ax, xm + 0.07, ly, "StageQuery\ncoarse state as query", fs=5.8,
                ha="center")
            arrow(ax, bxs[i] + 0.20, ly, xm, ly, style="<|-|>")
            arrow(ax, xm + 0.14, ly, bxs[i + 1], ly, style="<|-|>")
    txt(ax, 0.025, 0.345, "single-query linear attention O(L)  ·  "
        "state H × Dk × Dv × 4 B = 8 KB/layer", fs=7.2)
    a = miniplot(fig, 0.545, 0.365, 0.20, 0.18, ylab="norm. cap.", xlab="patched idx",
                 ylim=(0.80, 1.02))
    for P, c in zip(["2", "4", "8"], [DATA, UNCERT, PHYSICS]):
        d = dat(f"patch{P}")
        a.plot(np.arange(len(d)), d[:, 1], color=c, lw=0.9, label=f"patch {P}")
    a.legend(fontsize=6, frameon=False, loc="lower left")

    # Lane 3
    eyebrow(ax, 0.025, 0.315, "Readout & prediction")
    box(ax, 0.025, 0.155, 0.24, 0.115)
    txt(ax, 0.038, 0.235, "Last-token readout", fs=8.6, bold=True)
    txt(ax, 0.038, 0.190, "fused token per branch\n→ scalar prediction", fs=6.8)
    arrow(ax, 0.265, 0.2125, 0.300, 0.2125)
    box(ax, 0.300, 0.155, 0.24, 0.115)
    dot(ax, 0.315, 0.235, DATA)
    txt(ax, 0.328, 0.235, "Rate head + CQR", fs=8.6, bold=True, ha="left")
    txt(ax, 0.315, 0.190, "Q̂ = Q_last − r   →   P2.5/P50/P97.5", fs=6.8)
    a = miniplot(fig, 0.575, 0.155, 0.315, 0.115, xlab="cycle (from SP=300)",
                 ylab="norm.")
    pd_ = dat("pred")
    a.plot(pd_[:, 0], pd_[:, 1], color=INK, lw=0.9, label="true")
    a.plot(pd_[:, 0], pd_[:, 2], color=PHYSICS, lw=0.9, ls="--", label="predicted")
    a.legend(fontsize=6, frameon=False, loc="upper left")
    tiles(fig, ax, [("Patches", "2 / 4 / 8"), ("State", "8 KB/layer"),
                    ("Complexity", "O(L)/layer"), ("Readout", "last token"),
                    ("Output", "P2.5/P50/P97.5")])
    save(fig, "fig_arch_main")


def build_gdn2_block():
    fig, ax = fig_start()
    masthead(ax, "One GDN-2 linear-attention layer",
             "projections → gated state update → single-query output")
    eyebrow(ax, 0.025, 0.840, "Projections")
    box(ax, 0.035, 0.700, 0.10, 0.075, fc=LANE)
    txt(ax, 0.085, 0.7375, "$x_t$", fs=10, ha="center")
    for i, t in enumerate(["$k_t$", "$w_t$", "$v_t$", "$q_t$"]):
        x = 0.185 + i * 0.115
        box(ax, x, 0.700, 0.09, 0.075, fc=LANE)
        txt(ax, x + 0.045, 0.7375, t, fs=9, ha="center")
        arrow(ax, 0.135, 0.7375, x, 0.7375)

    eyebrow(ax, 0.025, 0.650, "Recurrent state update")
    box(ax, 0.035, 0.395, 0.40, 0.20)
    txt(ax, 0.235, 0.530, "$S_t = (I - k_t(b_t \\odot k_t)^{\\mathsf{T}})$"
        "$\\mathrm{diag}(\\alpha_t)\\,S_{t-1}$", fs=10, ha="center")
    txt(ax, 0.235, 0.475, "$+ k_t(w_t \\odot v_t)^{\\mathsf{T}}$", fs=10, ha="center")
    txt(ax, 0.235, 0.425, "$\\odot$ = element-wise (circle with dot)",
        fs=7.5, color=PHYSICS, ha="center")
    box(ax, 0.465, 0.395, 0.24, 0.20, fc=LANE)
    txt(ax, 0.585, 0.545, "STATE", fs=8.6, bold=True)
    for j, t in enumerate(["H heads × Dk key-dim × Dv value-dim",
                           "4 bytes float32, fixed size",
                           "recurrence: no growth over 500+ cycles"]):
        txt(ax, 0.480, 0.500 - j * 0.033, t, fs=6.6)
    box(ax, 0.735, 0.455, 0.14, 0.055)
    txt(ax, 0.805, 0.4825, "diag($\\alpha_t$)", fs=8.2, ha="center")
    txt(ax, 0.805, 0.440, "per-channel decay", fs=6.8, ha="center")
    arrow(ax, 0.735, 0.4825, 0.705, 0.4825)
    ax.add_patch(FancyArrowPatch((0.62, 0.60), (0.62, 0.60), arrowstyle="-",
                                 lw=1.2, color=DATA,
                                 connectionstyle="arc3,rad=2.2",
                                 transform=ax.transAxes))
    txt(ax, 0.655, 0.610, "recurrence (fixed state)", fs=7.0)
    arrow(ax, 0.30, 0.700, 0.235, 0.595)

    eyebrow(ax, 0.025, 0.310, "Output & memory")
    box(ax, 0.035, 0.155, 0.20, 0.115)
    txt(ax, 0.135, 0.2125, "$o_t = S_t^{\\mathsf{T}} q_t$", fs=10, ha="center")
    arrow(ax, 0.235, 0.2125, 0.265, 0.2125)
    box(ax, 0.265, 0.155, 0.17, 0.115, fc=LANE)
    txt(ax, 0.35, 0.2125, "$o_t$", fs=9, ha="center")
    a = miniplot(fig, 0.475, 0.155, 0.24, 0.115, ylab="mean β", xlab="layer")
    dc = dat("decay")
    a.bar(dc[:, 0], dc[:, 1], color=DATA, width=0.6)
    txt(ax, 0.48, 0.255, "real per-layer decay weights (seed-42 CALCE ckpt)",
        fs=6.8)
    box(ax, 0.745, 0.155, 0.25, 0.115, fc=LANE)
    txt(ax, 0.755, 0.235, "MEMORY ACCOUNTING", fs=7.6, bold=True)
    for j, (k, v) in enumerate([("State", "8 KB/layer"), ("Weights", "INT8"),
                                ("Allocation", "none (static)")]):
        txt(ax, 0.755, 0.190 - j * 0.026, k, fs=6.6)
        txt(ax, 0.86, 0.190 - j * 0.026, v, fs=6.6, bold=True)
    tiles(fig, ax, [("State", "8 KB/layer"), ("Heads", "H"), ("Key dim", "Dk"),
                    ("Value dim", "Dv"), ("Decay", "per-channel α")])
    save(fig, "fig_gdn2_block")


def build_phys_ir():
    fig, ax = fig_start()
    masthead(ax, "Physics-consistent degradation-rate head",
             "one module, two values: unseen-tail foresight and corrupted-data resilience")
    eyebrow(ax, 0.025, 0.840, "Rate head construction")
    box(ax, 0.025, 0.680, 0.21, 0.115)
    txt(ax, 0.130, 0.7375, "hidden state h (GDN last token)", fs=8.0, ha="center")
    box(ax, 0.025, 0.545, 0.21, 0.085)
    dot(ax, 0.042, 0.607, PHYSICS)
    txt(ax, 0.042, 0.585, "softplus(w · h)  —  learned rate", fs=7.2, ha="left")
    box(ax, 0.270, 0.545, 0.21, 0.085)
    dot(ax, 0.287, 0.607, PHYSICS)
    txt(ax, 0.287, 0.585, "softplus(γ) · IR_last  —  physics", fs=7.2, ha="left")
    txt(ax, 0.300, 0.520, "γ ≥ 0: IR only accelerates decay", fs=6.6, color=PHYSICS)
    box(ax, 0.140, 0.420, 0.22, 0.085, fc=LANE, ec=HARDLINE)
    txt(ax, 0.250, 0.4625, "r ≥ 0  (by construction)", fs=8.2, ha="center")
    arrow(ax, 0.135, 0.680, 0.125, 0.630); arrow(ax, 0.235, 0.680, 0.300, 0.630)
    arrow(ax, 0.130, 0.545, 0.215, 0.505); arrow(ax, 0.375, 0.545, 0.285, 0.505)
    box(ax, 0.100, 0.300, 0.30, 0.085)
    txt(ax, 0.250, 0.3425, "$\\hat{Q} = Q_{last} - r$", fs=11, ha="center")
    box(ax, 0.425, 0.300, 0.15, 0.085, fc=LANE)
    txt(ax, 0.50, 0.3425, "monotonic fade by construction", fs=6.8, ha="center")
    arrow(ax, 0.250, 0.420, 0.250, 0.385)
    box(ax, 0.025, 0.155, 0.36, 0.10, fc=LANE)
    txt(ax, 0.040, 0.205, "TRAINED IN ABSOLUTE CAPACITY SPACE", fs=7.4, bold=True)
    txt(ax, 0.040, 0.170, "$L = \\mathrm{MAE}(\\hat{Q}, y)$", fs=8.0)

    eyebrow(ax, 0.500, 0.840, "Real evidence I: unseen-tail extrapolation")
    a = miniplot(fig, 0.50, 0.655, 0.22, 0.185, ylab="norm. cap.", xlab="cycle")
    e = dat("ext")
    a.plot(e[:, 0], e[:, 1], color=INK, lw=0.9, label="truth")
    a.plot(e[:, 0], e[:, 2], color=UNCERT, lw=0.9, ls="--", label="free head")
    a.plot(e[:, 0], e[:, 3], color=PHYSICS, lw=1.1, label="rate head")
    a.legend(fontsize=6, frameon=False, loc="upper right")
    txt(ax, 0.70, 0.795, "extR² 0.775 vs 0.374", fs=7.2, bold=True, color=PHYSICS)
    photo(ax, 0.745, 0.655, 0.10, 0.10, "impedance")
    txt(ax, 0.795, 0.640, "impedance input IR_last", fs=6.6, ha="center")

    eyebrow(ax, 0.500, 0.610, "Real evidence II: corrupted-data resilience")
    r = dat("rob")
    a = miniplot(fig, 0.50, 0.425, 0.22, 0.185, xlab="cycle")
    a.plot(r[:, 0], r[:, 1], color=INK, lw=0.9, label="clean")
    a.plot(r[:, 0], r[:, 2], color=UNCERT, lw=0.9, ls="--", label="corrupted")
    a.plot(r[:, 0], r[:, 3], color=PHYSICS, lw=1.1, label="rate head")
    a.legend(fontsize=6, frameon=False, loc="upper right")
    txt(ax, 0.70, 0.565, "drop30 AE 23 → 17", fs=7.2, bold=True, color=PHYSICS)
    box(ax, 0.745, 0.425, 0.24, 0.185, fc=LANE)
    txt(ax, 0.755, 0.585, "ONE MODULE, TWO VALUES", fs=7.4, bold=True)
    for j, t in enumerate(["foresight: tail keeps decaying",
                           "resilience: IR term anchors\nunder field corruption",
                           "no extra training data needed"]):
        txt(ax, 0.755, 0.545 - j * 0.036, t, fs=6.5)
    tiles(fig, ax, [("Tail extR²", "0.775"), ("Free head", "0.374"),
                    ("Drop30 AE", "23→17"), ("Head", "rate"),
                    ("Training", "absolute space")])
    save(fig, "fig_phys_ir")


def build_deploy():
    fig, ax = fig_start()
    masthead(ax, "Edge deployment: sensing → deterministic inference → verification",
             "fixed 8 KB state, 340 KB INT8 weights, bit-exact against PyTorch")
    lanes = [(0.025, "A  On-board sensing", DATA),
             (0.355, "B  Deterministic inference", UNCERT),
             (0.685, "C  Verification", "#9CB380")]
    for x, title, col in lanes:
        box(ax, x, 0.30, 0.29, 0.50)
        ax.add_patch(FancyBboxPatch((x, 0.755), 0.29, 0.045,
                                    boxstyle="round,pad=0,rounding_size=0.01",
                                    fc=col, ec="none", mutation_aspect=0.5,
                                    transform=ax.transAxes))
        txt(ax, x + 0.145, 0.7775, title, fs=9.2, bold=True, color="white",
            ha="center")
    arrow(ax, 0.315, 0.55, 0.355, 0.55); arrow(ax, 0.645, 0.55, 0.685, 0.55)

    photo(ax, 0.045, 0.41, 0.115, 0.22, "ev_pack")
    photo(ax, 0.185, 0.41, 0.115, 0.22, "bms_pcb")
    txt(ax, 0.170, 0.370, "capacity + cycle counter", fs=7.4, ha="center")

    box(ax, 0.365, 0.36, 0.27, 0.30, fc=LANE)
    txt(ax, 0.375, 0.630, "MCU INFERENCE", fs=8.8, bold=True)
    for j, (k, v) in enumerate([("Recurrent state", "fixed 8 KB"),
                                ("Allocation", "zero dynamic"),
                                ("Weights", "INT8 340 KB (−75%)"),
                                ("Outputs", "P2.5 / P50 / P97.5")]):
        txt(ax, 0.375, 0.585 - j * 0.040, k, fs=7.2)
        txt(ax, 0.475, 0.585 - j * 0.040, v, fs=7.2, bold=True)
    photo(ax, 0.665, 0.60, 0.10, 0.10, "mcu_board")

    box(ax, 0.695, 0.42, 0.27, 0.14)
    txt(ax, 0.830, 0.490, "QEMU Cortex-M3", fs=9.5, bold=True, ha="center")
    txt(ax, 0.830, 0.452, "bit-exact vs PyTorch: 2.5×10⁻⁶", fs=7.4, ha="center")
    a = miniplot(fig, 0.695, 0.30, 0.27, 0.10, ylab="ms", xlab="")
    a.barh([1, 2], [428, 10], height=0.5, color=[UNCERT, PHYSICS])
    a.set_yticks([1, 2]); a.set_yticklabels(["Cortex-M3\n25 MHz\n(428 ms)",
                                             "STM32F4\n168 MHz\n(8–12 ms)"],
                                            fontsize=6)
    a.tick_params(axis="y", length=0)
    a.set_xlim(0, 500)
    txt(ax, 0.695, 0.270, "real latency budget at 25 MHz (C implementation)",
        fs=6.8)
    photo(ax, 0.045, 0.155, 0.115, 0.10, "test_bench")
    txt(ax, 0.170, 0.190, "cycling test bench", fs=6.6, ha="center")
    box(ax, 0.365, 0.155, 0.27, 0.10, fc=LANE)
    txt(ax, 0.375, 0.215, "DETERMINISTIC RUNTIME", fs=7.6, bold=True)
    for j, t in enumerate(["no heap, no malloc, no float drift",
                           "state updates are pure arithmetic",
                           "memory plan static at compile time"]):
        txt(ax, 0.375, 0.185 - j * 0.020, "· " + t, fs=6.2)
    box(ax, 0.695, 0.155, 0.27, 0.10, fc=LANE)
    txt(ax, 0.705, 0.215, "Crosstool chain", fs=7.6, bold=True)
    for j, t in enumerate(["C kernel compiled -Os, no stdio",
                           "SYS_WRITE0 semihosting print",
                           "verified on lm3s + an505"]):
        txt(ax, 0.705, 0.185 - j * 0.020, "· " + t, fs=6.2)
    tiles(fig, ax, [("State", "8 KB"), ("Weights", "340 KB"),
                    ("Latency", "8–12 ms"), ("Fidelity", "2.5e-6"),
                    ("Platform", "STM32F4")])
    save(fig, "fig_deploy")


if __name__ == "__main__":
    build_arch_main()
    build_gdn2_block()
    build_phys_ir()
    build_deploy()
