#!/usr/bin/env python3
"""Fig.7 edge-deployment story: photo-driven flow, pure matplotlib.

Row A (sensing): cycler bench -> 18650 cell (capacity+cycle count) -> BMS.
Row B (deterministic on-board inference): memory card, MCU board photo,
quantization/UQ outputs.
Row C (verification): libm bit-exact table, QEMU latency, <3e-6 full-scan diff.

Photos (already AI-generated photorealistic) are embedded via imshow;
conceptual elements are matplotlib rectangles/annotations. Output is a
single vector PDF + PNG preview.
"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle, FancyBboxPatch
from PIL import Image
import numpy as np
from pathlib import Path

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent
PH = ROOT / "photos"
FIG = (ROOT / ".." / "figures").resolve()

INK = "#1F262B"
MUTED = "#707A82"
TEAL = "#2E7D8C"
RED = "#D64533"
SAGE = "#4C8C5A"
BORDER = "#C9D1D8"
BOX = "#F3F5F7"

plt.rcParams.update({
    "font.family": "sans-serif",
    "mathtext.fontset": "cm",
    "savefig.bbox": None,
    "text.color": INK,
})


def cover_crop(im, ratio):
    """Center-crop a PIL image to the target pixel aspect ratio (cover)."""
    w, h = im.size
    if w / h > ratio:          # too wide -> crop sides
        nw = int(h * ratio)
        x0 = max((w - nw) // 2, 0)
        im = im.crop((x0, 0, x0 + nw, h))
    else:                       # too tall -> crop top/bottom
        nh = int(w / ratio)
        y0 = max((h - nh) // 2, 0)
        im = im.crop((0, y0, w, y0 + nh))
    return im


def photo(ax, img_name, xy, w, h, label, label_y=None):
    """Place a photo with white frame + caption below (in fig fractions)."""
    axf = fig.add_axes([xy[0], xy[1], w, h])
    im = Image.open(PH / img_name)
    axf.imshow(cover_crop(im, w / h))
    axf.set_xticks([]); axf.set_yticks([])
    for s in axf.spines.values():
        s.set_color(BORDER); s.set_linewidth(1.0)
    axf.set_aspect("auto")
    if label:
        fig.text(xy[0] + w / 2, label_y if label_y else xy[1] + h + 0.008,
                 label, ha="center", va="bottom", fontsize=8.5,
                 color=MUTED, fontstyle="italic")
    return axf


def card(ax, xy, w, h, fc=BOX, ec=BORDER):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.35,rounding_size=0.012",
                                fc=fc, ec=ec, lw=1.0, zorder=2))


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2,
                                shrinkA=0, shrinkB=0), zorder=3)


fig = plt.figure(figsize=(12.6, 7.1), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.axis("off")

# ---------------- Row A: sensing ----------------
card(ax, (0.020, 0.845), 0.96, 0.125, fc="#EAF1F8", ec="#8FB3CE")
fig.text(0.5, 0.958, "A · ON-BOARD SENSING", ha="center", va="center",
         fontsize=9, fontweight="bold", color=TEAL)

photo(ax, "test_bench.png",  (0.045, 0.848), 0.22, 0.075, "cycling bench")
photo(ax, "cell18650.png",   (0.398, 0.848), 0.185, 0.075, "capacity + cycle count")
photo(ax, "bms_pcb.png",     (0.725, 0.848), 0.22, 0.075, "on-board BMS")
arrow(ax, 0.325, 0.885, 0.375, 0.885)
arrow(ax, 0.665, 0.885, 0.715, 0.885)

# ---------------- Row B: deterministic inference ----------------
card(ax, (0.020, 0.520), 0.96, 0.315, fc=BOX)
fig.text(0.5, 0.822, "B · DETERMINISTIC ON-BOARD INFERENCE · FULL NETWORK IN C",
         ha="center", va="center", fontsize=9, fontweight="bold", color=INK)

# left: memory card
card(ax, (0.035, 0.535), 0.225, 0.275, fc="#EAF1F8")
fig.text(0.035 + 0.1125, 0.795, "DETERMINISTIC MEMORY", ha="center",
         fontsize=7.5, fontweight="bold", color=TEAL, va="center")
rows = [("state", "16 KB  (8 KB × 2 layers)", 2.3),
        ("weights", "340 KB  INT8 (−75%)", 100),
        ("alloc", "zero dynamic malloc", 0)]
for i, (lab, val, frac) in enumerate(rows):
    y = 0.755 - i * 0.062
    fig.text(0.045, y, lab, fontsize=7.5, ha="left", va="center")
    ax.add_patch(Rectangle((0.115, y - 0.016), 0.135, 0.032,
                           fc="#FFFFFF", ec=BORDER, lw=0.7, zorder=4))
    f = frac / 100.0
    ax.add_patch(Rectangle((0.115, y - 0.016), max(0.135 * f, 0.012), 0.032,
                           fc="#2E7D8C" if i < 2 else "#4C8C5A",
                           alpha=0.75, lw=0, zorder=5))
    fig.text(0.255, y, val, fontsize=7, ha="right", va="center")

# KV-cache contrast (below memory card)
fig.text(0.035 + 0.1125, 0.565, "KV-CACHE GROWTH (per layer)", ha="center",
         fontsize=7.0, fontweight="bold", color=MUTED, va="center")
for i, (lab, val) in enumerate([("self-attn", "grows O(L)"),
                                ("GDN-2", "fixed 16 KB")]):
    y = 0.545 - i * 0.026
    fig.text(0.045, y, lab, fontsize=6.8, ha="left", va="center")
    ax.add_patch(Rectangle((0.115, y - 0.009), 0.135, 0.018,
                           fc="#F6EED8", ec=BORDER, lw=0.5, zorder=4))
    ax.add_patch(Rectangle((0.115, y - 0.009),
                           0.135 * (0.78 if i == 0 else 0.023), 0.018,
                           fc=(RED if i == 0 else TEAL), alpha=0.8, lw=0,
                           zorder=5))
    fig.text(0.255, y, val, fontsize=6.8, ha="right", va="center")

# center: MCU photo
photo(ax, "mcu_wide.png", (0.275, 0.530), 0.42, 0.290, None)
fig.text(0.275 + 0.21, 0.805, "BMS MCU · full network in C",
         ha="center", va="center", fontsize=7.5, fontweight="bold",
         color=INK, zorder=6)

# right: quantization + outputs
card(ax, (0.710, 0.535), 0.255, 0.275, fc="#F6EFE9")
fig.text(0.710 + 0.1275, 0.795, "UQ / OUTPUTS", ha="center", fontsize=7.5,
         fontweight="bold", color=RED, va="center")
fig.text(0.725, 0.752, "INT8 lossless", fontsize=6.8, ha="left",
         va="center", color=MUTED)
ax.add_patch(Rectangle((0.855, 0.740), 0.095, 0.024, fc="#FFFFFF",
                       ec=BORDER, lw=0.6, zorder=4))
ax.add_patch(Rectangle((0.855, 0.740), 0.095, 0.024, fc=RED, alpha=0.75,
                       lw=0, zorder=5))
fig.text(0.955, 0.752, "AE 2.2 → 2.3", fontsize=6.8, ha="right",
         va="center", color=INK)
fig.text(0.725, 0.700, "calibrated outputs → RUL", fontsize=7.2,
         ha="left", va="center", color=SAGE, fontweight="bold")
for i, q in enumerate(["P2.5", "P50", "P97.5"]):
    x = 0.725 + i * 0.075
    ax.add_patch(Rectangle((x, 0.668), 0.062, 0.024, fc="#FFFFFF",
                           ec=BORDER, lw=0.6, zorder=4))
    fig.text(x + 0.031, 0.680, q, fontsize=6.5, ha="center", va="center")
fig.text(0.725, 0.600, "capacity-only input\n+ cycle counter", fontsize=6.8,
         ha="left", va="center", color=MUTED, linespacing=1.4)

# ---------------- Row C: verification ----------------
card(ax, (0.020, 0.055), 0.96, 0.455, fc=BOX)
fig.text(0.5, 0.495, "C · VERIFICATION · QEMU CORTEX-M3, BIT-EXACT vs PyTorch",
         ha="center", va="center", fontsize=9, fontweight="bold", color=SAGE)

# left: libm table
card(ax, (0.035, 0.075), 0.29, 0.400, fc="#ECF2E9")
fig.text(0.18, 0.455, "LIBM · BIT-EXACT ARM vs X86", ha="center",
         fontsize=7.5, fontweight="bold", color=SAGE, va="center")
for i, f in enumerate(["expf", "logf", "sqrtf", "erff"]):
    y = 0.415 - i * 0.052
    fig.text(0.065, y, f, fontsize=8, ha="left", va="center",
             fontfamily="monospace")
    fig.text(0.30, y, "identical ✓", fontsize=8, ha="right", va="center",
             color=SAGE)

# center: MCU chip photo (no overlay)
photo(ax, "chip_closeup.png", (0.345, 0.075), 0.31, 0.325, None)
fig.text(0.5, 0.435, "STM32-class MCU", ha="center", va="center",
         fontsize=7.5, color=INK, fontweight="bold")

# right: <3e-6 + latency bars
card(ax, (0.675, 0.075), 0.29, 0.400, fc="#F6EFE9")
fig.text(0.82, 0.390, "< 3 × 10⁻⁶", ha="center", va="center",
         fontsize=17, fontweight="bold", color=RED)
fig.text(0.82, 0.345, "full-scan diff vs PyTorch", ha="center", fontsize=7.5,
         va="center", color=INK)
fig.text(0.82, 0.305, "no heap · no float drift ·\nstatic allocation",
         ha="center", va="center", fontsize=6.8, color=MUTED,
         linespacing=1.5)
fig.text(0.82, 0.255, "QEMU-measured latency", ha="center", fontsize=7.0,
         color=MUTED, va="center", fontstyle="italic")
for i, (m, v, frac) in enumerate([("Cortex-M3 @25 MHz", "428 ms", 100.0),
                                  ("STM32F4 @168 MHz", "8–12 ms", 3.0)]):
    y = 0.222 - i * 0.044
    fig.text(0.690, y, m, fontsize=6.8, ha="left", va="center")
    ax.add_patch(Rectangle((0.815, y - 0.011), 0.135, 0.022,
                           fc="#FFFFFF", ec=BORDER, lw=0.6, zorder=4))
    ax.add_patch(Rectangle((0.815, y - 0.011), 0.135 * frac / 100.0, 0.022,
                           fc=(RED if i == 0 else TEAL), alpha=0.8,
                           lw=0, zorder=5))
    fig.text(0.955, y, v, fontsize=6.8, ha="right", va="center")
fig.text(0.80, 0.120, "input", fontsize=7, ha="center", va="center",
         color=MUTED, fontstyle="italic")
fig.text(0.845, 0.120, "→", fontsize=8, ha="center", va="center", color=INK)
fig.text(0.888, 0.120, "interval → RUL", fontsize=7, ha="center",
         va="center", color=MUTED, fontstyle="italic")

# foot strip
card(ax, (0.020, 0.008), 0.96, 0.038, fc="#F6EFE9")
fig.text(0.5, 0.027, "16 KB fixed state   ·   bit-exact   ·   8–12 ms per inference   ·   no cloud",
         ha="center", va="center", fontsize=7.5, color=INK)

fig.savefig(FIG / "fig_deploy_photo.pdf", facecolor="white", edgecolor="none",
            bbox_inches=None)
fig.savefig(FIG / "fig_deploy_photo.png", dpi=200, facecolor="white",
            edgecolor="none", bbox_inches=None)
print("ok fig_deploy_photo.pdf/png")
