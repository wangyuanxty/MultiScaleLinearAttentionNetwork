"""Graphical abstract for DeltaCycle, composed as a data figure.

Archetype: schematic-led composite. The accuracy comparison is the hero panel
-- it gets the most width and the largest type; the method and the headline
figures are subordinate.

The Elsevier examples are information graphics, not decorative schematics:
they carry the study's actual numbers. This composes the same way. Canvas is
13.28 x 5.31 cm, the 1328 x 531 aspect ratio Elsevier asks for; the output is
vector PDF so no dpi limit applies. Fonts are Arial, one of the four the guide
permits, and no glyph is below the 5 pt floor the figure contract sets.

Numbers are the verified values from Table 3 (per-SP means over ten seeds);
ours and the best baseline are both on the same normalized-capacity scale.

Writes submission/ga_data_figure.pdf and .png. Does not touch the existing
submission/graphical_abstract.* files.

Usage: cd src && python make_graphical_abstract.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 6,
    "savefig.bbox": None,          # the repo's rcParams merge tight-bbox in
    "savefig.pad_inches": 0,
    "pdf.fonttype": 42,            # editable text in the PDF
    "svg.fonttype": "none",
})

GREEN = "#166b34"
GREEN_L = "#2ca02c"
GREEN_PALE = "#cfe6d6"
GRAY = "#6f6f6f"
GRAY_L = "#cfcfcf"
INK = "#1f1f1f"
BG = "#f2f6f2"

PT = 1 / 72 / (5.31 / 2.54)      # one point, as a fraction of figure height
W_CM, H_CM = 13.28, 5.31

fig = plt.figure(figsize=(W_CM / 2.54, H_CM / 2.54), dpi=300)
fig.patch.set_facecolor("white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.axis("off")

# ------------------------------------------------------------------ title bar
ax.add_patch(FancyBboxPatch((0.008, 0.858), 0.984, 0.130,
                            boxstyle="round,pad=0,rounding_size=0.010",
                            linewidth=0, facecolor=GREEN))
ax.text(0.5, 0.950, "DeltaCycle", ha="center", va="center",
        fontsize=10, fontweight="bold", color="white")
ax.text(0.5, 0.888, "multi-scale linear attention for calibrated on-device battery prognostics",
        ha="center", va="center", fontsize=5.5, color=GREEN_PALE)

def header(cx, text):
    ax.text(cx, 0.822, text, ha="center", va="center",
            fontsize=7, fontweight="bold", color=GREEN)

C1 = (0.012, 0.288)     # method
C2 = (0.300, 0.700)     # hero
C3 = (0.712, 0.988)     # what it buys

# --------------------------------------------------------- column 1: method
header((C1[0] + C1[1]) / 2, "HOW IT WORKS")
x0, x1 = C1
mid = (x0 + x1) / 2

ax.text(mid, 0.755, "patch 2/4/8, exchanged per layer",
        ha="center", va="center", fontsize=5.5, color=INK)
for y, dash in [(0.712, (1.1, 1.1)), (0.690, (2.2, 1.4)), (0.668, (3.8, 1.6))]:
    ax.plot([x0 + 0.012, x1 - 0.014], [y, y], linestyle=(0, dash),
            linewidth=1.5, color=GREEN_L, solid_capstyle="butt")

for y, lab, fc, fw in [(0.612, "Gated DeltaNet-2", "white", "normal"),
                       (0.535, "fixed-size state", GREEN_PALE, "bold"),
                       (0.458, "physics rate head", "white", "normal")]:
    ax.add_patch(FancyBboxPatch((x0 + 0.020, y - 0.022), (x1 - x0) - 0.040, 0.044,
                                boxstyle="round,pad=0,rounding_size=0.007",
                                linewidth=0.7, edgecolor=GREEN, facecolor=fc))
    ax.text(mid, y, lab, ha="center", va="center", fontsize=5.5,
            color=INK, fontweight=fw)
for a in (0.590, 0.513):
    ax.add_patch(FancyArrowPatch((mid, a), (mid, a - 0.022), arrowstyle="-|>",
                                 mutation_scale=4, linewidth=0.6, color=GRAY))

ax.text(mid, 0.372, "one forward pass", ha="center", va="center",
        fontsize=5.5, color=GREEN, fontweight="bold")
ax.text(mid, 0.310, "P2.5 / P50 / P97.5", ha="center", va="center",
        fontsize=6.5, color=GREEN, fontweight="bold")
ax.text(mid, 0.250, "no ensemble, no sampling", ha="center", va="center",
        fontsize=5, color=GRAY)

# -------------------------------------------------------- column 2: the hero
header((C2[0] + C2[1]) / 2, "ACCURACY ACROSS SIX DATASETS")
x0, x1 = C2

DS = ["PANASONIC", "TJU", "GOTION", "MIT", "NASA", "CALCE"]
OURS = np.array([0.0038, 0.0015, 0.0080, 0.0025, 0.0089, 0.0079])
BEST = np.array([0.0077, 0.0020, 0.0087, 0.0032, 0.0078, 0.0069])
ratio = OURS / BEST

bx0 = x0 + 0.118
SCALE = (x1 - bx0 - 0.048) / 1.25
y_top, dy = 0.726, 0.054

ax.add_patch(FancyBboxPatch((x0 + 0.004, 0.406), (x1 - x0) - 0.008, 0.389,
                            boxstyle="round,pad=0,rounding_size=0.010",
                            linewidth=0.6, edgecolor=GREEN_PALE,
                            facecolor="#fafcfa"))
for i, (ds, r) in enumerate(zip(DS, ratio)):
    y = y_top - i * dy
    win = r < 1.0
    ax.text(bx0 - 0.008, y, ds, ha="right", va="center", fontsize=6,
            color=INK if win else GRAY)
    ax.plot([bx0, bx0 + SCALE * r], [y, y], linewidth=5,
            color=GREEN_L if win else GRAY_L, solid_capstyle="butt")
    ax.text(bx0 + SCALE * r + 0.006, y, f"{r:.2f}", ha="left", va="center",
            fontsize=6.5, color=GREEN if win else GRAY,
            fontweight="bold" if win else "normal")

ax.plot([bx0, bx0], [y_top - 5 * dy - 0.022, y_top + 0.022], linewidth=0.8,
        color=INK, linestyle=(0, (2.5, 1.8)))
ax.text(bx0 + 0.004, y_top + 0.040, "closest baseline = 1.00", ha="left",
        va="center", fontsize=5, color=INK)

ax.text((x0 + x1) / 2, 0.340, "MAE relative to the closest baseline",
        ha="center", va="center", fontsize=5.5, color=GRAY)
ax.text((x0 + x1) / 2, 0.285, "first on four, second on NASA and CALCE",
        ha="center", va="center", fontsize=6.5, color=GREEN, fontweight="bold")

# ----------------------------------------------------- column 3: what it buys
header((C3[0] + C3[1]) / 2, "WHAT IT BUYS")
x0, x1 = C3
rows = [
    ("calibrated coverage", "93.5%", "raw 89.1%, target 95%"),
    ("unseen-tail R2", "0.7745", "free head 0.374"),
    ("drop30 corruption MAE", "0.0074", "free head 0.0105"),
    ("weights INT8 / INT4", "504 / 268 KB", "fp32 1.85 MB"),
]
BH, PITCH = 0.145, 0.158
for i, (lab, big, sub) in enumerate(rows):
    yt = 0.790 - i * PITCH
    ax.add_patch(FancyBboxPatch((x0 + 0.006, yt - BH), (x1 - x0) - 0.012, BH,
                                boxstyle="round,pad=0,rounding_size=0.007",
                                linewidth=0.6, edgecolor=GREEN_PALE,
                                facecolor=BG))
    ax.text(x0 + 0.018, yt - 0.028, lab, ha="left", va="center",
            fontsize=5, color=GRAY)
    ax.text(x0 + 0.018, yt - 0.078, big, ha="left", va="center",
            fontsize=7, color=GREEN, fontweight="bold")
    ax.text(x0 + 0.018, yt - 0.120, sub, ha="left", va="center",
            fontsize=5, color=GRAY)

ax.text((x0 + x1) / 2, 0.135, "rate head: CALCE MAE  -53%", ha="center",
        va="center", fontsize=5.5, color=INK)

out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "submission", "ga_data_figure")
fig.savefig(out + ".pdf")
fig.savefig(out + ".png", dpi=600)
print("wrote", out + ".pdf and .png")

from PIL import Image
Image.open(out + ".png").convert("RGB").resize((500, 200), Image.LANCZOS)\
     .save(os.path.join(os.path.dirname(out), "_ga_preview500.png"))
print("wrote _ga_preview500.png")
