#!/usr/bin/env python3
"""Fig.1 System overview - three rows, python layout (no HTML).

Row 1 (applications): 6 photo tiles (existing photos).
Row 2 (pipeline): train / model / infer (existing ms_*.png 2048x880).
Row 3 (capabilities A-D): 4 new photo-realistic tiles (ov_p3a..d).
No data charts anywhere.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch
from PIL import Image
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PH = ROOT / "photos"
GEN = ROOT
FIG = (ROOT / ".." / "figures").resolve()

INK = "#1F262B"
MUTED = "#707A82"
BORDER = "#C9D1D8"
DEMI = [("#2E7D8C", "A · First-Tier\nAccuracy"),
        ("#B08A2E", "B · Physics-\nConsistent\nHead"),
        ("#4C8C5A", "C · Decision-\nGrade UQ"),
        ("#C96A4A", "D · Edge\nDeployment")]
APPS = [("ebike.png", "E-bike Swap"), ("second_life.png", "Second Life"),
        ("medical.png", "Portable Medical"), ("ev_pack.png", "EV Fleet"),
        ("robot.png", "Robot Fleet"), ("power_tool.png", "Power Tools")]
PIPE = [("ms_train.png", "① TRAINING"),
        ("ms_model.png", "② MODEL STRUCTURE"),
        ("ms_infer.png", "③ INFERENCE")]
CAP = [("ov_p3a.png", None), ("ov_p3b_v4.png", None),
       ("ov_p3c.png", None), ("ov_p3d.png", None)]

plt.rcParams.update({"font.family": "sans-serif", "savefig.bbox": None,
                     "text.color": INK})


def cover_crop(im, ratio):
    w, h = im.size
    if w / h > ratio:
        nw = int(h * ratio); x0 = (w - nw) // 2
        return im.crop((x0, 0, x0 + nw, h))
    nh = int(w / ratio); y0 = (h - nh) // 2
    return im.crop((0, y0, w, y0 + nh))


def tile(ax, path, x, y, w, h, ratio):
    axf = fig.add_axes([x, y, w, h])
    axf.imshow(cover_crop(Image.open(path), ratio))
    axf.set_xticks([]); axf.set_yticks([])
    for s in axf.spines.values():
        s.set_color(BORDER); s.set_linewidth(1.0)
    return axf


fig = plt.figure(figsize=(10.5, 7.2), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.axis("off")

# ---------- Row 1: six application photos (y: 0.80-0.92) ----------
R1Y, R1H, R1W = 0.800, 0.100, 0.148
for i, (img, lab) in enumerate(APPS):
    x = 0.014 + i * 0.1635
    tile(ax, PH / img, x, R1Y, R1W, R1H, R1W / R1H)
    fig.text(x + R1W / 2, R1Y - 0.010, lab, ha="center", va="top",
             fontsize=7.0, color=INK, fontweight="bold")

# ---------- Row 2: train / model / infer (y: 0.505-0.655) ----------
R2Y, R2H, R2W = 0.505, 0.130, 0.295
for i, (img, lab) in enumerate(PIPE):
    x = 0.014 + i * 0.332
    fig.text(x + R2W / 2, R2Y + R2H + 0.012, lab, ha="center", va="bottom",
             fontsize=8.2, color="#2E7D8C", fontweight="bold")
    tile(ax, GEN / img, x, R2Y, R2W, R2H, R2W / R2H)

# ---------- Row 3: four capability tiles A-D (y: 0.16-0.32) ----------
R3Y, R3H, R3W = 0.165, 0.160, 0.225
for i, (color, lab) in enumerate(DEMI):
    x = 0.014 + i * 0.245
    tile(ax, GEN / CAP[i][0], x, R3Y, R3W, R3H, R3W / R3H)
    fig.text(x + R3W / 2, R3Y - 0.014, lab, ha="center", va="top",
             fontsize=7.9, color=color, fontweight="bold", linespacing=1.5)

# ---------- bottom strip ----------
fig.text(0.5, 0.040, "Five datasets (CALCE · NASA · MIT · PANASONIC · TJU) — 12 SPs × 10 seeds — per-SP truncation & retraining",
         ha="center", va="center", fontsize=8.5, color=MUTED)

fig.savefig(FIG / "fig_overview_new.pdf", facecolor="white", edgecolor="none")
fig.savefig(FIG / "fig_overview_new.png", dpi=180, facecolor="white",
            edgecolor="none")
print("ok fig_overview_new.pdf/png")
