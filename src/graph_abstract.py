"""Graphical abstract for DeltaCycle (v7) -- Elsevier published-example style.

Canvas 17.0 x 10.0 cm (1.7:1), PNG 2677 x 1575 px at 400 dpi: above Elsevier's
minimum 531 x 1328 px (height x width), >= 300 dpi, Arial only.  The ratio is
NOT fixed by Elsevier -- their own published graphical abstracts (AJKD,
Kidney Medicine) sit at ~1.5-1.8:1; what they enforce is the pixel minimum and
legible type.  Display sites fit-to-width and letterbox, they never squash,
so a taller canvas buys real type size: 6-7.5 pt body here vs 4.5 pt at 2.5:1.

Visual language copied from the published examples:
  * tinted full-column backgrounds + filled header bars (Kidney Medicine)
  * one hero shape per column, thick few arrows, big boxes (placebo example)
  * headline numbers set LARGE, detail left to the paper (Kidney Medicine)
  * bordered CONCLUSION callout across the bottom (AJKD)

Vertical metrics from real extents (1 pt = 1/283.5 of canvas height):
  6 pt tick row + tick/pad  -> TICK_BAND 0.032
  7.5 pt item title         -> LINE_H    0.034
  7 pt claim strip          -> CLAIM_H   0.040
Every chart reserves height + TICK_BAND; y labels are NEVER rotated ax
labels (units live in titles) so nothing can leave its column.

Usage: python src/make_graphical_abstract_v5.py
"""
import os
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
sys.path.insert(0, _SRC)

# Must precede the rcParams block: make_figures.py sets savefig.bbox='tight'
# and axes.grid=True at import, and matplotlib MERGES rcParams dicts.
from make_figures import load_series   # noqa: E402


# --------------------------------------------------------------------------
# style
# --------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 6,
    "savefig.bbox": None,
    "savefig.pad_inches": 0,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "axes.grid": False,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 1.6,
    "ytick.major.size": 1.6,
})

NAVY = "#1B3A5C"
BLUE = "#0F4D92"
BLUE_FILL = "#DCE7F3"
CORAL = "#C8603C"
SAGE_GREEN = "#7C9A6D"
SAND = "#F6F1E4"
SAGE = "#E7EFE2"
INK = "#1A1A1A"
GRAY = "#3A3A3A"
GRAY_L = "#C4C4C4"
WHITE = "#FFFFFF"
COL_BG = ("#EEF3FA", "#F5F4EF", "#EEF5EE")   # per-column tint, Kidney-Medicine style

W_CM, H_CM = 17.0, 10.0
FIG_W, FIG_H = W_CM / 2.54, H_CM / 2.54

TITLE_Y, TITLE_H = 0.915, 0.085
HDR_Y, HDR_H = 0.845, 0.052
BODY_T, BODY_B = 0.830, 0.150
CONC_Y, CONC_H = 0.020, 0.115

MARGIN = 0.015
COL_X = (0.015, 0.372, 0.724)
COL_W = (0.345, 0.340, 0.261)

TICK_BAND = 0.032
LINE_H = 0.034
CLAIM_H = 0.040

fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=300)
fig.patch.set_facecolor("white")

_checks = []


def fit(artist, width, tag):
    _checks.append((artist, width, tag))
    return artist


def verify_fit():
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bad = []
    for art, allowed, tag in _checks:
        got = art.get_window_extent(renderer=r).width / (FIG_W * fig.dpi)
        if got > allowed + 1e-9:
            bad.append((tag, art.get_text(), got, allowed))
    for tag, s, got, allowed in bad:
        print(f"  OVERFLOW [{tag}] {got:.3f} > {allowed:.3f}  {s!r}")
    print(f"fit check: {len(_checks) - len(bad)}/{len(_checks)} strings fit")
    return bad


def rect(x, y, w, h, fc, ec="none", lw=0.0, z=1):
    fig.patches.append(Rectangle((x, y), w, h, linewidth=lw, facecolor=fc,
                                 edgecolor=ec, transform=fig.transFigure,
                                 figure=fig, zorder=z))


def txt(x, y, s, size=6, color=INK, weight="normal", ha="left", **kw):
    return fig.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                    ha=ha, va="center", zorder=6, **kw)


def arrow(x0, y0, x1, y1, color=NAVY, lw=1.2, ms=7.0, z=6, rad=0.0):
    fig.patches.append(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms, linewidth=lw,
        color=color, transform=fig.transFigure, figure=fig, zorder=z,
        connectionstyle=f"arc3,rad={rad}"))


def chart(x, y_top, w, h, gut_l=0.0):
    """Axes plus reserved tick band below; caller advances by h + TICK_BAND."""
    return fig.add_axes([x + gut_l, y_top - h, w - gut_l, h])


def bare(ax, xticks=None, xlabels=None, yticks=None, ylabels=None):
    """Left + bottom spines stay (the y axis line); top/right hidden.
    Never set ax x/y labels: they render outside the axes rect."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRAY)
    ax.tick_params(colors=GRAY, pad=1.0, labelsize=6, length=1.6)
    if xticks is not None:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)
    if yticks is not None:
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks if ylabels is None else ylabels)


def title_row(col_x, y_top, text):
    txt(col_x + 0.006, y_top - LINE_H / 2, text, size=7.5, weight="bold",
        color=NAVY)
    return y_top - LINE_H


def strip(col_x, col_w, y_top, text, size=7):
    rect(col_x + 0.006, y_top - CLAIM_H, col_w - 0.012, CLAIM_H, SAGE, z=2)
    t = txt(col_x + 0.014, y_top - CLAIM_H / 2, text, size=size, color=NAVY,
            weight="bold")
    fit(t, col_w - 0.028, "claim")
    return y_top - CLAIM_H


def vbox(col_x, col_w, y_top, h, line1, line2):
    """White process box, bold lead line + detail line."""
    rect(col_x + 0.010, y_top - h, col_w - 0.020, h, WHITE, ec=NAVY, lw=0.8,
         z=3)
    t1 = txt(col_x + col_w / 2, y_top - 0.32 * h, line1, size=6.5, color=NAVY,
             ha="center", weight="bold")
    t2 = txt(col_x + col_w / 2, y_top - 0.70 * h, line2, size=6, color=GRAY,
             ha="center")
    fit(t1, col_w - 0.030, "box1")
    fit(t2, col_w - 0.030, "box2")
    return y_top - h


# ==========================================================================
# title bar + column headers + column tints
# ==========================================================================
rect(MARGIN, TITLE_Y, 1.0 - 2 * MARGIN, TITLE_H, NAVY, z=2)
txt(MARGIN + 0.016, TITLE_Y + TITLE_H / 2, "DeltaCycle", size=13, color=WHITE,
    weight="bold")
t = txt(MARGIN + 0.118, TITLE_Y + TITLE_H / 2,
        ":  multi-scale linear attention for calibrated on-device battery "
        "prognostics", size=9.5, color="#C9D8E8")
fit(t, 1.0 - MARGIN - 0.118 - 0.014, "title")
for i, name in enumerate(("METHOD", "EVIDENCE", "OUTCOME")):
    rect(COL_X[i], BODY_B - 0.006, COL_W[i],
         (HDR_Y + HDR_H) - (BODY_B - 0.006), COL_BG[i], z=-1)
    rect(COL_X[i], HDR_Y, COL_W[i], HDR_H, NAVY, z=2)
    txt(COL_X[i] + 0.012, HDR_Y + HDR_H / 2, name, size=9, color=WHITE,
        weight="bold")

# ==========================================================================
# COLUMN 1 -- METHOD: one vertical flow, big boxes, few arrows
# ==========================================================================
x0, w = COL_X[0], COL_W[0]
y = BODY_T

y = title_row(x0, y, "Capacity is the only input")

caps, _tr, cell, _W, _sps, _eol = load_series("panasonic")
full = np.asarray(caps[cell], dtype=float)
_d = np.diff(full)
rises = np.where(_d > 0.004)[0]
rises = rises[(rises > 30) & (rises < len(full) - 30)]
c0 = len(full) // 2
if len(rises):
    _b, _n = rises[0], 0
    for c in rises:
        k = int(np.sum(np.abs(rises - c) < 30))
        if k > _n:
            _b, _n = c, k
    c0 = int(_b)
Z = 24
z0 = max(0, min(c0 - Z // 2, len(full) - Z))
seg = full[z0:z0 + Z]

H_SP = 0.185
ax = chart(x0, y, w, H_SP, gut_l=0.030)
ax.plot(np.arange(Z), seg, color=INK, lw=1.2, solid_capstyle="round")
bare(ax, xticks=[0, 8, 16, 24], xlabels=["0", "8", "16", "24"],
     yticks=[round(float(seg.min()), 2), round(float(seg.max()), 2)])
y -= H_SP + TICK_BAND

for _ in range(3):                      # three flow arrows, thick and few
    arrow(x0 + w / 2, y - 0.004, x0 + w / 2, y - 0.022)
    y -= 0.026
    if _ == 0:
        y = vbox(x0, w, y, 0.078, "multi-scale patch windows  2 / 4 / 8 cycles",
                 "12 / 6 / 3 segments per scale, shared weights")
    elif _ == 1:
        y = vbox(x0, w, y, 0.078, "Gated DeltaNet-2 x2 per scale",
                 "cross-scale exchange: coarse state queries finer scales")
    else:
        y = vbox(x0, w, y, 0.078, "fused readout, three heads",
                 "capacity  |  P2.5 / P50 / P97.5  |  rate + IR")

y -= 0.010
y = strip(x0, w, y, "runs in C on a Cortex-M3, 48 KB fixed state")
y -= 0.006
y = strip(x0, w, y, "no voltage, current or temperature used")
assert y >= BODY_B - 1e-9, f"METHOD overruns body by {BODY_B - y:.4f}"

# ==========================================================================
# COLUMN 2 -- EVIDENCE: two charts + one big-number block
# ==========================================================================
x0, w = COL_X[1], COL_W[1]
GUT = 0.026
EV_H = 0.185
y = BODY_T

y = title_row(x0, y, "Calibrated intervals, capacity vs cycle")
_z = np.load(os.path.join(_ROOT, "results", "ga_uq_band.npz"))
xb, lo_b, mid_b, hi_b = _z["x"], _z["lo"], _z["mid"], _z["hi"]
true_b, eol_b = _z["true"], float(_z["eol"])
ax = chart(x0, y, w, EV_H, gut_l=GUT)
ax.fill_between(xb, lo_b, hi_b, color=BLUE_FILL, linewidth=0, zorder=1)
ax.plot(xb, true_b, color=INK, lw=0.9, zorder=4)
ax.plot(xb, mid_b, color=BLUE, lw=0.9, zorder=3)
miss = (true_b < lo_b) | (true_b > hi_b)
if miss.any():
    ax.plot(xb[miss], true_b[miss], "v", ms=1.4, color=CORAL, zorder=5,
            markeredgewidth=0)
ax.axhline(eol_b, color=CORAL, lw=0.7, ls=(0, (2.4, 1.6)), zorder=2)
ax.set_xlim(xb[0], xb[-1])
ax.set_ylim(min(lo_b.min(), true_b.min()) - 0.005,
            max(hi_b.max(), true_b.max()) + 0.008)
bare(ax, xticks=[xb[0], 400, 800], xlabels=["64", "400", "800"],
     yticks=[0.2, 0.6, 1.0])
y -= EV_H + TICK_BAND
y = strip(x0, w, y, "93.5% coverage at 95% nominal")
y -= 0.010

y = title_row(x0, y, "Physics rate head, unseen tail")
_z2 = np.load(os.path.join(_SRC, "results", "phys_figs.npz"))
ex = _z2["ext_x"].astype(float)
tr = _z2["ext_truth"].astype(float)
ax = chart(x0, y, w, EV_H, gut_l=GUT)
ax.plot(ex, _z2["ext_free"], color=CORAL, lw=1.0, zorder=3)
ax.plot(ex, _z2["ext_rate"], color=BLUE, lw=1.5, zorder=4)
ax.plot(ex, tr, color=INK, lw=0.9, zorder=5)
ax.set_xlim(ex[0], ex[-1])
ax.set_ylim(0.02, 0.46)
bare(ax, xticks=[ex[0], 840, ex[-1]], xlabels=["792", "840", "880"],
     yticks=[0.1, 0.3])
for _i, (_t, _c) in enumerate((("measured", INK), ("+ physics", BLUE),
                               ("free head", CORAL))):
    txt(x0 + w - 0.010, y - 0.020 - _i * 0.026, _t, size=6, color=_c,
        ha="right", weight="bold")
y -= EV_H + TICK_BAND
y = strip(x0, w, y, "tail R2 0.37 to 0.77 with physics")
y -= 0.010

# ablation as ONE big number (Kidney-Medicine style); the 3x4 table lives in
# the paper's tab:ablation and was unreadable at thumbnail size anyway
rect(x0 + 0.006, y - 0.070, w - 0.012, 0.070, WHITE, ec=NAVY, lw=0.8, z=3)
txt(x0 + 0.018, y - 0.035, "-5.7%", size=13, color=BLUE, weight="bold")
txt(x0 + 0.082, y - 0.024, "mean MAE drop with exchange;", size=6, color=GRAY)
txt(x0 + 0.082, y - 0.048, "22/30 pairs improve, p = 0.014", size=6,
    color=GRAY)
y -= 0.070
assert y >= BODY_B - 1e-9, f"EVIDENCE overruns body by {BODY_B - y:.4f}"

# ==========================================================================
# COLUMN 3 -- OUTCOME
# ==========================================================================
x0, w = COL_X[2], COL_W[2]
y = BODY_T
GUT_BAR = 0.066
GUT_MEM = 0.036

y = title_row(x0, y, "Six public datasets")
ax = chart(x0, y, w, 0.270, gut_l=GUT_BAR)
DS = ["PANASONIC", "TJU", "GOTION", "MIT", "NASA", "CALCE"]
# MAE in Ah -- the unit the paper reports. These arrays held normalized values
# in an earlier draft. The factor is per dataset, so it cancels in the ratio;
# the bars and the printed labels are unchanged, only the unit of record is.
AH = np.array([1.0194, 0.7673, 6.3904, 0.2610, 0.8815, 0.9923])
OURS = np.array([0.003767, 0.001500, 0.008033, 0.002533, 0.008900, 0.007867]) * AH
BEST = np.array([0.007733, 0.001967, 0.008733, 0.003200, 0.007767, 0.006933]) * AH
ratio = OURS / BEST
yy = np.arange(6)[::-1].astype(float)
# grey wall ending exactly at 1.00 IS the baseline reference; no dashed line
# to collide with anything, value labels sit clear of every bar end.
ax.barh(yy, np.ones(6), height=0.62, color=GRAY_L, edgecolor="none", zorder=3)
ax.barh(yy, ratio, height=0.62,
        color=[BLUE if r < 1 else CORAL for r in ratio], edgecolor="none",
        zorder=4)
ax.set_xlim(0, 1.35)
ax.set_ylim(-0.5, 5.5)
bare(ax, xticks=[0.5, 1.0], xlabels=["0.5", "1.00 = best"], yticks=yy,
     ylabels=DS)
ax.tick_params(axis="y", length=0, pad=1.5)
for _r, _yv in zip(ratio, yy):
    ax.text(_r + 0.04, _yv, f"{_r:.2f}", fontsize=6, fontweight="bold",
            color=INK, va="center", zorder=6)
y -= 0.270 + TICK_BAND
y = strip(x0, w, y, "first on 4, second on 2", size=6.5)
y -= 0.012

y = title_row(x0, y, "Memory vs context (KB)")
TOK = np.logspace(np.log2(16), np.log2(4096), 60, base=2)
ax = chart(x0, y, w, 0.180, gut_l=GUT_MEM)
ax.plot(TOK, 4.0 * TOK ** 2 * 4.0 / 1024.0, color=CORAL, lw=1.0, zorder=3)
ax.plot(TOK, 6.0 * 2.0 * TOK * 64.0 * 4.0 / 1024.0, color=SAGE_GREEN, lw=1.0,
        ls=(0, (3.0, 1.7)), zorder=3)
ax.plot(TOK, np.full_like(TOK, 48.0), color=BLUE, lw=2.0, zorder=4)
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlim(13, 4200)
ax.set_ylim(0.4, 4e5)
ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
bare(ax, xticks=[16, 256, 4096], xlabels=["16", "256", "4096"],
     yticks=[16, 4096, 65536])
for _i, (_t, _c, _b) in enumerate((("attention scores (L2)", CORAL, False),
                                   ("KV cache (L)", SAGE_GREEN, False),
                                   ("recurrent state (ours)", BLUE, True))):
    txt(x0 + GUT_MEM + 0.008, y - 0.022 - _i * 0.026, _t, size=6, color=_c,
        weight="bold" if _b else "normal")
y -= 0.180 + TICK_BAND
y = strip(x0, w, y, "state stays 48 KB as context grows", size=6.5)
assert y >= BODY_B - 1e-9, f"OUTCOME overruns body by {BODY_B - y:.4f}"

# ==========================================================================
# CONCLUSION -- bordered callout, AJKD style
# ==========================================================================
rect(MARGIN, CONC_Y, 1.0 - 2 * MARGIN, CONC_H, SAND, ec=NAVY, lw=1.0, z=2)
t = txt(0.5, CONC_Y + CONC_H / 2,
        "One capacity-only model that runs inside a battery-management "
        "microcontroller (Cortex-M3, 48 KB state);\nintervals stay calibrated "
        "(93.5% coverage), tail stays physical (R2 0.77), rank first on 4 of "
        "6 datasets.", size=7.5, color=NAVY, weight="bold", ha="center",
        linespacing=1.6)
fit(t, 1.0 - 2 * MARGIN - 0.05, "conclusion")

# ==========================================================================
out = os.path.join(_ROOT, "submission", "ga_v5")
os.makedirs(os.path.dirname(out), exist_ok=True)
verify_fit()
plt.rcParams["savefig.bbox"] = None
fig.savefig(out + ".pdf")
fig.savefig(out + ".png", dpi=400)

from PIL import Image   # noqa: E402
Image.open(out + ".png").convert("RGB").resize((500, 294), Image.LANCZOS) \
     .save(os.path.join(os.path.dirname(out), "_ga_v5_preview500.png"))
print("wrote", out + ".pdf / .png and _ga_v5_preview500.png (500x294 preview)")
