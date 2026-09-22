"""Graphical abstract for DeltaCycle (v4) -- Elsevier's own template structure.

Three earlier drafts are superseded.  v1 was a grid of text boxes.  v2 filled
panels with real plots but had no picture of the model.  v3 fixed the
composition but was still a Nature-style multi-panel figure, and the method
was nowhere on it.

The design here follows the structure of Elsevier's published graphical
abstracts (their template figure gives its middle column to methodology
-- "Here is where you showcase your methodology" -- and all three of their
published examples carry a method/cohort column):

    title bar
    METHOD | EVIDENCE | OUTCOME          three columns, each with a filled
                                        header bar, as in their examples
    CONCLUSION box, full width          as in their AJKD example

The design reference that drove this is submission/ga_ref_B.png.  Every number
in it came back invented or garbled, so the content here is the paper's:

  * capacity window      load_series("panasonic"), test cell
  * calibrated interval  checkpoints/quantile_calce_seed42.pt -> results/ga_uq_band.npz
  * physics tail         src/results/phys_figs.npz
  * multi-scale exchange tab:ablation (SP300/400/500, ten seeds each)
  * six-dataset ratios   tab:lit_all, verified by D:/Temp/verify_ga_ratios.py
  * memory vs context    tab:deploy formulas

Canvas 13.28 x 5.31 cm: Elsevier require the 2.5:1 proportion ("If you are
submitting a larger image, please use the same ratio (500 wide x 200 high)").

Usage: python src/make_graphical_abstract_v4.py
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
# and axes.grid=True at import, and matplotlib MERGES rcParams dicts, so
# importing it later silently re-enables both.
from make_figures import load_series   # noqa: E402


# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 5,
    "savefig.bbox": None,
    "savefig.pad_inches": 0,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "axes.grid": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 1.6,
    "ytick.major.size": 1.6,
})

NAVY = "#1B3A5C"        # title bar, column headers, method blocks
BLUE = "#0F4D92"        # the proposed method
BLUE_L = "#7FA8CE"
BLUE_FILL = "#DCE7F3"
BLUE_TINT = "#F1F5FA"
CORAL = "#D97757"       # the contrasting baseline / the datasets we trail
SAND = "#F6F1E4"        # conclusion box
SAGE = "#E9F0E4"        # claim strips
INK = "#272727"
GRAY = "#767676"
GRAY_L = "#D2D2D2"

W_CM, H_CM = 13.28, 5.31
FIG_W, FIG_H = W_CM / 2.54, H_CM / 2.54

TITLE_Y, TITLE_H = 0.902, 0.093
HDR_Y, HDR_H = 0.848, 0.044
BODY_Y0, BODY_Y1 = 0.135, 0.836
CONC_Y, CONC_H = 0.018, 0.100

MARGIN = 0.012
COL_X = (MARGIN, 0.402, 0.752)
COL_W = (0.370, 0.330, 0.236)
COL_NAMES = ("METHOD", "EVIDENCE", "OUTCOME")

fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=300)
fig.patch.set_facecolor("white")

_checks = []


def fit(text_artist, width, tag):
    _checks.append((text_artist, width, tag))
    return text_artist


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


def band(x, y, w, h, fc, ec="none", lw=0.0, z=1):
    fig.patches.append(Rectangle((x, y), w, h, linewidth=lw, facecolor=fc,
                                 edgecolor=ec, transform=fig.transFigure,
                                 figure=fig, zorder=z))


def txt(x, y, s, size=5, color=INK, weight="normal", ha="left", va="center",
        z=5, **kw):
    return fig.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                    ha=ha, va=va, zorder=z, **kw)


def strip_axes(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRAY)
    ax.tick_params(colors=GRAY, pad=1.0, labelsize=5, length=1.6)


def claim(col, y, s):
    """Pale strip carrying one claim, the device the reference uses per chart."""
    x0, w = COL_X[col], COL_W[col]
    band(x0 + 0.006, y, w - 0.012, 0.038, SAGE, z=2)
    t = txt(x0 + 0.016, y + 0.019, s, size=5, color=NAVY, weight="bold")
    fit(t, w - 0.032, "claim")


# ==========================================================================
# title bar and column headers
# ==========================================================================
band(MARGIN, TITLE_Y, 1.0 - 2 * MARGIN, TITLE_H, NAVY, z=2)
txt(MARGIN + 0.014, TITLE_Y + TITLE_H / 2, "DeltaCycle", size=9,
    color="white", weight="bold")
txt(MARGIN + 0.178, TITLE_Y + TITLE_H / 2,
    ":  multi-scale linear attention for calibrated on-device battery prognostics",
    size=7, color="#C9D8E8")

for i, name in enumerate(COL_NAMES):
    band(COL_X[i], HDR_Y, COL_W[i], HDR_H, NAVY, z=2)
    txt(COL_X[i] + 0.010, HDR_Y + HDR_H / 2, name, size=6.5, color="white",
        weight="bold")

# ==========================================================================
# COLUMN 1 -- METHOD
# ==========================================================================
x0, w = COL_X[0], COL_W[0]

# (a) the input: the one measured channel, a real 24-cycle window
caps, _tr, cell, _W, _sps, _eol = load_series("panasonic")
full = np.asarray(caps[cell], dtype=float)
d = np.diff(full)
rises = np.where(d > 0.004)[0]
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

txt(x0 + 0.008, 0.816, "capacity — the only input", size=5.5, weight="bold")
# no x label: matplotlib draws ticks and labels OUTSIDE the axes rect,
# and that 0.06 of height is what collided with the row below
axi = fig.add_axes([x0 + 0.008, 0.722, 0.170, 0.088])
axi.plot(np.arange(Z), seg, color=INK, lw=1.0, solid_capstyle="round")
axi.set_xticks([0, 8, 16, 24])
axi.set_xticklabels(["0", "8", "16", "24"])
axi.set_yticks([])
strip_axes(axi)

band(0.212, 0.722, w - 0.200, 0.088, SAGE, z=2)
txt(0.222, 0.770, "No voltage, current", size=5, color=NAVY)
txt(0.222, 0.742, "or temperature used.", size=5, color=NAVY)

# (b, c) three patch branches, each through its own GDN-2 pair
txt(x0 + 0.008, 0.668, "patch window (shared)", size=5, color=GRAY,
    weight="bold")
txt(x0 + 0.284, 0.668, "one query", size=5, color=BLUE, weight="bold",
    ha="center")
LANES = ((0.564, 2, BLUE_L), (0.474, 4, BLUE_L), (0.384, 8, BLUE_L))
sx0, sx1 = x0 + 0.042, x0 + 0.196
for yc, p, col in LANES:
    for s in range(0, Z, p):
        band(sx0 + (sx1 - sx0) * s / Z + 0.003, yc, (sx1 - sx0) * p / Z - 0.006,
             0.052, col, z=3)
    txt(x0 + 0.034, yc + 0.026, str(p), size=5, color=NAVY, weight="bold",
        ha="right")
    band(x0 + 0.214, yc, 0.140, 0.052, BLUE_TINT, ec=NAVY, lw=0.6, z=3)
    txt(x0 + 0.284, yc + 0.026, "GDN-2 ×2", size=5, color=NAVY, ha="center")
# the coarse branch's compressed state queries the two finer branches
# between the segment strip and the GDN box, so the head lands in the gap
# instead of on the box it is supposed to point past
for ya, yb in ((0.436, 0.468), (0.526, 0.558)):
    fig.patches.append(
        FancyArrowPatch((x0 + 0.205, ya), (x0 + 0.205, yb),
                        arrowstyle="-|>", mutation_scale=4.5, linewidth=0.7,
                        color=BLUE, transform=fig.transFigure, figure=fig,
                        zorder=6))

# (d) readout, then the three heads
band(x0 + 0.008, 0.302, w - 0.016, 0.040, BLUE_TINT, ec=NAVY, lw=0.6, z=3)
txt(x0 + w / 2 - 0.004, 0.322, "fused readout", size=5, color=NAVY,
    ha="center")
for a, b in ((0.384, 0.346), (0.302, 0.282)):
    fig.patches.append(
        FancyArrowPatch((x0 + 0.205, a), (x0 + 0.205, b), arrowstyle="-|>",
                        mutation_scale=4.5, linewidth=0.7, color=NAVY,
                        transform=fig.transFigure, figure=fig, zorder=6))
HEADS = ("capacity", "P2.5/P50/P97.5", "rate + IR")
hw = (w - 0.016 - 2 * 0.006) / 3
for i, hname in enumerate(HEADS):
    hx = x0 + 0.008 + i * (hw + 0.006)
    band(hx, 0.228, hw, 0.046, "white", ec=NAVY, lw=0.6, z=3)
    txt(hx + hw / 2, 0.251, hname, size=5, color=NAVY, ha="center")

txt(x0 + 0.008, 0.166,
    "implemented in C, runs on a Cortex-M3;", size=5, color=GRAY)
txt(x0 + 0.008, 0.140,
    "its state does not grow with the context length", size=5, color=GRAY)

# ==========================================================================
# COLUMN 2 -- EVIDENCE
# ==========================================================================
x0, w = COL_X[1], COL_W[1]
PZ = os.path.join(_SRC, "results", "phys_figs.npz")
if not os.path.exists(PZ):
    PZ = os.path.join(_ROOT, "results", "phys_figs.npz")

# (a) calibrated intervals, the real CQR band over the CALCE test cell
CACHE = os.path.join(_ROOT, "results", "ga_uq_band.npz")
_z = np.load(CACHE)
xb, lo_b, mid_b, hi_b, true_b, eol_b = (_z["x"], _z["lo"], _z["mid"],
                                        _z["hi"], _z["true"], float(_z["eol"]))

txt(x0 + 0.006, 0.822, "Calibrated intervals", size=5.5, weight="bold")
ax = fig.add_axes([x0 + 0.030, 0.684, w - 0.036, 0.108])
ax.fill_between(xb, lo_b, hi_b, color=BLUE_FILL, linewidth=0, zorder=1)
ax.plot(xb, true_b, color=INK, lw=0.7, zorder=4)
ax.plot(xb, mid_b, color=BLUE, lw=0.7, zorder=3)
miss = (true_b < lo_b) | (true_b > hi_b)
if miss.any():
    ax.plot(xb[miss], true_b[miss], "v", ms=1.0, color=CORAL, zorder=5,
            markeredgewidth=0)
ax.axhline(eol_b, color=CORAL, lw=0.6, ls=(0, (2.4, 1.6)), zorder=2)
ax.set_xlim(xb[0], xb[-1])
ax.set_ylim(min(lo_b.min(), true_b.min()) - 0.05,
            max(hi_b.max(), true_b.max()) + 0.10)
ax.set_xticks([xb[0], 400, 800])
ax.set_xticklabels([str(xb[0]), "400", "800"])
ax.set_yticks([0.2, 0.6, 1.0])
ax.set_ylabel("capacity", fontsize=5, color=GRAY, labelpad=1.0)
strip_axes(ax)
claim(1, 0.612, "93.5% coverage vs 95% nominal")

# (b) physics rate head on the unseen tail
txt(x0 + 0.006, 0.584, "Physics rate head", size=5.5, weight="bold")
z = np.load(PZ)
ex, tr = z["ext_x"].astype(float), z["ext_truth"].astype(float)
ax = fig.add_axes([x0 + 0.030, 0.446, w - 0.036, 0.108])
ax.plot(ex, z["ext_free"], color=CORAL, lw=0.8, label="free head", zorder=3)
ax.plot(ex, z["ext_rate"], color=BLUE, lw=1.3, label="+ physics", zorder=4)
ax.plot(ex, tr, color=INK, lw=0.7, zorder=5)
ax.axvline(ex[0], color=GRAY_L, lw=0.6, zorder=1)
ax.set_xlim(ex[0], ex[-1] + 3)
ax.set_ylim(min(z["ext_free"].min(), tr.min()) - 0.03,
            max(tr.max(), z["ext_last"].max()) + 0.02)
ax.set_xticks([ex[0], 840, ex[-1]])
ax.set_xticklabels([str(int(ex[0])), "840", str(int(ex[-1]))])
ax.set_yticks([])
ax.set_ylabel("capacity", fontsize=5, color=GRAY, labelpad=1.0)
ax.legend(loc="lower left", frameon=False, fontsize=5, handlelength=1.3,
          handletextpad=0.35, labelspacing=0.25, borderpad=0.1,
          borderaxespad=0.2)
strip_axes(ax)
claim(1, 0.374, "tail R² 0.37 → 0.77 with physics")

# (c) the exchange itself, against a parameter-matched control
txt(x0 + 0.006, 0.346, "Multi-scale exchange", size=5.5, weight="bold")
ax = fig.add_axes([x0 + 0.030, 0.208, w - 0.036, 0.108])
SP = ["SP300", "SP400", "SP500"]
NOEX = np.array([0.0040, 0.0038, 0.0042])
EXCH = np.array([0.0037, 0.0036, 0.0040])
NERR = np.array([0.0004, 0.0002, 0.0004])
EERR = np.array([0.0003, 0.0002, 0.0003])
yy = np.arange(3)[::-1]
for k in range(3):
    ax.plot([NOEX[k], EXCH[k]], [yy[k], yy[k]], color=GRAY_L, lw=1.6,
            solid_capstyle="round", zorder=2)
ax.errorbar(NOEX, yy, xerr=NERR, fmt="o", ms=3.0, color=CORAL,
            markeredgewidth=0, capsize=1.5, elinewidth=0.6, zorder=4,
            label="no exchange")
ax.errorbar(EXCH, yy, xerr=EERR, fmt="o", ms=3.0, color=BLUE,
            markeredgewidth=0, capsize=1.5, elinewidth=0.6, zorder=5,
            label="+ exchange")
ax.set_yticks(yy)
ax.set_yticklabels(SP)
ax.set_ylim(-0.7, 2.7)
ax.set_xlim(0.0031, 0.0056)
ax.set_xticks([0.0035, 0.0040, 0.0045])
ax.set_xticklabels(["0.0035", "0.0040", "0.0045"])
ax.tick_params(axis="y", length=0, pad=1.5)
ax.legend(loc="upper left", frameon=False, fontsize=5, handlelength=0.9,
          handletextpad=0.3, labelspacing=0.25, borderpad=0.1,
          borderaxespad=0.2)
strip_axes(ax)
claim(1, 0.136, "−5.7% mean MAE over 30 pairs (p = 0.014)")

# ==========================================================================
# COLUMN 3 -- OUTCOME
# ==========================================================================
x0, w = COL_X[2], COL_W[2]

txt(x0 + 0.006, 0.822, "Six public datasets", size=5.5, weight="bold")
DS = ["PANASONIC", "TJU", "GOTION", "MIT", "NASA", "CALCE"]
OURS = np.array([0.003767, 0.001500, 0.008033, 0.002533, 0.008900, 0.007867])
BEST = np.array([0.007733, 0.001967, 0.008733, 0.003200, 0.007767, 0.006933])
ratio = OURS / BEST
ax = fig.add_axes([x0 + 0.072, 0.558, w - 0.078, 0.240])
yy = np.arange(6)[::-1]
ax.barh(yy, ratio, height=0.46,
        color=[BLUE if r < 1 else CORAL for r in ratio], edgecolor="none",
        linewidth=0, zorder=3)
# drawn only in the gaps between bars: a full-height line runs
# through the value label of any bar that stops just short of 1.00
for _g in np.arange(0.5, 5.4, 1.0):
    ax.plot([1.0, 1.0], [_g - 0.25, _g + 0.25], color=INK, lw=0.6,
            ls=(0, (2.2, 1.6)), zorder=4)
for y_, r in zip(yy, ratio):
    ax.text(r + 0.03, y_, f"{r:.2f}", fontsize=5, va="center", ha="left",
            color=BLUE if r < 1 else CORAL, fontweight="bold")
ax.set_yticks(yy)
ax.set_yticklabels(DS)
ax.set_ylim(-0.7, 5.7)
ax.set_xlim(0, 1.62)
ax.set_xticks([0.5, 1.0])
ax.set_xticklabels(["0.5", "1.00"])
ax.tick_params(axis="y", length=0, pad=1.5)
strip_axes(ax)

txt(x0 + 0.006, 0.452, "Memory vs. context length", size=5.5, weight="bold")
TOK = np.logspace(np.log2(16), np.log2(4096), 60, base=2)
ax = fig.add_axes([x0 + 0.078, 0.238, w - 0.084, 0.190])
ax.plot(TOK, 4.0 * TOK ** 2 * 4.0 / 1024.0, color=CORAL, lw=0.8,
        label="attention scores")
ax.plot(TOK, 6.0 * 2.0 * TOK * 64.0 * 4.0 / 1024.0, color="#8FAF8A", lw=0.8,
        ls=(0, (3.0, 1.7)), label="KV cache")
ax.plot(TOK, np.full_like(TOK, 48.0), color=BLUE, lw=1.6,
        label="ours: state")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlim(13, 4200)
ax.set_ylim(0.05, 4e5)
ax.set_xticks([16, 256, 4096])
ax.set_xticklabels(["16", "256", "4096"])
ax.set_yticks([16, 4096, 65536])
ax.set_yticklabels(["16", "4096", "65536"])
ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.set_ylabel("memory (KB)", fontsize=5, color=GRAY, labelpad=1.0)
ax.legend(loc="lower right", frameon=False, fontsize=5, handlelength=1.2,
          handletextpad=0.3, labelspacing=0.25, borderpad=0.1,
          borderaxespad=0.2)
strip_axes(ax)
claim(2, 0.166, "flat in context length")

# ==========================================================================
# CONCLUSION
# ==========================================================================
band(MARGIN, CONC_Y, 1.0 - 2 * MARGIN, CONC_H, SAND, ec=NAVY, lw=0.7, z=2)
t = txt(0.5, CONC_Y + CONC_H / 2,
        "One capacity-only model, small enough to run inside a battery-management "
        "microcontroller, whose predictions stay\ncalibrated, physically "
        "consistent, and ahead of the closest baseline on four of six public "
        "datasets.", size=5.5, color=NAVY, weight="bold", ha="center",
        linespacing=1.5)
fit(t, 1.0 - 2 * MARGIN - 0.04, "conclusion")

# ==========================================================================
out = os.path.join(_ROOT, "submission", "ga_v4")
os.makedirs(os.path.dirname(out), exist_ok=True)
verify_fit()
plt.rcParams["savefig.bbox"] = None
fig.savefig(out + ".pdf")
fig.savefig(out + ".png", dpi=600)

from PIL import Image   # noqa: E402
Image.open(out + ".png").convert("RGB").resize((500, 200), Image.LANCZOS) \
     .save(os.path.join(os.path.dirname(out), "_ga_v4_preview500.png"))
print("wrote", out + ".pdf / .png and _ga_v4_preview500.png")
