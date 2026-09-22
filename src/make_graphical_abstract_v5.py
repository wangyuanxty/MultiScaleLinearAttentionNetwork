"""Graphical abstract for DeltaCycle (v5) -- Elsevier's own template structure.

The layout is cursor-based: every block advances a y cursor by its true
height, and an assert fires if a column overruns its body.  The rule this
encodes, which an earlier draft got wrong everywhere:

    matplotlib draws tick labels and axis labels OUTSIDE the axes rect.

So a chart occupies its axes height PLUS a tick band below and a label gutter
to the left.  Sizing charts by their axes rect alone silently overruns.

Structure follows Elsevier's published graphical abstracts -- their template
figure gives its middle column to methodology, and all three of their
published examples carry a method or cohort column:

    title bar
    METHOD | EVIDENCE | OUTCOME          three columns, filled header bars
    CONCLUSION box, full width

Content is the paper's:
  * capacity window      load_series("panasonic"), test cell
  * calibrated interval  results/ga_uq_band.npz (quantile_calce_seed42.pt)
  * physics tail         src/results/phys_figs.npz
  * multi-scale exchange tab:ablation, ten seeds per start point
  * six-dataset ratios   tab:lit_all, verified by D:/Temp/verify_ga_ratios.py
  * memory vs context    tab:deploy formulas

Canvas 13.28 x 5.31 cm -- Elsevier require the 2.5:1 proportion.
Usage: python src/make_graphical_abstract_v5.py
"""
import os
import sys

import io
import json

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

NAVY = "#1B3A5C"
BLUE = "#0F4D92"
BLUE_FILL = "#DCE7F3"
BLUE_TINT = "#F1F5FA"
CORAL = "#C8603C"
SAGE_GREEN = "#7C9A6D"
SAND = "#F6F1E4"
SAGE = "#E7EFE2"
INK = "#1A1A1A"
GRAY = "#3A3A3A"      # dark: grey text is unreadable here
GRAY_L = "#B8B8B8"

W_CM, H_CM = 13.28, 5.31
FIG_W, FIG_H = W_CM / 2.54, H_CM / 2.54

TITLE_Y, TITLE_H = 0.908, 0.076
HDR_Y, HDR_H = 0.856, 0.042
BODY_T, BODY_B = 0.844, 0.128
CONC_Y, CONC_H = 0.016, 0.096

MARGIN = 0.012
COL_X = (0.012, 0.402, 0.752)
COL_W = (0.370, 0.330, 0.236)

TICK_BAND = 0.036      # one 4.5 pt tick row.  No axis titles anywhere:
                       # they are drawn outside the axes and were the
                       # source of every overlap in this figure.
LINE_H = 0.030         # one 5.5 pt title row
CLAIM_H = 0.032

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


def txt(x, y, s, size=5, color=INK, weight="normal", ha="left", **kw):
    return fig.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                    ha=ha, va="center", zorder=6, **kw)


def arrow(x0, y0, x1, y1, color=NAVY, lw=0.7, ms=4.5, z=6, rad=0.0):
    fig.patches.append(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms, linewidth=lw,
        color=color, transform=fig.transFigure, figure=fig, zorder=z,
        connectionstyle=f"arc3,rad={rad}"))


def chart(x, y_top, w, h, gut_l=0.0):
    """Axes with room reserved for the ticks and labels drawn outside it.

    The caller must advance its cursor by h + TICK_BAND, not h.
    """
    return fig.add_axes([x + gut_l, y_top - h, w - gut_l, h])


def bare(ax, xticks=None, xlabels=None, yticks=None, ylabels=None,
         xlabel=None, ylabel=None):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRAY)
    ax.tick_params(colors=GRAY, pad=1.0, labelsize=4.5, length=1.6)
    if xticks is not None:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)
    if yticks is not None:
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks if ylabels is None else ylabels)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=4.5, color=GRAY, labelpad=1.0)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=4.5, color=GRAY, labelpad=1.0)


def title_row(col_x, y_top, text):
    """One item title row; returns the cursor below it."""
    txt(col_x + 0.004, y_top - LINE_H / 2, text, size=5.5, weight="bold")
    return y_top - LINE_H


def strip(col_x, col_w, y_top, text):
    """Pale strip carrying one result, as in Elsevier's published examples."""
    rect(col_x + 0.004, y_top - CLAIM_H, col_w - 0.008, CLAIM_H, SAGE, z=2)
    t = txt(col_x + 0.012, y_top - CLAIM_H / 2, text, size=4.5, color=NAVY,
            weight="bold")
    fit(t, col_w - 0.024, "claim")
    return y_top - CLAIM_H

# ==========================================================================
# title bar + column headers
# ==========================================================================
rect(MARGIN, TITLE_Y, 1.0 - 2 * MARGIN, TITLE_H, NAVY, z=2)
txt(MARGIN + 0.014, TITLE_Y + TITLE_H / 2, "DeltaCycle", size=9, color="white",
    weight="bold")
t = txt(MARGIN + 0.170, TITLE_Y + TITLE_H / 2,
        ":  one capacity-only model for calibrated on-device battery prognostics",
        size=7, color="#C9D8E8")
fit(t, 1.0 - MARGIN - 0.170 - 0.012, "title")
for i, name in enumerate(("METHOD", "EVIDENCE", "OUTCOME")):
    rect(COL_X[i], HDR_Y, COL_W[i], HDR_H, NAVY, z=2)
    txt(COL_X[i] + 0.010, HDR_Y + HDR_H / 2, name, size=6.5, color="white",
        weight="bold")

# ==========================================================================
# COLUMN 1 -- METHOD
#
# No patch-segment rectangles.  A reader cannot count 3 / 6 / 12 blocks at
# this size anyway, and they were the busiest thing on the page.  Three named
# branches carry the same information and leave room for a full y axis on the
# input chart.
# ==========================================================================
x0, w = COL_X[0], COL_W[0]
y = BODY_T

y = title_row(x0, y, "capacity (Ah) — the only input")

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

ax = chart(x0, y, w, 0.230, gut_l=0.048)
ax.plot(np.arange(Z), seg, color=INK, lw=1.0, solid_capstyle="round")
bare(ax, xticks=[0, 8, 16, 24], xlabels=["0", "8", "16", "24"],
     yticks=[round(float(seg.min()), 2), round(float(seg.max()), 2)], ylabel="capacity (Ah)")
y -= 0.230 + TICK_BAND + 0.012

BR_W, BR_GAP = (w - 0.016 - 2 * 0.010) / 3, 0.010
y -= 0.006
for i, p in enumerate((2, 4, 8)):
    bx = x0 + 0.008 + i * (BR_W + BR_GAP)
    rect(bx, y - 0.044, BR_W, 0.044, BLUE_FILL, ec=NAVY, lw=0.6, z=3)
    txt(bx + BR_W / 2, y - 0.022, f"patch {p}", size=5.5, color=NAVY,
        ha="center", weight="bold")
y -= 0.044 + 0.010

for i in range(3):
    bx = x0 + 0.008 + i * (BR_W + BR_GAP)
    arrow(bx + BR_W / 2, y + 0.008, bx + BR_W / 2, y - 0.004)
    rect(bx, y - 0.052, BR_W, 0.052, BLUE_TINT, ec=NAVY, lw=0.6, z=3)
    txt(bx + BR_W / 2, y - 0.026, "GDN-2 x2", size=5, color=NAVY, ha="center")
y -= 0.052 + 0.032

# the exchange reads as a bracket from the coarse branch back over the finer
# two, which is what the mechanism is: one query, generated by the coarse
# branch's state, that reads the fine and mid sequences
arrow(x0 + 0.008 + 2 * (BR_W + BR_GAP) + BR_W / 2, y + 0.014,
      x0 + 0.008 + BR_W / 2, y + 0.014, color=BLUE, rad=-0.30)
txt(x0 + w / 2, y + 0.020, "cross-scale exchange: one query", size=5,
    color=BLUE, ha="center", weight="bold")
y -= 0.016

rect(x0 + 0.008, y - 0.040, w - 0.016, 0.040, BLUE_TINT, ec=NAVY, lw=0.6, z=3)
txt(x0 + w / 2, y - 0.020, "fused readout", size=5, color=NAVY, ha="center")
y -= 0.040 + 0.012

heads = ("capacity", "P2.5/P50/P97.5", "rate + IR")
hw = (w - 0.016 - 2 * 0.010) / 3
for i, hname in enumerate(heads):
    hx = x0 + 0.008 + i * (hw + 0.010)
    arrow(hx + hw / 2, y + 0.004, hx + hw / 2, y - 0.004)
    rect(hx, y - 0.046, hw, 0.046, "white", ec=NAVY, lw=0.6, z=3)
    t = txt(hx + hw / 2, y - 0.023, hname, size=5, color=NAVY, ha="center")
    fit(t, hw - 0.004, f"head{i}")
y -= 0.046 + 0.020

txt(x0 + 0.004, y - LINE_H / 2,
    "runs in C on a Cortex-M3",
    size=5, color=GRAY)
y -= LINE_H
assert y >= BODY_B - 1e-9, f"METHOD overruns body by {BODY_B - y:.4f}"

# ==========================================================================
# COLUMN 2 -- EVIDENCE: title row, chart, pale strip carrying the result
# ==========================================================================
x0, w = COL_X[1], COL_W[1]
GUT = 0.034                      # y tick NUMBERS only, no ylabel
EV_H = 0.126
y = BODY_T


# (a) calibrated intervals
y = title_row(x0, y, "Calibrated intervals — capacity (Ah) vs cycle")
_z = np.load(os.path.join(_ROOT, "results", "ga_uq_band.npz"))
xb, lo_b, mid_b, hi_b = _z["x"], _z["lo"], _z["mid"], _z["hi"]
true_b, eol_b = _z["true"], float(_z["eol"])
ax = chart(x0, y, w, EV_H, gut_l=GUT)
ax.fill_between(xb, lo_b, hi_b, color=BLUE_FILL, linewidth=0, zorder=1)
ax.plot(xb, true_b, color=INK, lw=0.7, zorder=4)
ax.plot(xb, mid_b, color=BLUE, lw=0.7, zorder=3)
miss = (true_b < lo_b) | (true_b > hi_b)
if miss.any():
    ax.plot(xb[miss], true_b[miss], "v", ms=1.0, color=CORAL, zorder=5,
            markeredgewidth=0)
ax.axhline(eol_b, color=CORAL, lw=0.6, ls=(0, (2.4, 1.6)), zorder=2)
ax.set_xlim(xb[0], xb[-1])
ax.set_ylim(min(lo_b.min(), true_b.min()) - 0.005,
            max(hi_b.max(), true_b.max()) + 0.008)
bare(ax, xticks=[xb[0], 400, 800], xlabels=["64", "400", "800"],
     yticks=[0.2, 0.6, 1.0])
y -= EV_H + TICK_BAND
y = strip(x0, w, y, "93.5% coverage vs 95% nominal")
y -= 0.018

# (b) physics rate head on the unseen tail
y = title_row(x0, y, "Physics rate head — normalized capacity")
_z2 = np.load(os.path.join(_SRC, "results", "phys_figs.npz"))
ex = _z2["ext_x"].astype(float)
tr = _z2["ext_truth"].astype(float)
ax = chart(x0, y, w, EV_H, gut_l=GUT)
ax.plot(ex, _z2["ext_free"], color=CORAL, lw=0.8, zorder=3)
ax.plot(ex, _z2["ext_rate"], color=BLUE, lw=1.3, zorder=4)
ax.plot(ex, tr, color=INK, lw=0.7, zorder=5)
ax.set_xlim(ex[0], ex[-1])
ax.set_ylim(0.02, 0.46)
bare(ax, xticks=[ex[0], 840, ex[-1]], xlabels=["792", "840", "880"],
     yticks=[0.1, 0.3])
t = txt(x0 + w - 0.006, y - 0.010, "+ physics", size=4.5, color=BLUE,
        ha="right", weight="bold")
txt(x0 + w - 0.006, y - 0.026, "free head", size=4.5, color=CORAL, ha="right",
    weight="bold")
y -= EV_H + TICK_BAND
y = strip(x0, w, y, "tail R² 0.37 → 0.77 with physics")
y -= 0.018

# (c) The exchange is a statistical result, not a curve, and 30 dots at this
# size read as noise.  State it as a table -- the form Elsevier's own AJKD
# example uses for its headline numbers.
_pt = json.load(io.open(os.path.join(_SRC, "results", "per_sp_train.json"),
                        encoding="utf-8"))["panasonic"]
_pn = json.load(io.open(os.path.join(
    _SRC, "results", "per_sp_ablation_panasonic_multi_wide.json"),
    encoding="utf-8"))
_rows = []
_npos = 0
for _sp in ("300", "400", "500"):
    _x = np.array([v[0]["MAE"] for v in _pt[_sp].values()])
    _n = np.array([v[0]["MAE"] for v in _pn[_sp].values()])
    _npos += int((_n - _x > 0).sum())
    _rows.append((f"SP{_sp}", f"{_n.mean():.4f}", f"{_x.mean():.4f}",
                  f"{(1 - _x.mean() / _n.mean()) * 100:.1f}%"))
_ty = y - 0.004
_hdr = ("start", "no exchange", "+ exchange", "change")
_colw = (0.062, 0.098, 0.098, 0.062)
_cx = x0 + 0.006
for _i, (_h, _wd) in enumerate(zip(_hdr, _colw)):
    _cx += _wd
    txt(_cx, _ty, _h, size=4.5, color=NAVY, ha="right", weight="bold")
rect(x0 + 0.006, _ty - 0.010, sum(_colw), 0.003, INK, z=3)
for _r, _row in enumerate(_rows):
    _ry = _ty - 0.026 - _r * 0.024
    _cx = x0 + 0.006
    for _i, (_v, _wd) in enumerate(zip(_row, _colw)):
        _cx += _wd
        txt(_cx, _ry, _v, size=4.5, color=INK if _i < 3 else BLUE,
            ha="right", weight="bold" if _i == 3 else "normal")
y = _ty - 0.026 - 2 * 0.024 - 0.014
y = strip(x0, w, y, "%d of 30 paired runs improve; mean −5.7%% (p = 0.014)"
          % _npos)
assert y >= BODY_B - 1e-9, f"EVIDENCE overruns body by {BODY_B - y:.4f}"

# ==========================================================================
# COLUMN 3 -- OUTCOME
# ==========================================================================
x0, w = COL_X[2], COL_W[2]
y = BODY_T
GUT_BAR = 0.066                 # room for the dataset names
GUT_MEM = 0.046                 # room for the KB ticks and the ylabel

y = title_row(x0, y, "Six public datasets")
ax = chart(x0, y, w, 0.230, gut_l=GUT_BAR)
DS = ["PANASONIC", "TJU", "GOTION", "MIT", "NASA", "CALCE"]
OURS = np.array([0.003767, 0.001500, 0.008033, 0.002533, 0.008900, 0.007867])
BEST = np.array([0.007733, 0.001967, 0.008733, 0.003200, 0.007767, 0.006933])
ratio = OURS / BEST
yy = np.arange(6)[::-1].astype(float)
# Two bars per dataset.  A single ratio bar is not a comparison -- this shows
# ours against the baseline on every dataset, with the baseline at 1.00.
ax.barh(yy + 0.19, np.ones(6), height=0.34, color=GRAY_L, edgecolor="none",
        linewidth=0, zorder=3)
ax.barh(yy - 0.19, ratio, height=0.34,
        color=[BLUE if r < 1 else CORAL for r in ratio], edgecolor="none",
        linewidth=0, zorder=4)
ax.axvline(1.0, color=INK, lw=0.6, ls=(0, (2.2, 1.6)), zorder=5)
ax.set_xlim(0, 1.30)
ax.set_ylim(-0.78, 5.78)
bare(ax, xticks=[0.5, 1.0], xlabels=["0.5", "1.00"], yticks=yy, ylabels=DS)
ax.tick_params(axis="y", length=0, pad=1.5)
y -= 0.230 + TICK_BAND
y = strip(x0, w, y, "first on four, second on two")
y -= 0.022

y = title_row(x0, y, "Memory vs. context length")
TOK = np.logspace(np.log2(16), np.log2(4096), 60, base=2)
ax = chart(x0, y, w, 0.230, gut_l=GUT_MEM)
ax.plot(TOK, 4.0 * TOK ** 2 * 4.0 / 1024.0, color=CORAL, lw=0.8, zorder=3)
ax.plot(TOK, 6.0 * 2.0 * TOK * 64.0 * 4.0 / 1024.0, color=SAGE_GREEN, lw=0.8,
        ls=(0, (3.0, 1.7)), zorder=3)
ax.plot(TOK, np.full_like(TOK, 48.0), color=BLUE, lw=1.6, zorder=4)
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlim(13, 4200)
ax.set_ylim(0.4, 4e5)
ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
bare(ax, xticks=[16, 256, 4096], xlabels=["16", "256", "4096"],
     yticks=[16, 4096, 65536], ylabels=["16", "4096", "65536"], ylabel="memory (KB)")
# all three curves sit above y = 100 KB for x > 32, so the band under the flat
# state line is the one strip that is clear across the panel
for _i, (_t, _c, _b) in enumerate((("scores", CORAL, False),
                                   ("KV cache", SAGE_GREEN, False),
                                   ("ours", BLUE, True))):
    txt(x0 + 0.058, y - 0.048 - _i * 0.026, _t, size=5, color=_c,
        weight="bold" if _b else "normal")
y -= 0.230 + TICK_BAND
assert y >= BODY_B - 1e-9, f"OUTCOME overruns body by {BODY_B - y:.4f}"

# ==========================================================================
# CONCLUSION
# ==========================================================================
rect(MARGIN, CONC_Y, 1.0 - 2 * MARGIN, CONC_H, SAND, ec=NAVY, lw=0.7, z=2)
t = txt(0.5, CONC_Y + CONC_H / 2,
        "One capacity-only model, small enough to run inside a battery-management "
        "microcontroller, whose predictions stay\ncalibrated, physically "
        "consistent, and ahead of the closest baseline on four of six public "
        "datasets.", size=5.5, color=NAVY, weight="bold", ha="center",
        linespacing=1.5)
fit(t, 1.0 - 2 * MARGIN - 0.04, "conclusion")

# ==========================================================================
out = os.path.join(_ROOT, "submission", "ga_v5")
os.makedirs(os.path.dirname(out), exist_ok=True)
verify_fit()
plt.rcParams["savefig.bbox"] = None
fig.savefig(out + ".pdf")
fig.savefig(out + ".png", dpi=600)

from PIL import Image   # noqa: E402
Image.open(out + ".png").convert("RGB").resize((500, 200), Image.LANCZOS) \
     .save(os.path.join(os.path.dirname(out), "_ga_v5_preview500.png"))
print("wrote", out + ".pdf / .png and _ga_v5_preview500.png")
y = strip(x0, w, y, "memory stays flat")
y -= CLAIM_H
