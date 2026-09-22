"""Graphical abstract for DeltaCycle (v3) -- organised by the paper's highlights.

Two earlier drafts failed the same way: v1 was a grid of text boxes, which a
graphical abstract should never be, and v2 filled five panels with real plots
but still read as a card grid with a lone bar chart under it.  v3 keeps the
highlight order and every panel's real data, and fixes the composition:

  * the bottom strip is no longer a bar chart of ratios -- six thin bars spread
    over 13 cm was the ugliest element on the page.  It is now six sparklines
    of the actual test-cell trajectories, measured against predicted, one per
    dataset, so the widest part of the canvas carries the most data instead of
    the least;
  * panel 1 gets filled areas rather than three bare lines, so the deployment
    argument reads as a shape;
  * panel 2 loses its dead margins and the wasted band under the patch rows.

Canvas is 13.28 x 5.31 cm.  The 2.5:1 proportion is not a choice: Elsevier
require it ("If you are submitting a larger image, please use the same ratio
(500 wide x 200 high)"; the image is scaled to fit a 500x200 window).  Making
the figure feel less like a letterbox therefore has to come from composition,
not from the canvas.

Data provenance, all read-only:
  * panel 1           tab:deploy formulas (working memory against context)
  * panels 2, 5       make_figures.load_series / predict_series (real cells)
  * panel 3           checkpoints/quantile_calce_seed42.pt -> results/ga_uq_band.npz
  * panel 4           src/results/phys_figs.npz from make_figures_phys.py
  * panel 5 ratios    tab:lit_all, checked cell by cell by D:/Temp/verify_ga_ratios.py

Usage: python src/make_graphical_abstract_v3.py
"""
import os
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle

_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
sys.path.insert(0, _SRC)

# Imported for its loaders, and it MUST come before the rcParams block below:
# make_figures.py sets savefig.bbox='tight' and axes.grid=True at import time,
# and matplotlib MERGES rcParams dicts.  Importing it afterwards silently
# re-enables both -- the tight bbox stretched an earlier draft to 3.6:1, and
# the grid drew a stroked line through every label the collision audit checks.
from make_figures import load_series, predict_series   # noqa: E402


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

BLUE = "#0F4D92"        # the proposed method
BLUE_MID = "#5B8FC7"
BLUE_FILL = "#DCE7F3"
BLUE_TINT = "#F1F5FA"
RED = "#B64342"
RED_FILL = "#F4DEDC"
INK = "#272727"
GRAY = "#767676"
GRAY_FILL = "#E9E9E9"
GRAY_L = "#D2D2D2"

W_CM, H_CM = 13.28, 5.31
FIG_W, FIG_H = W_CM / 2.54, H_CM / 2.54

# Vertical zones, in figure fractions.  There is no sub-line row: the design
# reference (D:/Temp/ga_ref/ga_design_ref.png) carries one bold takeaway per
# column and nothing else, and dropping the grey sub-line bought 0.05 of chart
# height.  The qualifiers that row used to hold move to the figure caption.
HEADER_Y, RULE_Y, P_TITLE_Y = 0.966, 0.930, 0.900
AX_B, AX_T = 0.474, 0.870
HEAD_Y = 0.334
BAND_B, BAND_T, BAND_TITLE_Y = 0.008, 0.282, 0.258
SPARK_T, SPARK_B, SPARK_NAME_Y, SPARK_RATIO_Y = 0.190, 0.076, 0.210, 0.050

MARGIN, GUTTER = 0.012, 0.024
PW = (1.0 - 2 * MARGIN - 3 * GUTTER) / 4.0
PANEL_L = [MARGIN + i * (PW + GUTTER) for i in range(4)]

SPARK_GAP = 0.012
SPARK_W = (1.0 - 2 * MARGIN - 5 * SPARK_GAP) / 6.0
SPARK_L = [MARGIN + i * (SPARK_W + SPARK_GAP) for i in range(6)]

fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=300)
fig.patch.set_facecolor("white")

_checks = []


def fit(text_artist, width, tag):
    """Queue a text artist whose rendered width must stay inside `width`."""
    _checks.append((text_artist, width, tag))
    return text_artist


def verify_fit():
    """Report any string rendered wider than the box it was placed in."""
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


def clean(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRAY)
    ax.tick_params(colors=GRAY, pad=1.0, labelsize=5, length=1.6)


def chip(x, y, n, color=BLUE):
    """Numbered disc.  The reference uses circles, and a circle in figure
    fractions is an ellipse unless the 2.5:1 aspect is divided out.  r is set
    by the 5 pt glyph floor: any smaller and the disc is all digit."""
    r = 0.0230
    fig.patches.append(
        Ellipse((x, y), 2 * r * FIG_H / FIG_W, 2 * r, linewidth=0,
                facecolor=color, transform=fig.transFigure, figure=fig,
                zorder=6))
    fig.text(x, y - 0.0015, n, ha="center", va="center", fontsize=5,
             fontweight="bold", color="white", zorder=7)


def panel_head(i, n, title, head):
    x = PANEL_L[i]
    chip(x + 0.007, P_TITLE_Y, n)
    t = fig.text(x + 0.028, P_TITLE_Y, title, ha="left", va="center",
                 fontsize=6, fontweight="bold", color=INK)
    fit(t, PW - 0.028, f"title{i}")
    t = fig.text(x, HEAD_Y, head, ha="left", va="center", fontsize=7.5,
                 fontweight="bold", color=BLUE)
    fit(t, PW, f"head{i}")


# ==========================================================================
# header
# ==========================================================================
fig.text(MARGIN, HEADER_Y, "DeltaCycle", ha="left", va="center",
         fontsize=9.5, fontweight="bold", color=BLUE)
t = fig.text(MARGIN + 0.142, HEADER_Y,
             "multi-scale linear attention for on-device battery prognostics",
             ha="left", va="center", fontsize=6.5, color=INK)
fit(t, 1.0 - MARGIN - 0.142 - 0.02, "header")
fig.patches.append(Rectangle((MARGIN, RULE_Y), 1.0 - 2 * MARGIN, 0.0032,
                             linewidth=0, facecolor=GRAY_L,
                             transform=fig.transFigure, figure=fig, zorder=2))

# ==========================================================================
# 1 -- edge deployment: filled areas, so the fan-out is a shape not three lines
# ==========================================================================
ax = fig.add_axes([PANEL_L[0] + 0.044, AX_B, PW - 0.044, AX_T - AX_B])
TOK = np.logspace(np.log2(16), np.log2(4096), 80, base=2)
SCORES = 4.0 * TOK ** 2 * 4.0 / 1024.0      # H*L^2*4 B, peak of one layer
KV = 6.0 * 2.0 * TOK * 64.0 * 4.0 / 1024.0  # six layers, resident
STATE = 48.0                                # six-layer recurrent state

ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlim(13, 4200)
ax.set_ylim(0.05, 4e5)
ax.fill_between(TOK, 0.05, SCORES, color=RED_FILL, linewidth=0, zorder=1)
ax.fill_between(TOK, 0.05, KV, color=GRAY_FILL, linewidth=0, zorder=2)
ax.fill_between(TOK, 0.05, np.full_like(TOK, STATE), color=BLUE_FILL,
                linewidth=0, zorder=3)
ax.plot(TOK, SCORES, color=RED, lw=0.9, zorder=4, label="scores")
ax.plot(TOK, KV, color=GRAY, lw=0.9, ls=(0, (3.2, 1.8)), zorder=4,
        label="KV cache")
ax.plot(TOK, np.full_like(TOK, STATE), color=BLUE, lw=1.8, zorder=5,
        label="ours: state")
ax.axvline(32, color=GRAY_L, lw=0.6, zorder=6)
ax.plot([32], [STATE], "o", ms=2.4, mfc="white", mec=BLUE, mew=0.9, zorder=7)
ax.set_xticks([16, 128, 1024, 4096])
ax.set_xticklabels(["16", "128", "1024", "4096"])
ax.set_yticks([16, 256, 4096, 65536])
ax.set_yticklabels(["16 KB", "256", "4096", "65536"])
ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.set_xlabel("context length L (tokens)", fontsize=5, color=GRAY, labelpad=1.2)
# short labels: a wide key floats in the middle of the panel; a narrow one
# sits under the flat curve where there is genuinely no data
ax.legend(loc="lower right", frameon=False, fontsize=5, handlelength=1.3,
          handletextpad=0.35, labelspacing=0.25, borderpad=0.1,
          borderaxespad=0.3)
clean(ax)

panel_head(0, "1", "Edge deployment", "runs C on Cortex-M3")

# ==========================================================================
# 2 -- the method itself.  Elsevier's own graphical-abstract template gives
# its middle panel to methodology ("Here is where you showcase your
# methodology"), and all three of their published examples carry a method or
# cohort column.  An earlier draft of this figure had five evidence panels and
# no picture of the model at all; this panel is that picture, and the
# cross-scale claim rides on it as the headline.
# ==========================================================================
ax = fig.add_axes([PANEL_L[1], AX_B, PW, AX_T - AX_B])
caps, _tr, cell, _W, _sps, _eol = load_series("panasonic")
full = np.asarray(caps[cell], dtype=float)
d = np.diff(full)
rises = np.where(d > 0.004)[0]
rises = rises[(rises > 30) & (rises < len(full) - 30)]
c0 = len(full) // 2
if len(rises):
    best, best_n = rises[0], 0
    for c in rises:
        n = int(np.sum(np.abs(rises - c) < 30))
        if n > best_n:
            best, best_n = c, n
    c0 = int(best)
Z = 24
z0 = max(0, min(c0 - Z // 2, len(full) - Z))
seg = full[z0:z0 + Z]

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xticks([])
ax.set_yticks([])
for side in ("top", "right", "left", "bottom"):
    ax.spines[side].set_visible(False)

# --- input: the real 24-cycle window the branches read ---------------------
lo_s, hi_s = seg.min(), seg.max()
pad = 0.14 * (hi_s - lo_s + 1e-9)
y_in = 0.845 + 0.145 * ((seg - (lo_s - pad)) / ((hi_s + pad) - (lo_s - pad)))
ax.plot(np.linspace(0.06, 0.94, Z), y_in, color=INK, lw=1.0,
        solid_capstyle="round", zorder=4, clip_on=False)
ax.text(0.03, 0.985, "capacity", fontsize=5, color=GRAY, ha="left", va="top")
ax.annotate("", xy=(0.5, 0.812), xytext=(0.5, 0.836), zorder=5,
            arrowprops=dict(arrowstyle="-|>", lw=0.7, color=GRAY,
                            mutation_scale=4.5, shrinkA=0, shrinkB=0))

# --- three patch branches, each ending in its own GDN-2 stack --------------
LANES = ((0.735, 8, BLUE), (0.575, 4, BLUE_MID), (0.415, 2, "#A8C6E2"))
SX0, SX1 = 0.155, 0.665
for yc, p, col in LANES:
    for s in range(0, Z, p):
        ax.add_patch(Rectangle((SX0 + (SX1 - SX0) * s / Z + 0.004, yc - 0.036),
                               (SX1 - SX0) * p / Z - 0.008, 0.072,
                               facecolor=col, edgecolor="none", zorder=3))
    ax.text(0.135, yc, str(p), fontsize=5, color=col, fontweight="bold",
            ha="right", va="center")
    ax.add_patch(Rectangle((0.700, yc - 0.048), 0.180, 0.096,
                           facecolor=BLUE_TINT, edgecolor=col, linewidth=0.6,
                           zorder=3))
    ax.text(0.790, yc, "GDN-2", fontsize=5, color=col, ha="center",
            va="center", zorder=4)

# --- the exchange: the coarse branch's state queries the finer branches ----
ax.annotate("", xy=(0.945, 0.400), xytext=(0.945, 0.750), zorder=6,
            arrowprops=dict(arrowstyle="-|>", lw=0.9, color=BLUE,
                            mutation_scale=5.0, shrinkA=0, shrinkB=0))
ax.text(0.900, 0.575, "query", fontsize=5, color=BLUE, ha="center",
        va="center", rotation=90, zorder=6)

# --- what the branches feed ------------------------------------------------
ax.add_patch(Rectangle((0.135, 0.225), 0.745, 0.052, facecolor=BLUE_TINT,
                       edgecolor=BLUE, linewidth=0.6, zorder=3))
ax.text(0.5075, 0.251, "readout", fontsize=5, color=BLUE, ha="center",
        va="center", zorder=4)
ax.annotate("", xy=(0.5075, 0.281), xytext=(0.5075, 0.366), zorder=5,
            arrowprops=dict(arrowstyle="-|>", lw=0.7, color=GRAY,
                            mutation_scale=4.5, shrinkA=0, shrinkB=0))
ax.text(0.135, 0.130, "capacity  ·  P2.5 / P50 / P97.5  ·  rate",
        fontsize=5, color=GRAY, ha="left", va="center")
ax.text(0.97, 0.985, "matched parameters", fontsize=5, color=GRAY, ha="right",
        va="top")

panel_head(1, "2", "Cross-scale method", "−5.7% MAE (p=0.014)")

# ==========================================================================
# 3 -- calibrated uncertainty in one forward pass
# ==========================================================================
CACHE = os.path.join(_ROOT, "results", "ga_uq_band.npz")


def uq_band():
    """CQR band over the CALCE test cell; computed once, then cached."""
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        return (z["x"], z["lo"], z["mid"], z["hi"], z["true"], float(z["eol"]))

    import torch
    from gdn_model import build_gdn_model
    from train_per_sp import window_std

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(_ROOT, "checkpoints", "quantile_calce_seed42.pt"),
                    map_location=dev, weights_only=False)
    cfg = ck["config"]
    model = build_gdn_model(
        multiscale=cfg["multiscale"], stage_query=cfg["stage_query"],
        input_dim=cfg["input_dim"], window_size=cfg["window_size"],
        output_len=cfg["output_len"], num_quantiles=cfg["num_quantiles"],
        readout=cfg["readout"]).to(dev)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    c_lo, c_hi = ck["lo"], ck["hi"]
    caps, _t, c_cell, Wc, _s, eol = load_series("calce")
    tc = (np.asarray(caps[c_cell], dtype=float) - c_lo) / (c_hi - c_lo + 1e-6)
    qs = np.empty((len(tc) - Wc, 3))
    with torch.no_grad():
        for k, i in enumerate(range(Wc, len(tc))):
            win = tc[i - Wc:i]
            cin = torch.tensor(win[:, None], dtype=torch.float32).unsqueeze(0).to(dev)
            q = model(cin).cpu().numpy().squeeze()
            qs[k] = q * (window_std(win) + 1e-6) + float(win.mean())
    q_adj = 0.0052                      # CQR constant, CS2_36, n=869
    span = c_hi - c_lo + 1e-6
    out = (np.arange(Wc, len(tc)),
           (qs[:, 0] - q_adj) * span + c_lo,
           qs[:, 1] * span + c_lo,
           (qs[:, 2] + q_adj) * span + c_lo,
           tc[Wc:] * span + c_lo,
           eol)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, x=out[0], lo=out[1], mid=out[2], hi=out[3],
             true=out[4], eol=out[5])
    return out


ax = fig.add_axes([PANEL_L[2], AX_B, PW, AX_T - AX_B])
xb, lo_b, mid_b, hi_b, true_b, eol_b = uq_band()
ax.fill_between(xb, lo_b, hi_b, color=BLUE_FILL, linewidth=0, zorder=1)
ax.plot(xb, true_b, color=INK, lw=0.8, zorder=4)
ax.plot(xb, mid_b, color=BLUE, lw=0.8, zorder=3)
miss = (true_b < lo_b) | (true_b > hi_b)
if miss.any():
    ax.plot(xb[miss], true_b[miss], "v", ms=1.1, color=RED, zorder=5,
            markeredgewidth=0)
ax.axhline(eol_b, color=RED, lw=0.7, ls=(0, (2.4, 1.6)), zorder=2)
ax.text(xb[0] + 10, eol_b + 0.011, "EOL", fontsize=5, color=RED, ha="left",
        va="bottom")
# the band hugs the curve high up and falls away only near EOL, so the upper
# right -- empty once the trajectory drops -- is where the key belongs
# The band's bounding box spans nearly the whole panel, so a key placed
# anywhere inside it straddles that box. Clear its top edge instead: the
# trajectory has already fallen away at the top right, so the label reads as
# a panel subtitle over empty space.
ax.text(0.97, 0.91, "P2.5 / P50 / P97.5", transform=ax.transAxes, fontsize=5,
        color=BLUE, ha="right", va="bottom")
ax.set_xlim(xb[0], xb[-1])
ax.set_ylim(min(lo_b.min(), true_b.min()) - 0.045,
            max(hi_b.max(), true_b.max()) + 0.095)
ax.set_yticks([])
ax.set_xticks([xb[0], 400, 800])
ax.set_xticklabels([str(xb[0]), "400", "800"])
ax.set_xlabel("cycle", fontsize=5, color=GRAY, labelpad=1.2)
clean(ax)

panel_head(2, "3", "Calibrated intervals", "93.5% at 95% nominal")

# ==========================================================================
# 4 -- physics-consistent degradation-rate head
# ==========================================================================
ax = fig.add_axes([PANEL_L[3], AX_B, PW, AX_T - AX_B])
PZ = os.path.join(_SRC, "results", "phys_figs.npz")
if not os.path.exists(PZ):
    PZ = os.path.join(_ROOT, "results", "phys_figs.npz")
z = np.load(PZ)
ex, tr = z["ext_x"].astype(float), z["ext_truth"].astype(float)
free, rate, last = z["ext_free"], z["ext_rate"], z["ext_last"]

ax.plot(ex, free, color=RED, lw=0.9, label="free head", zorder=3)
ax.plot(ex, rate, color=BLUE, lw=1.5, label="rate head", zorder=4)
ax.plot(ex, tr, color=INK, lw=0.8, zorder=5)
ax.axvline(ex[0], color=GRAY_L, lw=0.7, zorder=1)
ax.text(ex[0] + 4, 0.435, "training data ends", fontsize=5, color=GRAY,
        ha="left", va="top")
# the curves converge at the right edge, so right-edge labels collide; two
# entries is the most that clears the regeneration spike near x=855
ax.legend(loc="upper right", frameon=False, fontsize=5, handlelength=1.3,
          handletextpad=0.35, labelspacing=0.30, borderpad=0.1,
          borderaxespad=0.3)
ax.set_xlim(ex[0], ex[-1] + 3)
ax.set_ylim(min(free.min(), tr.min()) - 0.030,
            max(tr.max(), last.max()) + 0.060)
ax.set_yticks([])
ax.set_xticks([ex[0], 840, ex[-1]])
ax.set_xticklabels([str(int(ex[0])), "840", str(int(ex[-1]))])
ax.set_xlabel("cycle, final 10% of life (measured)", fontsize=5, color=GRAY,
              labelpad=1.2)
clean(ax)

panel_head(3, "4", "Physics rate head", "tail R² 0.37 → 0.77")

# ==========================================================================
# 5 -- six datasets, as six real test-cell trajectories
# ==========================================================================
DS = ["PANASONIC", "TJU", "GOTION", "MIT", "NASA", "CALCE"]
OURS = np.array([0.003767, 0.001500, 0.008033, 0.002533, 0.008900, 0.007867])
BEST = np.array([0.007733, 0.001967, 0.008733, 0.003200, 0.007767, 0.006933])
ratio = OURS / BEST

TRAJ_CACHE = os.path.join(_ROOT, "results", "ga_six_traj.npz")


def six_traj():
    """Measured against predicted over each dataset's test cell; cached."""
    if os.path.exists(TRAJ_CACHE):
        z = np.load(TRAJ_CACHE)
        return {k: (z[k + "_true"], z[k + "_pred"]) for k in DS}

    out = {}
    for k in DS:
        pv, tv, _lo, _hi, _W, _s, _e = predict_series(k.lower(), K=1)
        out[k] = (tv[:, 0] if tv.ndim > 1 else tv, pv[:, 0])
    np.savez(TRAJ_CACHE, **{f"{k}_{s}": out[k][i] for k in DS
                            for i, s in enumerate(("true", "pred"))})
    return out


traj = six_traj()

fig.patches.append(Rectangle((MARGIN, BAND_B), 1.0 - 2 * MARGIN,
                             BAND_T - BAND_B, linewidth=0, facecolor=BLUE_TINT,
                             transform=fig.transFigure, figure=fig, zorder=-1))
chip(MARGIN + 0.017, BAND_TITLE_Y, "5")
t = fig.text(MARGIN + 0.038, BAND_TITLE_Y, "Six datasets, test cell",
             fontsize=6, fontweight="bold", color=INK, ha="left", va="center")
fit(t, 0.18, "bandtitle")
t = fig.text(MARGIN + 0.206, BAND_TITLE_Y,
             "grey = measured, blue = predicted; number = MAE / baseline",
             fontsize=5, color=GRAY, ha="left", va="center")
fit(t, 0.44, "bandsub")
t = fig.text(1.0 - MARGIN, BAND_TITLE_Y,
             "first on four, second on the other two", fontsize=6.5,
             fontweight="bold", color=BLUE, ha="right", va="center")
fit(t, 0.33, "bandright")

for i, k in enumerate(DS):
    x0 = SPARK_L[i]
    axt = fig.add_axes([x0 + 0.005, SPARK_B, SPARK_W - 0.010,
                        SPARK_T - SPARK_B])
    tv, pv = traj[k]
    # measured is drawn WIDE and pale, predicted thin and on top: the two
    # nearly coincide, so an equal-weight pair of lines just shows the blue
    # one and the key's "grey = measured" becomes unverifiable
    axt.plot(np.arange(len(tv)), tv, color="#C4C4C4", lw=1.7, zorder=2,
             solid_capstyle="round")
    axt.plot(np.arange(len(tv) - len(pv), len(tv)), pv, color=BLUE, lw=0.7,
             zorder=3)
    axt.set_xlim(0, len(tv))
    axt.set_ylim(-0.12, 1.04)
    axt.set_xticks([])
    axt.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        axt.spines[side].set_visible(False)
    win = ratio[i] < 1.0
    col = BLUE if win else RED
    t = fig.text(x0 + 0.005, SPARK_NAME_Y, k, fontsize=6, fontweight="bold",
                 color=INK, ha="left", va="center")
    fit(t, SPARK_W - 0.020, f"ds{i}")
    fig.text(x0 + 0.005, SPARK_RATIO_Y, f"{ratio[i]:.2f}", fontsize=5.5,
             fontweight="bold", color=col, ha="left", va="center")
    # the underline carries the verdict: blue = first on that dataset,
    # red = second, which is what the band's title sentence claims
    fig.patches.append(Rectangle((x0 + 0.005, BAND_B + 0.004),
                                 SPARK_W - 0.010, 0.0065, linewidth=0,
                                 facecolor=col, transform=fig.transFigure,
                                 figure=fig, zorder=4))

# ==========================================================================
out = os.path.join(_ROOT, "submission", "ga_v3")
os.makedirs(os.path.dirname(out), exist_ok=True)
verify_fit()
plt.rcParams["savefig.bbox"] = None      # re-assert: nothing may re-enable it
fig.savefig(out + ".pdf")
fig.savefig(out + ".png", dpi=600)

from PIL import Image   # noqa: E402
Image.open(out + ".png").convert("RGB").resize((500, 200), Image.LANCZOS) \
     .save(os.path.join(os.path.dirname(out), "_ga_v3_preview500.png"))
print("wrote", out + ".pdf / .png and _ga_v3_preview500.png")
