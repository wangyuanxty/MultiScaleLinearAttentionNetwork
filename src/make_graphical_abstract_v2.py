"""Graphical abstract for DeltaCycle (v2) -- organised by the paper's highlights.

The v1 draft was a grid of text boxes and numbers.  A graphical abstract made
of text and numbers is a table with rounded corners, so this version inverts
it: every one of the five panels is a plot carrying real experimental data,
and the reading order is the order of the highlights in paper/highlights.txt.

    1  Edge deployment         working memory against context length: the
                               recurrent state is the only flat curve
    2  Cross-scale exchange    a real capacity window with the 2/4/8 patch
                               grids drawn at the boundaries they actually have
    3  Calibrated uncertainty  the CQR-calibrated P2.5/P50/P97.5 band over the
                               whole CALCE test cell
    4  Physics rate head       the unseen-tail extrapolation, rate head against
                               a free head on identical inputs
    5  Six-dataset scorecard   per-SP MAE relative to the closest baseline

Data provenance, read-only:
  * panels 1, 5   tab:deploy and tab:lit_all (verified: D:/Temp/verify_ga_ratios.py)
  * panel 2       make_figures.load_series, PANASONIC test cell
  * panel 3       checkpoints/quantile_calce_seed42.pt, cached to
                  results/ga_uq_band.npz on the first run
  * panel 4       src/results/phys_figs.npz (written by make_figures_phys.py;
                  nothing is retrained here)

Canvas 13.28 x 5.31 cm, Elsevier's 2.5:1 graphical-abstract proportion.  The
submission artifact is vector PDF, so no dpi ceiling applies; the PNG is
rasterised at 600 dpi (3137 x 1254 px, above the 1328 x 531 minimum).

Usage: python src/make_graphical_abstract_v2.py
"""
import os
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
sys.path.insert(0, _SRC)

# Imported for its data loaders.  This MUST happen before the rcParams block
# below: make_figures.py sets savefig.bbox='tight' at import time and
# matplotlib MERGES rcParams dicts, so importing it afterwards silently
# re-enables the tight bbox -- which with mathtext on matplotlib 3.11 yields a
# broken canvas (the first draft of this figure came out 3.6:1, not 2.5:1).
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
    # make_figures.py also sets axes.grid=True at import; a gridline is a
    # stroked path and the collision audit counts one crossing every label it
    # passes through, which is every label on a bar chart
    "axes.grid": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 1.6,
    "ytick.major.size": 1.6,
})

BLUE = "#0F4D92"        # the proposed method
BLUE_MID = "#3775BA"
BLUE_PALE = "#C9DAEE"   # opaque fills only: alpha forces a PDF /ExtGState
RED = "#B64342"         # contrast, and the one dataset we trail
INK = "#272727"
GRAY = "#767676"
GRAY_L = "#D2D2D2"
BAND_BG = "#F2F6FB"

W_CM, H_CM = 13.28, 5.31
FIG_W, FIG_H = W_CM / 2.54, H_CM / 2.54

# Vertical zones, in figure fractions.
HEADER_Y, RULE_Y, P_TITLE_Y = 0.952, 0.916, 0.876
AX_B, AX_T = 0.472, 0.845
HEAD_Y, SUB_Y = 0.356, 0.306
BAND_B, BAND_T = 0.014, 0.256
BAND_TITLE_Y = 0.234
BAND_AX_B, BAND_AX_T = 0.082, 0.198

MARGIN, GUTTER = 0.012, 0.024
PW = (1.0 - 2 * MARGIN - 3 * GUTTER) / 4.0
PANEL_L = [MARGIN + i * (PW + GUTTER) for i in range(4)]
BAND_L, BAND_W = MARGIN, 1.0 - 2 * MARGIN

fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=300)
fig.patch.set_facecolor("white")

_checks = []          # (artist, allowed width in figure fraction, tag)


def fit(text_artist, width, tag):
    """Queue a text artist whose rendered width must stay inside `width`."""
    _checks.append((text_artist, width, tag))
    return text_artist


def verify_fit():
    """Report any string wider than the box it was placed in."""
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


def panel_axes(i):
    """Panel 1 carries a y axis, so it alone takes a left gutter for its tick
    labels; labelling those inside the plot put them on top of the curves."""
    gutter = 0.044 if i == 0 else 0.0
    return fig.add_axes([PANEL_L[i] + gutter, AX_B, PW - gutter, AX_T - AX_B])


def panel_head(i, n, title, head, sub):
    """Numbered chip, column title, one-line headline, sub-line."""
    x = PANEL_L[i]
    # chip centred on the title line; lifting it to the header band put the
    # digit on top of the header rule
    fig.patches.append(
        FancyBboxPatch((x - 0.0095, P_TITLE_Y - 0.0180), 0.019, 0.036,
                       boxstyle="round,pad=0,rounding_size=0.008", linewidth=0,
                       facecolor=BLUE, transform=fig.transFigure, figure=fig,
                       zorder=6))
    fig.text(x, P_TITLE_Y, str(n), ha="center", va="center",
             fontsize=5, fontweight="bold", color="white", zorder=7)
    t = fig.text(x + 0.020, P_TITLE_Y, title, ha="left", va="center",
                 fontsize=6.5, fontweight="bold", color=INK)
    fit(t, PW - 0.020, f"title{i}")
    t = fig.text(x, HEAD_Y, head, ha="left", va="center", fontsize=7,
                 fontweight="bold", color=BLUE)
    fit(t, PW, f"head{i}")
    t = fig.text(x, SUB_Y, sub, ha="left", va="center", fontsize=5,
                 color=GRAY)
    fit(t, PW, f"sub{i}")


# ==========================================================================
# header
# ==========================================================================
t = fig.text(MARGIN, HEADER_Y, "DeltaCycle", ha="left", va="center",
             fontsize=9.5, fontweight="bold", color=BLUE)
t2 = fig.text(MARGIN + 0.142, HEADER_Y,
              "multi-scale linear attention for on-device battery prognostics",
              ha="left", va="center", fontsize=6.5, color=INK)
fit(t2, BAND_W - 0.142 - 0.02, "header")
fig.patches.append(Rectangle((MARGIN, RULE_Y), BAND_W, 0.0032, linewidth=0,
                             facecolor=GRAY_L, transform=fig.transFigure,
                             figure=fig, zorder=2))

# ==========================================================================
# 1 -- edge deployment: the state is the only curve that does not grow
# ==========================================================================
ax = panel_axes(0)
TOK = np.logspace(np.log2(16), np.log2(4096), 60, base=2)
SCORES = 4.0 * TOK ** 2 * 4.0 / 1024.0     # H*L^2*4 B, peak of one layer
KV = 6.0 * 2.0 * TOK * 64.0 * 4.0 / 1024.0  # six layers, resident
STATE = 48.0                                # six-layer recurrent state

ax.plot(TOK, SCORES, color=RED, lw=0.9, zorder=3, label="attention scores")
ax.plot(TOK, KV, color=GRAY, lw=0.9, ls=(0, (3.2, 1.8)), zorder=3,
        label="KV cache")
ax.plot(TOK, np.full_like(TOK, STATE), color=BLUE, lw=1.8, zorder=4,
        label="ours: fixed state")
ax.axvline(32, color=GRAY_L, lw=0.6, zorder=1)
ax.plot([32], [STATE], "o", ms=2.4, mfc="white", mec=BLUE, mew=0.9, zorder=5)
ax.text(0.0, 0.0, "", transform=ax.transAxes)   # keep axes autoscale inert
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlim(13, 4200)
# the floor sits at 0.05 KB rather than 5 KB so the flat state line rides high
# enough to leave the whole lower-right quadrant free for the key
ax.set_ylim(0.05, 4e5)
ax.set_xticks([16, 128, 1024, 4096])
ax.set_xticklabels(["16", "128", "1024", "4096"])
ax.set_yticks([16, 256, 4096, 65536])
ax.set_yticklabels(["16 KB", "256", "4096", "65536"])
ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.set_xlabel("context length L (tokens)", fontsize=5, color=GRAY, labelpad=1.2)
# three curves cross the panel diagonally and no one of them has clear space
# beside it; below the flat state line is the only region without data
ax.legend(loc="lower right", frameon=False, fontsize=5, handlelength=1.5,
          handletextpad=0.4, labelspacing=0.25, borderpad=0.1,
          borderaxespad=0.3)
ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.set_xlabel("context length L (tokens)", fontsize=5, color=GRAY, labelpad=1.2)
clean(ax)

panel_head(0, 1, "Edge deployment",
           "runs in C on Cortex-M3",
           # plain "1.07e-6", not mathtext: mathtext renders a superscript at
           # ~0.7x the base size, i.e. under the 5 pt floor for a 5 pt line.
           # The window mark rides in this line too -- inside the plot the
           # key's handle stroke runs through any label near the flat curve
           "1.07e-6 vs PyTorch, at L = 32")

# ==========================================================================
# 2 -- multi-scale branches with per-layer cross-scale exchange
# ==========================================================================
ax = panel_axes(1)
caps, _tr, test_cell, _W, _sps, _eol = load_series("panasonic")
full = np.asarray(caps[test_cell], dtype=float)

# densest cluster of regeneration rises -> a 24-cycle window around it
d = np.diff(full)
rises = np.where(d > 0.004)[0]
rises = rises[(rises > 30) & (rises < len(full) - 30)]
if len(rises):
    best, best_n = rises[0], 0
    for c in rises:
        n = int(np.sum(np.abs(rises - c) < 30))
        if n > best_n:
            best, best_n = c, n
    c0 = int(best)
else:
    c0 = len(full) // 2
Z = 24
z0 = max(0, min(c0 - Z // 2, len(full) - Z))
seg = full[z0:z0 + Z]
xs = np.arange(Z)

# the capacity window lives in the top half of the panel; the patch grids
# occupy the bottom half, so the curve is rescaled into a 0..1.05 space
y_curve_lo, y_curve_hi = 0.56, 1.00
lo_s, hi_s = seg.min(), seg.max()
pad = 0.12 * (hi_s - lo_s + 1e-9)
curve = y_curve_lo + (y_curve_hi - y_curve_lo) * (
    (seg - (lo_s - pad)) / ((hi_s + pad) - (lo_s - pad)))
ax.plot(xs, curve, color=INK, lw=1.1, solid_capstyle="round", zorder=5)

ROWS = [(0.44, 8, BLUE, "8"), (0.21, 4, BLUE_MID, "4"), (-0.02, 2, BLUE_PALE, "2")]
for y0, p, col, lab in ROWS:
    for s in range(0, Z, p):
        ax.add_patch(Rectangle((s + 0.14, y0), p - 0.28, 0.068,
                               facecolor=col, edgecolor="none", zorder=3))
    ax.text(-1.4, y0 + 0.034, lab, fontsize=5, color=col, fontweight="bold",
            ha="right", va="center")

ax.add_patch(FancyArrowPatch((13, 0.415), (13, 0.295), arrowstyle="-|>",
                             mutation_scale=5.0, linewidth=0.8, color=BLUE,
                             zorder=6))
ax.add_patch(FancyArrowPatch((13, 0.185), (13, 0.065), arrowstyle="-|>",
                             mutation_scale=5.0, linewidth=0.8, color=BLUE,
                             zorder=6))
ax.text(11.0, 0.360, "query", fontsize=5, color=BLUE, ha="right", va="center")

ax.set_xlim(-6.0, Z + 0.4)
ax.set_ylim(-0.12, 1.09)
ax.set_yticks([])
ax.set_xticks([0, 8, 16, 24])
ax.set_xticklabels(["0", "8", "16", "24"])
ax.set_xlabel("cycles", fontsize=5, color=GRAY, labelpad=1.2)
clean(ax)

panel_head(1, 2, "Cross-scale exchange",
           "−5.7% MAE (p = 0.014)",
           "parameter-matched control")

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
    caps, _t, cell, Wc, _s, eol = load_series("calce")
    tc = (np.asarray(caps[cell], dtype=float) - c_lo) / (c_hi - c_lo + 1e-6)
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


ax = panel_axes(2)
xb, lo_b, mid_b, hi_b, true_b, eol_b = uq_band()
ax.fill_between(xb, lo_b, hi_b, color=BLUE_PALE, linewidth=0, zorder=1)
ax.plot(xb, true_b, color=INK, lw=0.8, zorder=4)
ax.plot(xb, mid_b, color=BLUE, lw=0.8, zorder=3)
miss = (true_b < lo_b) | (true_b > hi_b)
if miss.any():
    ax.plot(xb[miss], true_b[miss], "v", ms=1.1, color=RED, zorder=5,
            markeredgewidth=0)
ax.axhline(eol_b, color=RED, lw=0.7, ls=(0, (2.4, 1.6)), zorder=2)
ax.text(xb[0] + 10, eol_b + 0.011, "EOL", fontsize=5, color=RED, ha="left",
        va="bottom")
# inside the axes at the lower left: the band sits high at the left of this
# panel and falls only near EOL, which is exactly what leaves this corner free
ax.text(0.03, 0.05, "P2.5 / P50 / P97.5", transform=ax.transAxes, fontsize=5,
        color=BLUE, ha="left", va="bottom")
ax.set_xlim(xb[0], xb[-1])
ax.set_ylim(min(lo_b.min(), true_b.min()) - 0.045,
            max(hi_b.max(), true_b.max()) + 0.095)
ax.set_yticks([])
ax.set_xticks([xb[0], 400, 800])
ax.set_xticklabels([str(xb[0]), "400", "800"])
ax.set_xlabel("cycle", fontsize=5, color=GRAY, labelpad=1.2)
clean(ax)

panel_head(2, 3, "Calibrated intervals",
           "93.5% at 95% nominal",
           "two scalars, no ensemble")

# ==========================================================================
# 4 -- physics-consistent degradation-rate head
# ==========================================================================
ax = panel_axes(3)
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
# the three curves converge at the right edge, so right-edge labels collide.
# Only two entries fit without reaching the regeneration spike at x~855, so
# the measured curve is named in the x label instead of the key.
ax.legend(loc="upper right", frameon=False, fontsize=5, handlelength=1.5,
          handletextpad=0.4, labelspacing=0.30, borderpad=0.1,
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

panel_head(3, 4, "Physics rate head",
           "tail R² 0.37 → 0.77",
           "wins every corrupted-input setting")

# ==========================================================================
# 5 -- six-dataset scorecard
# ==========================================================================
# zorder must be NEGATIVE: Figure._get_draw_artists sorts axes and figure
# patches together by zorder, and on a tie the list order puts axes first --
# so a zorder=0 rectangle here paints over every bar in the band.
fig.patches.append(Rectangle((BAND_L, BAND_B), BAND_W, BAND_T - BAND_B,
                             linewidth=0, facecolor=BAND_BG,
                             transform=fig.transFigure, figure=fig, zorder=-1))

DS = ["PANASONIC", "TJU", "GOTION", "MIT", "NASA", "CALCE"]
# per-SP mean MAE (tab:lit_all), ours and the closest baseline on that dataset
OURS = np.array([0.003767, 0.001500, 0.008033, 0.002533, 0.008900, 0.007867])
BEST = np.array([0.007733, 0.001967, 0.008733, 0.003200, 0.007767, 0.006933])
ratio = OURS / BEST

t = fig.text(MARGIN, BAND_TITLE_Y, "Across six datasets", fontsize=6.5,
             fontweight="bold", color=INK, ha="left", va="center")
fit(t, 0.174, "bandtitle")     # ends before the sub-line starts at 0.196
t = fig.text(MARGIN + 0.196, BAND_TITLE_Y,
             "MAE relative to the closest baseline; 1.00 = the baseline",
             fontsize=5, color=GRAY, ha="left", va="center")
fit(t, 0.45, "bandsub")
t = fig.text(BAND_L + BAND_W - 0.004, BAND_TITLE_Y,
             "first on four, second on the other two", fontsize=6.5,
             fontweight="bold", color=BLUE, ha="right", va="center")
fit(t, 0.34, "bandright")

axb = fig.add_axes([BAND_L, BAND_AX_B, BAND_W, BAND_AX_T - BAND_AX_B])
axb.set_facecolor("none")          # let the band tint show through
xs = np.arange(len(DS))
axb.bar(xs, ratio, width=0.40, color=[BLUE if r < 1 else RED for r in ratio],
        edgecolor="none", linewidth=0, zorder=3)
# the 1.00 reference is drawn only in the gaps between bars: a full-width
# axhline runs straight through the value labels stacked above each bar,
# because a bar just under 1.00 puts its label across the line
for x in xs[:-1]:
    axb.plot([x + 0.22, x + 0.78], [1.0, 1.0], color=INK, lw=0.7,
             ls=(0, (2.4, 1.8)), zorder=4)
axb.text(-0.55, 1.02, "1.00", fontsize=5, color=INK, ha="left", va="bottom")
for x, r in zip(xs, ratio):
    axb.text(x, r + 0.05, f"{r:.2f}", fontsize=5.5,
             color=BLUE if r < 1 else RED, fontweight="bold", ha="center",
             va="bottom")
axb.set_xlim(-0.62, len(DS) - 0.38)
axb.set_ylim(0, 1.58)
axb.set_yticks([])
axb.set_xticks(xs)
axb.set_xticklabels(DS)
axb.tick_params(axis="x", length=0, pad=1.6, labelsize=5.5)
clean(axb)

# ==========================================================================
out = os.path.join(_ROOT, "submission", "ga_v2")
os.makedirs(os.path.dirname(out), exist_ok=True)
verify_fit()
plt.rcParams["savefig.bbox"] = None      # re-assert: nothing may re-enable it
fig.savefig(out + ".pdf")
fig.savefig(out + ".png", dpi=600)

from PIL import Image   # noqa: E402
Image.open(out + ".png").convert("RGB").resize((500, 200), Image.LANCZOS) \
     .save(os.path.join(os.path.dirname(out), "_ga_v2_preview500.png"))
print("wrote", out + ".pdf / .png and _ga_v2_preview500.png")
