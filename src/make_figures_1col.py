"""Column-width renders of Fig. 7 (fig_extrap) and Fig. 10 (fig_uq_band).

Both are drawn at 6.5 x 3.2 in in their own modules, so as full-width
floats they waste page.  This wrapper re-runs the *existing* plotting and
data pipelines with the canvas forced to the elsarticle 5p column width
(3.45 in), so nothing is reimplemented and no data path can drift.

Two things are patched, both cosmetic and both explicit:
  * plt.subplots -- figsize width forced to COL_W, height scaled to keep the
    original aspect but never below MIN_H (a proportional shrink would give a
    squat 1.7 in panel that the four-entry legend would dominate);
  * Axes.set_title -- the two original titles do not fit a column, so they are
    swapped for shorter equivalents.  Nothing else is touched.

make_figures_phys.py and make_figures_v2.py are left untouched.

Run FROM src/ (both modules write to "../paper/", a cwd-relative path):
    cd src && python make_figures_1col.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

COL_W = 3.45          # elsarticle 5p column width, inches
MIN_H = 2.2           # keep panels from becoming letterbox strips
HEADROOM = 0.18       # extra y-range above the data, for the legend band

_orig_subplots = plt.subplots
_orig_set_title = Axes.set_title
_orig_legend = Axes.legend
_orig_savefig = Figure.savefig

SHORT_TITLES = {
    "CALCE unseen-tail extrapolation (train 90%, eval 10%)":
        "Unseen-tail extrapolation (CALCE)",
    "CALCE test cell: CQR-calibrated interval":
        "CQR-calibrated interval",
}


def _subplots(*args, **kwargs):
    fs = kwargs.get("figsize")
    if fs:
        w, h = fs
        kwargs["figsize"] = (COL_W, max(COL_W * h / w, MIN_H))
    return _orig_subplots(*args, **kwargs)


def _set_title(self, label, *args, **kwargs):
    return _orig_set_title(self, SHORT_TITLES.get(label, label), *args, **kwargs)


def _compact_legend(self, *args, **kwargs):
    """Small legend, pinned to the top of a band opened by _headroom_savefig.

    The curves fill the panel: at 7pt the four entries cover the upper-right
    (where the true and last-value curves run), and moved low-left they cover
    the free-head curve instead.  Neither corner is free, so the y-range gets
    extra room above the data and the legend sits in that empty band.
    """
    kwargs["fontsize"] = 5.5
    kwargs["loc"] = "upper left"
    kwargs["borderpad"] = 0.25
    kwargs["labelspacing"] = 0.20
    kwargs["handlelength"] = 1.6
    kwargs["frameon"] = False
    return _orig_legend(self, *args, **kwargs)


def _headroom_savefig(self, *args, **kwargs):
    """Open an empty band above the data, then save normally."""
    for ax in self.axes:
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + HEADROOM * (hi - lo))
    return _orig_savefig(self, *args, **kwargs)


plt.subplots = _subplots
Axes.set_title = _set_title

# Imported after the patches so the module-level rcParams cannot undo them.
import make_figures_phys as P          # noqa: E402
import make_figures_v2 as V            # noqa: E402

assert os.path.exists(P.NPA), (
    "cached arrays missing: " + P.NPA
    + " -- run make_figures_phys.py once from src/ to build them"
)

d = np.load(P.NPA)
Axes.legend = _compact_legend          # applies to fig_extrap only
Figure.savefig = _headroom_savefig
P.plot_extrap(d["ext_x"], d["ext_truth"], d["ext_last"],
              d["ext_free"], d["ext_rate"])
Axes.legend = _orig_legend             # fig_uq_band keeps its own legend
Figure.savefig = _orig_savefig
V.fig_uq_band()
print("fig_extrap + fig_uq_band done (single column)")
