"""Fig. 5 (fig_compare), relaid out as 2 rows x 3 columns to match Fig. 4.

make_figures_extra.fig_compare() draws this as a 3x2 portrait canvas
(7.8 x 8.6 in), which reads awkwardly as a full-width float.  Fig. 4
(fig_traj) uses 2 x 3 at 10.5 x 5.6 in; this script reproduces that
canvas and rcParams exactly so the two figures sit together.

The data dict is copied verbatim from make_figures_extra.py:97-128 --
same six datasets, same values, same colour roles.  make_figures_extra.py
is left untouched.

Usage: python src/make_figures_compare_2x3.py
Writes: paper/figures/fig_compare.pdf and .png (replacing the 3x2 render).
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "paper", "figures"
)
os.makedirs(FIG, exist_ok=True)

# rcParams copied from make_figures.py:37-56 -- the block that produced
# fig_traj, so the two figures share fonts, grid and bbox behaviour.
plt.rcParams.update(
    {
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

OURS = "#2ca02c"
SELF_RUN = "#1f77b4"
OTHER = "#b0b0b0"

# Mean per-SP MAE in Ah, read from Table 3 -- identical to the data dict in
# make_figures_extra.fig_compare().  The baselines were always in Ah; the
# "Ours" entries here used to be normalized, which on a log axis overstated
# our GOTION margin by 6.4x and understated MIT by 3.7x.  Keep this dict in
# sync with make_figures_extra.py whenever Table 3 changes.
DATA = {
    "NASA": {
        "TimeMixer": 0.0228, "TimesNet": 0.0292, "PatchTST": 0.0204,
        "MambaSimple": 0.0246, "ModernTCN": 0.0130, "Autoformer": 0.0226,
        "FEDformer": 0.0195, "iTransformer": 0.0080, "PathFormer": 0.0225,
        "PatchFormer": 0.0068, "RUL-Mamba": 0.0090, "Ours": 0.0078,
    },
    "TJU": {
        "TimeMixer": 0.0093, "TimesNet": 0.0146, "PatchTST": 0.0071,
        "MambaSimple": 0.0107, "ModernTCN": 0.0015, "Autoformer": 0.0049,
        "FEDformer": 0.0056, "iTransformer": 0.0018, "PathFormer": 0.0121,
        "PatchFormer": 0.0016, "RUL-Mamba": 0.0015, "Ours": 0.0012,
    },
    "CALCE": {
        "TimeMixer": 0.0291, "TimesNet": 0.0461, "PatchTST": 0.0235,
        "MambaSimple": 0.0284, "ModernTCN": 0.0127, "Autoformer": 0.0246,
        "FEDformer": 0.0227, "iTransformer": 0.0168, "PathFormer": 0.0541,
        "PatchFormer": 0.0069, "RUL-Mamba": 0.0207, "Ours": 0.0078,
    },
    "PANASONIC": {
        "iTransformer": 0.0180, "Autoformer": 0.0245, "FEDformer": 0.0274,
        "PatchFormer": 0.0079, "RUL-Mamba": 0.0189, "Ours": 0.0039,
    },
    "MIT": {
        "PatchFormer": 0.0008, "RUL-Mamba": 0.0016, "Ours": 0.0007,
    },
    "GOTION": {
        "ModernTCN": 0.0581, "Autoformer": 0.0731, "FEDformer": 0.0749,
        "iTransformer": 0.0597, "PathFormer": 0.1965,
        "PatchFormer": 0.0568, "RUL-Mamba": 0.0558, "Ours": 0.0514,
    },
}


def main():
    """2x3 panels, log y-axis shared across all six datasets."""
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6), sharey=True)
    for ax, (ds, d) in zip(axes.ravel(), DATA.items()):
        names = list(d.keys())
        vals = [d[n] for n in names]
        colors = [
            OURS if n == "Ours" else SELF_RUN if n in ("PatchFormer", "RUL-Mamba")
            else OTHER
            for n in names
        ]
        ax.bar(range(len(names)), vals, color=colors, width=0.62)
        ax.set_yscale("log")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=6.5)
        ax.set_title(ds)

    # With sharey the tick labels repeat on every panel; label the left
    # column of each row so the shared axis is named in both rows.
    for ax in (axes[0, 0], axes[1, 0]):
        ax.set_ylabel("average MAE, Ah (log)")

    axes[0, 0].legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, fc=OURS),
            plt.Rectangle((0, 0), 1, 1, fc=SELF_RUN),
            plt.Rectangle((0, 0), 1, 1, fc=OTHER),
        ],
        labels=["Ours", "PatchFormer / RUL-Mamba", "other baselines"],
        frameon=False,
        loc="upper left",
        fontsize=6.5,
    )

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_compare.pdf"))
    fig.savefig(os.path.join(FIG, "fig_compare.png"), dpi=300)
    plt.close(fig)
    print("fig_compare.pdf/png done (2x3)")


if __name__ == "__main__":
    main()
