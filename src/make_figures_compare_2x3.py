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

# Mean per-SP MAE, read from Table 3 (all series on the same
# normalized-capacity scale).
DATA = {
    "NASA": {
        "TimeMixer": 0.0258, "TimesNet": 0.0331, "PatchTST": 0.0231,
        "MambaSimple": 0.0279, "ModernTCN": 0.0148, "Autoformer": 0.0256,
        "FEDformer": 0.0222, "iTransformer": 0.0091, "PathFormer": 0.0255,
        "PatchFormer": 0.0078, "RUL-Mamba": 0.0102, "Ours": 0.0089,
    },
    "TJU": {
        "TimeMixer": 0.0122, "TimesNet": 0.019, "PatchTST": 0.0093,
        "MambaSimple": 0.0139, "ModernTCN": 0.002, "Autoformer": 0.0063,
        "FEDformer": 0.0073, "iTransformer": 0.0023, "PathFormer": 0.0158,
        "PatchFormer": 0.0021, "RUL-Mamba": 0.002, "Ours": 0.0015,
    },
    "CALCE": {
        "TimeMixer": 0.0293, "TimesNet": 0.0465, "PatchTST": 0.0237,
        "MambaSimple": 0.0287, "ModernTCN": 0.0128, "Autoformer": 0.0248,
        "FEDformer": 0.0229, "iTransformer": 0.017, "PathFormer": 0.0545,
        "PatchFormer": 0.0069, "RUL-Mamba": 0.0209, "Ours": 0.0079,
    },
    "PANASONIC": {
        "iTransformer": 0.0177, "Autoformer": 0.024, "FEDformer": 0.0268,
        "PatchFormer": 0.0077, "RUL-Mamba": 0.0186, "Ours": 0.0038,
    },
    "MIT": {
        "PatchFormer": 0.0032, "RUL-Mamba": 0.0062, "Ours": 0.0025,
    },
    "GOTION": {
        "ModernTCN": 0.0091, "Autoformer": 0.0115, "FEDformer": 0.0117,
        "iTransformer": 0.0093, "PathFormer": 0.0307, "PatchFormer": 0.0089,
        "RUL-Mamba": 0.0087, "Ours": 0.008,
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
        ax.set_ylabel("average MAE (log)")

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
