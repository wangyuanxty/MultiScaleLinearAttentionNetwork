#!/usr/bin/env python3
"""Real-data SVG for fig_gdn2_block: per-layer mean decay from the
seed-42 CALCE checkpoint (data/decay.dat)."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent
D = ROOT / "data"
OUT = ROOT / "charts"

INK = "#1F262B"
MUTED = "#707A82"
TEAL = "#2E7D8C"


def gdn_decay():
    d = np.loadtxt(D / "decay.dat")
    fig, ax = plt.subplots(figsize=(4.4, 3.0), facecolor="none")
    ax.bar(d[:, 0], d[:, 1], color=TEAL, width=0.62,
           edgecolor="#1F262B", linewidth=0.6)
    for x, y in zip(d[:, 0], d[:, 1]):
        ax.text(x, y + 0.0004, f"{y:.4f}", ha="center", fontsize=10.5,
                fontweight="bold", color=INK)
    ax.set_xlabel("GDN-2 layer index", fontsize=12)
    ax.set_ylabel("mean decay", fontsize=12)
    ax.set_title("real per-layer decay weights (seed-42, CALCE ckpt)",
                 fontsize=11.5, color=MUTED, pad=6, fontstyle="italic")
    ax.tick_params(labelsize=10.5, length=3, width=0.7)
    for s in ax.spines.values():
        s.set_linewidth(0.7)
    ax.set_ylim(0, 0.032)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "gdn_decay.svg", format="svg", bbox_inches="tight",
                facecolor="none", edgecolor="none")
    plt.close(fig)
    print("ok gdn_decay.svg")


if __name__ == "__main__":
    gdn_decay()
