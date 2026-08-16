"""Insight figures:
  - fig_w_interp:   readout W matrix (32x128) interpretability
                    (row norms, adjacent-row similarity, SVD spectrum)
  - fig_state_evol: GDN-2 state S norm / update magnitude across a
                    window scan (PANASONIC, regeneration-rich)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from make_figures import predict_series, load_series

CKPT = "D:/research/degradation_prognostics/Transformer_and_Multi_Scale_Models/checkpoints"
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper", "figures")
os.makedirs(FIG, exist_ok=True)

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

plt.rcParams.update(
    {
        "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
        "legend.fontsize": 7, "figure.dpi": 300, "savefig.dpi": 300,
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif", "axes.grid": True,
        "grid.alpha": 0.3, "grid.linewidth": 0.4,
        "axes.spines.top": False, "axes.spines.right": False,
        # rcParams.update() MERGES: make_figures.py (imported above)
        # sets savefig.bbox='tight', which with matplotlib 3.11
        # computes a broken portrait bbox for this figure. Override.
        "savefig.bbox": None,
    }
)


def fig_w_interp():
    """Readout W matrix (K=32): row norms, similarity, SVD spectrum."""
    sd = torch.load(f"{CKPT}/unified_calce_K32.pt", map_location="cpu",
                    weights_only=True)
    W = sd["head_cap.3.weight"].numpy()  # (32, 128)
    K = W.shape[0]

    row_norm = np.linalg.norm(W, axis=1)
    sim = np.array([
        np.dot(W[i], W[i + 1]) / (np.linalg.norm(W[i]) * np.linalg.norm(W[i + 1]) + 1e-8)
        for i in range(K - 1)
    ])
    s = np.linalg.svd(W, compute_uv=False)
    cumvar = np.cumsum(s ** 2) / np.sum(s ** 2)
    dim95 = int(np.argmax(cumvar >= 0.95) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0))
    ax = axes[0]
    ax.plot(np.arange(1, K + 1), row_norm, color=COLORS[0], lw=1.2)
    ax.set_xlabel("output step k")
    ax.set_ylabel("$\\|W_k\\|_2$")
    ax.set_title("Row norm: no decay toward k=32")
    ax.set_ylim(bottom=0)

    ax = axes[1]
    ax.plot(np.arange(1, K), sim, color=COLORS[1], lw=1.2)
    ax.axhline(sim.mean(), color="0.4", ls="--", lw=0.8,
               label=f"mean {sim.mean():.3f}")
    ax.set_xlabel("adjacent step pair")
    ax.set_ylabel("cosine similarity")
    ax.set_title("Adjacent rows near-identical")
    ax.legend(frameon=False)

    ax = axes[2]
    ax.plot(np.arange(1, len(s) + 1), cumvar, color=COLORS[2], lw=1.2)
    ax.axhline(0.95, color="0.4", ls="--", lw=0.8)
    ax.axvline(dim95, color="0.4", ls="--", lw=0.8)
    ax.text(dim95 + 2, 0.6, f"95% at {dim95} dims", fontsize=7)
    ax.set_xlabel("singular value index")
    ax.set_ylabel("cumulative variance")
    ax.set_title("Low-rank trajectory template")
    ax.set_xlim(0, 40)

    fig.subplots_adjust(left=0.06, right=0.985, top=0.82, bottom=0.20,
                        wspace=0.42)
    fig.savefig(os.path.join(FIG, "fig_w_interp.png"), dpi=300)
    plt.close(fig)
    print(f"fig_w_interp.pdf done (row_sim={sim.mean():.3f}, dim95={dim95})")


def fig_state_evol():
    """GDN-2 state norm / update magnitude across a PANASONIC window."""
    from gdn_model import build_gdn_model

    pv, tv, lo, hi, W, sps, eol_ah = predict_series("panasonic", K=1)
    caps, _, test_cell, _, _, _ = load_series("panasonic")
    full = caps[test_cell]
    tc = (full - lo) / (hi - lo + 1e-8)

    model = build_gdn_model(multiscale=True, cross_exchange=True,
                            input_dim=1, window_size=W, output_len=1,
                            readout="last")
    model.load_state_dict(torch.load(f"{CKPT}/unified_panasonic_K1.pt",
                                     map_location="cpu", weights_only=True))
    model.eval()
    b0 = model.branches[0]            # fine branch (patch 2)
    gdn = b0.layers[0].gdn

    # pick a window inside the prediction region (with regeneration)
    t0 = 420
    x = torch.tensor(tc[t0 - W:t0], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

    with torch.no_grad():
        h = b0.layers[0].norm(b0.init_proj(x))  # (1, L/2, d_model)
        Lp = h.size(1)
        q = gdn.q_conv(gdn.q_proj(h))
        k = gdn.k_conv(gdn.k_proj(h))
        v = gdn.v_conv(gdn.v_proj(h))
        q = torch.nn.functional.normalize(q, p=2, dim=-1)
        k = torch.nn.functional.normalize(k, p=2, dim=-1)
        raw_g = gdn.f_proj(h).view(1, Lp, gdn.num_heads, gdn.head_k_dim)
        g = -gdn.A_log.exp().view(1, 1, gdn.num_heads, 1) * torch.nn.functional.softplus(
            raw_g + gdn.dt_bias.view(1, 1, gdn.num_heads, gdn.head_k_dim))
        b = gdn.b_proj(h).sigmoid()
        w = gdn.w_proj(h).sigmoid()
        q = q.view(1, Lp, gdn.num_heads, gdn.head_k_dim).transpose(1, 2)
        k = k.view(1, Lp, gdn.num_heads, gdn.head_k_dim).transpose(1, 2)
        v = v.view(1, Lp, gdn.num_heads, gdn.head_v_dim).transpose(1, 2)
        g = g.transpose(1, 2)
        b = b.view(1, Lp, gdn.num_heads, gdn.head_k_dim).transpose(1, 2)
        w = w.view(1, Lp, gdn.num_heads, gdn.head_v_dim).transpose(1, 2)
        decay = torch.exp(g)

        H, Dk, Dv = gdn.num_heads, gdn.head_k_dim, gdn.head_v_dim
        I = torch.eye(Dk).view(1, 1, Dk, Dk)
        S = torch.zeros(1, H, Dk, Dv)
        norms, deltas = [], []
        for t in range(Lp):
            kt, vt = k[:, :, t], v[:, :, t]
            bt, wt, dt = b[:, :, t], w[:, :, t], decay[:, :, t]
            erase = I - (kt * bt).unsqueeze(-1) * kt.unsqueeze(-2)
            write = kt.unsqueeze(-1) * (vt * wt).unsqueeze(-2)
            S_new = erase @ (dt.unsqueeze(-1) * S) + write
            norms.append(S_new.norm(dim=(-2, -1)).squeeze(0).numpy())
            deltas.append((S_new - S).norm(dim=(-2, -1)).squeeze(0).numpy())
            S = S_new
    norms = np.stack(norms)     # (Lp, H)
    deltas = np.stack(deltas)
    inp = tc[t0 - W:t0][::2]    # align to patch-2 steps

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.1))
    ax = axes[0]
    for h in range(H):
        ax.plot(np.arange(1, Lp + 1), norms[:, h], lw=1.0,
                label=f"head {h + 1}")
    ax.set_xlabel("scan step within window (patch-2)")
    ax.set_ylabel("$\\|S_t\\|_F$")
    ax.set_title("State norm per head (fine branch, layer 1)")
    ax.legend(frameon=False, ncol=2, fontsize=6.5)

    ax = axes[1]
    ax.plot(np.arange(1, Lp + 1), deltas.mean(axis=1), color=COLORS[0], lw=1.0,
            label="mean update $\\|\\Delta S_t\\|_F$")
    ax2 = ax.twinx()
    ax2.plot(np.arange(1, Lp + 1), inp, color="0.4", lw=1.0, ls="--",
             label="input capacity (norm.)")
    ax2.set_ylabel("input capacity (normalized)")
    ax2.grid(False)
    ax.set_xlabel("scan step within window")
    ax.set_ylabel("mean $\\|\\Delta S_t\\|_F$")
    ax.set_title("State update magnitude tracks input variations")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper right", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_state_evol.pdf"))
    plt.close(fig)
    print("fig_state_evol.pdf done")


if __name__ == "__main__":
    fig_w_interp()
    fig_state_evol()
