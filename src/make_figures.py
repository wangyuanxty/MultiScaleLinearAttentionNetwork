"""Generate all data figures for the Gated DeltaFormer paper.

Figures written to ../paper/figures/*.pdf (vector). Sources:
  - fig_traj:   non-recursive SOH trajectory + prediction per dataset (unified_*_K1.pt)
  - fig_ablation:  per-dataset MAE by config (published ablation numbers)
  - fig_stages:  PCA of PANASONIC coarse-branch patches with stage coloring (unified_panasonic_K1.pt)
  - fig_k32:     CALCE K=32 early-sensing demo (unified_calce_K32.pt)
  - fig_deploy:  INT8 memory + MCU latency panels (measured numbers)
  - fig_quantile: quantile coverage panel (reported numbers)
"""
import sys
import os
import pickle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from gdn_model import build_gdn_model
from load_datasets import (
    load_calce_cells_multivar,
    load_nasa_multivar,
    load_mit_stanford,
    load_panasonic_cells,
    load_tju_cells,
)

CKPT = "D:/research/degradation_prognostics/Transformer_and_Multi_Scale_Models/checkpoints"
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper", "figures")
os.makedirs(FIG, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


_CACHE_DIR = os.path.join(CKPT, "data_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)


def load_series(ds):
    """Return (caps dict, train cells, test cell, W, sps, eol_ah).

    CALCE xlsx parsing takes ~107 s, so results are pickled and reused.
    """
    cache_path = os.path.join(_CACHE_DIR, f"load_series_{ds}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    result = _load_series_raw(ds)
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    return result


def _load_series_raw(ds):
    """Actual loader (no caching)."""
    if ds == "calce":
        caps_all, _, _ = load_calce_cells_multivar()
        caps = {c: caps_all[c].copy().astype(np.float32) for c in caps_all}
        return caps, ["CS2_36", "CS2_37", "CS2_38"], "CS2_35", 64, [300, 400, 500], 0.77
    if ds == "nasa":
        caps = {}
        for b in ["B0005", "B0006", "B0007", "B0018"]:
            caps[b] = load_nasa_multivar(b)["capacity"].astype(np.float32)
        return caps, ["B0006", "B0007", "B0018"], "B0005", 30, [50, 70, 90], 1.40
    if ds == "mit":
        from load_datasets import MIT_TRAIN_CELLS, MIT_TEST_CELLS
        caps_all = load_mit_stanford()
        caps = {c: caps_all[c].copy().astype(np.float32)
                for c in MIT_TRAIN_CELLS + MIT_TEST_CELLS}
        return caps, MIT_TRAIN_CELLS, MIT_TEST_CELLS[0], 64, [200, 300, 400], 0.86
    if ds == "panasonic":
        caps_all = load_panasonic_cells()
        caps = {c: caps_all[c].copy().astype(np.float32) for c in caps_all}
        cells = sorted(caps.keys())
        return caps, cells[:-1], cells[-1], 30, [300, 500, 700], 2.12
    # tju
    caps_all = load_tju_cells()
    caps = {c: caps_all[c].copy().astype(np.float32) for c in caps_all}
    return caps, ["CY25_2", "CY25_3"], "CY25_1", 64, [200, 300, 400], 1.75


def norm_setup(caps, train_cells):
    all_tr = np.concatenate([caps[c] for c in train_cells])
    lo, hi = all_tr.min(), all_tr.max()
    return lo, hi


def predict_series(ds, K=1, seeds=(42, 43, 44)):
    """Non-recursive K-step predictions over the test cell. Returns (pv, tv, lo, hi, W, sps, eol_ah).

    Averages the predictions over the given seed checkpoints
    (default: the three seeds used in the paper). Legacy single
    checkpoints (no _seed suffix) are used as fallback for seed 42.
    """
    caps, train_cells, test_cell, W, sps, eol_ah = load_series(ds)
    lo, hi = norm_setup(caps, train_cells)
    tc = (caps[test_cell] - lo) / (hi - lo + 1e-8)
    windows = np.stack([tc[i - W:i] for i in range(W, len(tc))])  # (N, W)
    cin = torch.tensor(windows, dtype=torch.float32).unsqueeze(-1).to(DEV)

    if seeds is None:
        paths = [f"{CKPT}/unified_{ds}_K{K}.pt"]
    else:
        paths = []
        for s in seeds:
            p = f"{CKPT}/unified_{ds}_K{K}_seed{s}.pt"
            if s == 42 and not os.path.exists(p):
                # legacy naming; MIT additionally had the mit-subset
                # prefix before the full-dataset rename
                for alt in (f"{CKPT}/unified_{ds}_K{K}.pt",
                            f"{CKPT}/unified_mit-subset_K{K}_seed42.pt"):
                    if os.path.exists(alt):
                        p = alt
                        break
            if os.path.exists(p):
                paths.append(p)  # skip seeds still training
    if not paths:
        raise FileNotFoundError(f"no unified checkpoint for {ds} K={K}")
    preds = []
    for ckpt in paths:
        model = build_gdn_model(
            multiscale=True, cross_exchange=False, stage_query=True,
            input_dim=1, window_size=W, output_len=K, readout="last",
        ).to(DEV)
        model.load_state_dict(torch.load(ckpt, map_location=DEV, weights_only=True))
        model.eval()
        with torch.no_grad():
            p_norm = model(cin).cpu().numpy()  # (N, K) in per-window domain
        # per-window de-normalize: each window scaled by its own mean/std
        wmean = windows.mean(axis=1, keepdims=True)
        wstd = windows.std(axis=1, keepdims=True) + 1e-6
        preds.append(p_norm * wstd + wmean)  # back to (0,1) capacity space
    pv = np.mean(preds, axis=0)[: len(tc) - W]
    tv = tc[W:]
    return pv, tv, lo, hi, W, sps, eol_ah


def fig_traj():
    """5-dataset SOH + non-recursive K=1 prediction."""
    ds_meta = {
        "calce": ("CALCE (CS2-35)", "LCO · 1.1 Ah"),
        "nasa": ("NASA (B0005)", "LCO · 2.0 Ah"),
        "mit": ("MIT (batch2-05)", "LFP · 1.07 Ah"),
        "panasonic": ("PANASONIC", "NCA · 3.03 Ah"),
        "tju": ("TJU (CY25-1)", "NCM+NCA · 2.5 Ah"),
    }
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6), sharex=False)
    for ax, (ds, (title, sub)) in zip(axes.ravel(), ds_meta.items()):
        pv, tv, lo, hi, W, sps, eol_ah = predict_series(ds, K=1)
        caps, _, test_cell, _, _, _ = load_series(ds)
        full = caps[test_cell]
        x_full = np.arange(len(full))
        # prediction segment (from W to end), de-normalized
        x_pred = np.arange(W, W + len(tv))
        ax.plot(x_full, full, color="0.35", lw=1.1, label="true capacity")
        ax.plot(x_pred, pv[:, 0] * (hi - lo) + lo, color=COLORS[0], lw=0.9, ls="--",
                label="prediction (K=1)")
        ax.axhline(eol_ah, color="#d62728", lw=0.9, ls=":", label="EOL threshold")
        for sp in sps:
            ax.axvline(sp, color="#2ca02c", lw=0.7, ls="-.", alpha=0.8)
        ax.set_title(f"{title}\n{sub}", fontsize=9)
        ax.set_xlabel("cycle")
        ax.set_ylabel("capacity (Ah)")
        ax.legend(loc="best", frameon=False, ncol=1)
    axes.ravel()[-1].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_traj.pdf"))
    plt.close(fig)
    print("fig_traj.pdf done")


def fig_ablation():
    """Grouped bar of per-dataset mean MAE by config (log scale)."""
    data = {
        "CALCE": {"single": 0.0092, "multi": 0.0131, "xchg": 0.0139, "physics": 0.0123},
        "NASA": {"single": 0.0071, "multi": 0.0150, "xchg": 0.0108, "physics": 0.0164},
        "PANASONIC": {"single": 0.0058, "multi": 0.0136, "xchg": 0.0036},
        "MIT": {"single": 0.0018, "multi": 0.0011, "xchg": 0.0022, "physics": 0.0047},
        "TJU": {"single": 0.0051, "multi": 0.0032, "xchg": 0.0064},
    }
    names = list(data.keys())
    cfgs = ["single", "multi", "xchg", "physics"]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = np.arange(len(names))
    width = 0.2
    for i, cfg in enumerate(cfgs):
        vals = [data[d].get(cfg, np.nan) for d in names]
        ax.bar(x + (i - 1.5) * width, vals, width, label=cfg,
               color=COLORS[i], alpha=0.85)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("mean trajectory MAE (log)")
    ax.legend(frameon=False, ncol=4, loc="upper center")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_ablation.pdf"))
    plt.close(fig)
    print("fig_ablation.pdf done")


def coarse_patch_features(ds="panasonic"):
    """Extract coarse-branch (patch 8) per-window mean-pooled features."""
    caps, train_cells, test_cell, W, sps, eol_ah = load_series(ds)
    lo, hi = norm_setup(caps, train_cells)
    tc = (caps[test_cell] - lo) / (hi - lo + 1e-8)
    ckpt = f"{CKPT}/unified_{ds}_K1.pt"
    model = build_gdn_model(
        multiscale=True, cross_exchange=True, input_dim=1,
        window_size=W, output_len=1, readout="last",
    ).to(DEV)
    model.load_state_dict(torch.load(ckpt, map_location=DEV, weights_only=True))
    model.eval()
    b2 = model.branches[2]  # coarse branch (patch 8)
    windows = np.stack([tc[i - W:i] for i in range(W, len(tc))])  # (N, W)
    cin = torch.tensor(windows, dtype=torch.float32).unsqueeze(-1).to(DEV)
    with torch.no_grad():
        h = b2.init_proj(cin)
        pm = b2.make_phys_mod(cin, None)
        for lidx in range(len(b2.layers)):
            h = b2.apply_layer(h, lidx, pm)
        feats = h.mean(dim=1).cpu().numpy()  # (N, d_model)
    return feats, np.arange(W, len(tc))


def fig_stages():
    """PCA of coarse-branch features, colored by degradation stage."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression

    feats, idx = coarse_patch_features()
    n = len(feats)
    frac = (idx - idx[0]) / (idx[-1] - idx[0])
    labels = np.clip((frac * 3).astype(int), 0, 2)  # early/mid/late terciles
    pca = PCA(n_components=2, random_state=0)
    proj = pca.fit_transform(feats)
    stage_names = ["early", "mid", "late"]
    stage_colors = ["#1f77b4", "#ff7f0e", "#d62728"]

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
    ax = axes[0]
    for s in range(3):
        m = labels == s
        ax.scatter(proj[m, 0], proj[m, 1], s=9, c=stage_colors[s],
                   label=stage_names[s], alpha=0.85)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"Coarse branch (patch 8) — PCA\n{n} windows, explained var "
                 f"{pca.explained_variance_ratio_.sum():.2f}")
    ax.legend(frameon=False, loc="best")

    ax = axes[1]
    # linear separability: logistic regression on stage labels
    clf = LogisticRegression(max_iter=2000)
    clf.fit(feats, labels)
    acc = clf.score(feats, labels)
    cm = np.zeros((3, 3), dtype=int)
    pred = clf.predict(feats)
    for a, b in zip(labels, pred):
        cm[a, b] += 1
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.2)
    ax.set_xticks(range(3))
    ax.set_xticklabels(stage_names)
    ax.set_yticks(range(3))
    ax.set_yticklabels(stage_names)
    ax.set_xlabel("predicted stage")
    ax.set_ylabel("true stage")
    ax.set_title(f"Linear stage classifier — accuracy {acc * 100:.1f}% "
                 f"(chance {100 / 3:.1f}%)")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() * 0.6 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_stages.pdf"))
    plt.close(fig)
    print(f"fig_stages.pdf done (separability {acc * 100:.1f}%)")


def fig_k32():
    """CALCE early-sensing demo: window-end predictions + first crossing."""
    pv, tv, lo, hi, W, sps, eol_ah = predict_series("calce", K=32,
                                                    seeds=(42, 43, 44))
    caps, _, test_cell, _, _, _ = load_series("calce")
    full = caps[test_cell]
    x_full = np.arange(len(full))

    # window-end prediction: pred[j] is the j-th step ahead; window starting at W+t
    x_start = np.arange(W, W + len(pv))
    window_end = pv[:, -1] * (hi - lo) + lo  # last value of each 32-step window

    true_eol = int(np.argmax(full < eol_ah))
    # first predicted crossing (AE = 16.9 → predicted EOL at true_eol - 16.9)
    pred_eol = None
    for t in range(len(pv)):
        w = pv[t] * (hi - lo) + lo
        if w[-1] < eol_ah:
            for j in range(len(w) - 1):
                if w[j] >= eol_ah > w[j + 1]:
                    frac = (eol_ah - w[j]) / (w[j + 1] - w[j] + 1e-8)
                    pred_eol = x_start[t] + j + frac
                    break
            else:
                pred_eol = x_start[t] + len(w) - 1
            break

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2))
    ax = axes[0]
    ax.plot(x_full, full, color="0.35", lw=1.2, label="true capacity")
    # a few example windows
    for t in [300, 360, 420, 470]:
        if t < W or t - W >= len(pv):
            continue
        w = pv[t - W] * (hi - lo) + lo
        xs = np.arange(t, t + 32)
        ax.plot(xs, w, lw=0.8, alpha=0.8,
                color=COLORS[(t // 60) % 4],
                label=f"window@{t}" if t in (300, 470) else None)
    ax.axhline(eol_ah, color="#d62728", lw=1, ls=":", label="EOL 0.77 Ah")
    ax.axvline(true_eol, color="#2ca02c", lw=1, ls="-.", label=f"true EOL ({true_eol})")
    if pred_eol is not None:
        ax.axvline(pred_eol, color="#8e44ad", lw=1, ls="--",
                   label=f"predicted EOL ({pred_eol:.0f})")
    ax.set_xlabel("cycle")
    ax.set_ylabel("capacity (Ah)")
    ax.set_title("K=32 one-shot windows expose the future 31 cycles")
    ax.legend(frameon=False, loc="best", fontsize=6.5)

    ax = axes[1]
    ax.plot(x_start, window_end, color=COLORS[0], lw=0.9,
            label="window-end prediction (32 steps ahead)")
    ax.plot(x_full[W:], tv * (hi - lo) + lo, color="0.35", lw=1.1, label="true capacity")
    ax.axhline(eol_ah, color="#d62728", lw=1, ls=":", label="EOL threshold")
    ax.axvline(true_eol, color="#2ca02c", lw=1, ls="-.")
    if pred_eol is not None:
        ax.axvline(pred_eol, color="#8e44ad", lw=1, ls="--")
    ax.annotate(f"31-cycle early\nsensing (AE 15.9, 3 seeds)",
                xy=(pred_eol, eol_ah + 0.02),
                xytext=(pred_eol + 40, eol_ah + 0.06),
                fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlabel("cycle")
    ax.set_ylabel("capacity (Ah)")
    ax.set_title("Window-end trajectory stays near truth until EOL")
    ax.legend(frameon=False, loc="lower left", fontsize=6.5)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_k32.pdf"))
    plt.close(fig)
    print(f"fig_k32.pdf done (pred_eol={pred_eol:.1f}, true={true_eol})")


def fig_deploy():
    """Two panels: INT8 memory, and latency on MCUs."""
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0))

    ax = axes[0]
    mems = ["fp32\nsingle-branch", "INT8\nsingle-branch", "fp32\nmulti-scale", "INT8\nmulti-scale"]
    mem_vals = [1330, 337, 1330, 340]
    colors = ["#a6a6a6", COLORS[0], "#a6a6a6", COLORS[0]]
    bars = ax.bar(mems, mem_vals, color=colors, alpha=0.9, width=0.6)
    for b, v in zip(bars, mem_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 20, f"{v} KB",
                ha="center", fontsize=7.5)
    ax.set_ylabel("weights memory (KB)")
    ax.set_title("INT8 quantization: −75% memory, lossless AE")
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    hw = ["Cortex-M3\n25 MHz (soft FPU)", "Cortex-M4F\n48 MHz", "STM32F4\n168 MHz"]
    lat = [428, 35, 10]
    lat_lo = [428, 30, 8]
    lat_hi = [428, 40, 12]
    ax.bar(hw, lat, yerr=[np.array(lat) - np.array(lat_lo),
                          np.array(lat_hi) - np.array(lat)],
           color=[COLORS[1], COLORS[2], COLORS[0]], alpha=0.9, width=0.55,
           capsize=3)
    for b, v in zip(ax.patches, lat):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.06, f"{v} ms",
                ha="center", fontsize=7.5)
    ax.set_ylabel("full inference latency (ms)")
    ax.set_title("Projected MCU latency (2 GDN-2 layers, L=64)")
    ax.grid(axis="x", visible=False)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_deploy.pdf"))
    plt.close(fig)
    print("fig_deploy.pdf done")


def fig_quantile():
    """Quantile coverage panel from reported numbers."""
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    thirds = ["overall", "early 1/3", "mid 1/3", "late 1/3"]
    cov = [90.6, 96, 97, 79]
    bars = ax.bar(thirds, cov, color=[COLORS[0], "#7fae5a", "#7fae5a", "#d62728"],
                  alpha=0.9, width=0.55)
    ax.axhline(80, color="0.3", lw=1, ls="--", label="80% target")
    for b, v in zip(bars, cov):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v}%", ha="center", fontsize=8)
    ax.set_ylabel("P10–P90 coverage (%)")
    ax.set_title("Quantile calibration: conservative overall,\nwell-calibrated near EOL")
    ax.set_ylim(0, 110)
    ax.legend(frameon=False)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_quantile.pdf"))
    plt.close(fig)
    print("fig_quantile.pdf done")


if __name__ == "__main__":
    fig_traj()
    fig_ablation()
    fig_stages()
    fig_k32()
    fig_deploy()
    fig_quantile()
