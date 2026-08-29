"""Regenerate paper figures from the new checkpoints (seed 42).

fig_traj:       per-dataset test-cell trajectory, truth vs prediction
fig_metrics_sp: per-SP MAE/RMSE/R^2 (from results/table_a_seed42.json)
fig_compare:    ours vs published baselines (NASA/TJU numbers from tab:lit)
fig_stages:     coarse-branch stage separability (PCA + linear classifier,
                PANASONIC, coarse branch = branches[2])

Outputs paper/figures/{fig_traj,fig_metrics_sp,fig_compare,fig_stages}.pdf
(+ .png). All data from checkpoints/full_*_seed42.pt — no retraining.
"""
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

sys.path.insert(0, '.')
from gdn_model import build_gdn_model
from make_figures import load_series

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-6
OUT = "../paper/figures/"

# matplotlib 3.11 tight-bbox bug guard: explicit rcParams after imports
plt.rcParams.update({
    "savefig.bbox": None,
    "figure.dpi": 150,
    "font.size": 8,
})


def load_model(ds):
    ck = torch.load(f"../checkpoints/full_{ds}_seed42.pt",
                    map_location=DEV, weights_only=False)
    cfg = ck["config"]
    model = build_gdn_model(
        multiscale=cfg["multiscale"], stage_query=cfg["stage_query"],
        input_dim=cfg["input_dim"], window_size=cfg["window_size"],
        output_len=cfg["output_len"], num_quantiles=cfg["num_quantiles"],
        readout=cfg["readout"],
    ).to(DEV)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def predict_seq(model, ds, lo, hi):
    caps, train_cells, test_cell, W, sps, eol_ah = load_series(ds)
    tc = (caps[test_cell] - lo) / (hi - lo + EPS)
    seg_p = []
    with torch.no_grad():
        for i in range(W, len(tc)):
            win = tc[i - W:i, None]
            cin = torch.tensor(win, dtype=torch.float32).unsqueeze(0).to(DEV)
            wmean = float(win[:, 0].mean())
            wstd = float(win[:, 0].std()) + EPS
            seg_p.append(model(cin).item() * wstd + wmean)
    seg_p = np.array(seg_p)
    tv = tc[W:]
    seg_ah_t = tv * (hi - lo + EPS) + lo
    seg_ah_p = seg_p * (hi - lo + EPS) + lo
    return caps, test_cell, W, sps, eol_ah, seg_ah_t, seg_ah_p


# ---------------- fig_traj ----------------
def fig_traj():
    ds_list = ["calce", "nasa", "mit", "panasonic", "tju"]
    fig, axes = plt.subplots(2, 3, figsize=(9, 5))
    axes = axes.flatten()
    for k, ds in enumerate(ds_list):
        model, ck = load_model(ds)
        caps, tc_cell, W, sps, eol, seg_t, seg_p = predict_seq(
            model, ds, ck["lo"], ck["hi"])
        ax = axes[k]
        x_t = np.arange(len(seg_t))
        x_p = np.arange(W, W + len(seg_p))
        ax.plot(x_t, seg_t, color="tab:blue", lw=1.0, label="true")
        ax.plot(x_p, seg_p, color="tab:red", lw=0.8, alpha=0.8,
                label="pred")
        for sp in sps:
            ax.axvline(sp, color="green", ls="--", lw=0.7)
        ax.axhline(eol, color="red", ls=":", lw=0.7)
        ax.set_title(ds.upper(), fontsize=9)
        if k in (0, 3):
            ax.set_ylabel("capacity (Ah)")
        if k >= 3:
            ax.set_xlabel("cycle")
        if k == 0:
            ax.legend(fontsize=6)
    axes[5].axis("off")
    fig.tight_layout()
    fig.savefig(OUT + "fig_traj.pdf")
    fig.savefig(OUT + "fig_traj.png", dpi=150)
    plt.close(fig)
    print("saved fig_traj", flush=True)


# ---------------- fig_metrics_sp ----------------
def fig_metrics_sp():
    import json
    d = json.load(open("results/table_a_seed42.json"))
    ds_list = ["calce", "nasa", "mit", "panasonic", "tju"]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, metric in zip(axes, ["MAE", "RMSE", "R2"]):
        for ds in ds_list:
            sps = [r["SP"] for r in d[ds]]
            vals = [r[metric] for r in d[ds]]
            ax.plot(sps, vals, marker="o", ms=3, lw=1, label=ds.upper())
        ax.set_title(metric)
        ax.set_xlabel("SP")
        if metric == "R2":
            ax.set_ylim(0.9, 1.01)
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(OUT + "fig_metrics_sp.pdf")
    fig.savefig(OUT + "fig_metrics_sp.png", dpi=150)
    plt.close(fig)
    print("saved fig_metrics_sp", flush=True)


# ---------------- fig_compare ----------------
def fig_compare():
    # baseline numbers from tab:lit (published, per-SP MAE); grouped bars
    methods = ["PatchFormer", "RUL-Mamba", "iTransformer",
               "ModernTCN", "TimeMixer", "Ours"]
    nasa = {  # SP50/70/90 MAE
        "PatchFormer": [0.0056, 0.0061, 0.0068],
        "RUL-Mamba": [0.0083, 0.0091, 0.0092],
        "iTransformer": [0.0076, 0.0078, 0.0086],
        "ModernTCN": [0.0166, 0.0127, 0.0098],
        "TimeMixer": [0.0239, 0.0241, 0.0203],
        "Ours": [0.0078, 0.0087, 0.0075],
    }
    tju = {
        "PatchFormer": [0.0013, 0.0014, 0.0015],
        "RUL-Mamba": [0.0014, 0.0015, 0.0016],
        "iTransformer": [0.0017, 0.0018, 0.0019],
        "ModernTCN": [0.0014, 0.0015, 0.0016],
        "TimeMixer": [0.0071, 0.0086, 0.0123],
        "Ours": [0.0011, 0.0012, 0.0012],
    }
    sps = ["SP1", "SP2", "SP3"]
    colors = {"Ours": "#4C8C5A"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, data, title in [(axes[0], nasa, "NASA"),
                            (axes[1], tju, "TJU")]:
        xs = np.arange(len(sps))
        width = 0.13
        for mi, m in enumerate(methods):
            offset = (mi - (len(methods) - 1) / 2) * width
            ax.bar(xs + offset, data[m], width,
                   color=colors.get(m, f"C{mi}"), edgecolor="black",
                   lw=0.5, label=m if title == "NASA" else None)
        ax.set_xticks(xs)
        ax.set_xticklabels(sps)
        ax.set_title(title)
        ax.set_xlabel("starting point")
        ax.set_ylabel("MAE (normalized)")
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.25)
        if title == "NASA":
            ax.legend(fontsize=6.5, ncol=2, loc="center left",
                      bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(OUT + "fig_compare.pdf")
    fig.savefig(OUT + "fig_compare.png", dpi=150)
    plt.close(fig)
    print("saved fig_compare", flush=True)


# ---------------- fig_stages ----------------
def fig_stages():
    """Coarse-branch (patch 8) pooled reps, PCA + linear separability."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    model, ck = load_model("panasonic")
    caps, train_cells, test_cell, W, sps, eol_ah = load_series("panasonic")
    lo, hi = ck["lo"], ck["hi"]

    pooled = []
    labels = []

    # hook the coarse branch's second-layer GDN2Block (its forward IS
    # called in the stage-query path; returns (out, S) -> take out)
    target = model.branches[2].layers[1]['gdn']

    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        pooled.append(h.mean(dim=1).cpu().numpy())

    model.eval()
    for c in list(train_cells) + [test_cell]:
        seq = (caps[c] - lo) / (hi - lo + EPS)
        n = len(seq)
        for i in range(W, n):
            win = seq[i - W:i, None]
            cin = torch.tensor(win, dtype=torch.float32).unsqueeze(0).to(DEV)
            with torch.no_grad():
                handle = target.register_forward_hook(hook)
                model(cin)
                handle.remove()
            labels.append((i + 1) / n)
    X = np.vstack(pooled)
    lab = np.array(labels)
    terc = np.digitize(lab, [1 / 3, 2 / 3])

    pca = PCA(n_components=2)
    Xp = pca.fit_transform(X)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xp, terc)
    acc = clf.score(Xp, terc)
    pred = clf.predict(Xp)

    # confusion matrix (3 stages)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(terc, pred, normalize="true")

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
    for t in np.unique(terc):
        axes[0].scatter(Xp[terc == t, 0], Xp[terc == t, 1], s=3,
                        label=f"stage {t}")
    axes[0].set_title("PCA projection")
    axes[0].legend(fontsize=6)
    im = axes[1].imshow(cm, cmap="Greens", vmin=0, vmax=1)
    axes[1].set_xticks([0, 1, 2])
    axes[1].set_yticks([0, 1, 2])
    axes[1].set_xticklabels(["early", "mid", "late"], fontsize=7)
    axes[1].set_yticklabels(["early", "mid", "late"], fontsize=7)
    axes[1].set_xlabel("predicted")
    axes[1].set_ylabel("true")
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, f"{cm[i, j]:.2f}", ha="center",
                         va="center", fontsize=7,
                         color="white" if cm[i, j] > 0.5 else "black")
    axes[1].set_title(f"confusion (acc = {acc:.3f})")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(OUT + "fig_stages.pdf")
    fig.savefig(OUT + "fig_stages.png", dpi=150)
    plt.close(fig)
    print(f"saved fig_stages (acc={acc:.3f})", flush=True)


# ---------------- fig_regen ----------------
def fig_regen():
    """PANASONIC regeneration zoom: the main model tracks local rises."""
    model, ck = load_model("panasonic")
    caps, tc_cell, W, sps, eol, seg_t, seg_p = predict_seq(
        model, "panasonic", ck["lo"], ck["hi"])
    # find regeneration segments in the truth (sustained local rise)
    d = np.diff(seg_t)
    up = d > 0.0002
    segs = []
    i = 0
    while i < len(up):
        if up[i]:
            j = i
            while j < len(up) and up[j]:
                j += 1
            if j - i >= 3:
                segs.append((i, j))
            i = j
        else:
            i += 1
    best = max(segs, key=lambda s: seg_t[s[1]] - seg_t[s[0]]) if segs else (200, 215)
    a, b = best
    z0, z1 = max(0, a - 25), min(len(seg_t), b + 25)
    x_full = np.arange(len(seg_t)) + W
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].plot(x_full, seg_t, color="tab:blue", lw=1.0, label="true")
    axes[0].plot(x_full, seg_p, color="tab:red", lw=0.8, alpha=0.8, label="pred")
    for s in segs:
        axes[0].axvspan(s[0] + W, s[1] + W, color="green", alpha=0.25)
    axes[0].axhline(eol, color="red", ls=":", lw=0.7)
    axes[0].set_title("PANASONIC test cell")
    axes[0].set_xlabel("cycle")
    axes[0].set_ylabel("capacity (Ah)")
    axes[0].legend(fontsize=6)
    axes[1].plot(x_full[z0:z1], seg_t[z0:z1], color="tab:blue", lw=1.4,
                 label="true", marker="o", ms=2.5)
    axes[1].plot(x_full[z0:z1], seg_p[z0:z1], color="tab:red", lw=1.0,
                 label="pred", marker="s", ms=2.5)
    axes[1].axvspan(a + W, b + W, color="green", alpha=0.25)
    axes[1].set_title("regeneration zone (zoom)")
    axes[1].set_xlabel("cycle")
    axes[1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(OUT + "fig_regen.pdf")
    fig.savefig(OUT + "fig_regen.png", dpi=150)
    plt.close(fig)
    print(f"saved fig_regen (zone {a + W}-{b + W}, "
          f"rise {seg_t[b] - seg_t[a]:.4f} Ah)", flush=True)


# ---------------- fig_uq_band ----------------
def fig_uq_band():
    """P2.5/P50/P97.5 interval band (CQR-calibrated) on the test cell."""
    ck = torch.load("../checkpoints/quantile_calce_seed42.pt",
                    map_location=DEV, weights_only=False)
    cfg = ck["config"]
    model = build_gdn_model(
        multiscale=cfg["multiscale"], stage_query=cfg["stage_query"],
        input_dim=cfg["input_dim"], window_size=cfg["window_size"],
        output_len=cfg["output_len"], num_quantiles=cfg["num_quantiles"],
        readout=cfg["readout"],
    ).to(DEV)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    lo, hi = ck["lo"], ck["hi"]
    caps, train_cells, test_cell, W, sps, eol_ah = load_series("calce")
    tc = (caps[test_cell] - lo) / (hi - lo + EPS)
    qs = []
    with torch.no_grad():
        for i in range(W, len(tc)):
            win = tc[i - W:i, None]
            cin = torch.tensor(win, dtype=torch.float32).unsqueeze(0).to(DEV)
            wmean = float(win[:, 0].mean())
            wstd = float(win[:, 0].std()) + EPS
            qz = model(cin).cpu().numpy().squeeze()
            qs.append(qz * wstd + wmean)
    qs = np.array(qs)
    q_adj = 0.0052  # CQR calibration constant (CS2_36, n=869)
    lo_q = (qs[:, 0] - q_adj) * (hi - lo + EPS) + lo
    mid_q = qs[:, 1] * (hi - lo + EPS) + lo
    hi_q = (qs[:, 2] + q_adj) * (hi - lo + EPS) + lo
    tv = (tc[W:] * (hi - lo + EPS) + lo)
    x = np.arange(W, len(tc))
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 3.2))
    ax.fill_between(x, lo_q, hi_q, color="tab:blue", alpha=0.18,
                    label="95% interval (CQR)")
    ax.plot(x, mid_q, color="tab:red", lw=1.0, label="P50")
    ax.plot(x, tv, color="black", lw=1.2, label="true", alpha=0.9)
    ax.axhline(eol_ah, color="red", ls=":", lw=0.7)
    ax.set_xlabel("cycle")
    ax.set_ylabel("capacity (Ah)")
    ax.set_title("CALCE test cell: CQR-calibrated interval")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT + "fig_uq_band.pdf")
    fig.savefig(OUT + "fig_uq_band.png", dpi=150)
    plt.close(fig)
    print("saved fig_uq_band", flush=True)


if __name__ == "__main__":
    fig_traj()
    fig_metrics_sp()
    fig_compare()
    fig_stages()
    fig_regen()
    fig_uq_band()
