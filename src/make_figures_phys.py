"""Physics-extension figures for the paper (Section 4.6).

Two figures, trained fresh (the original diag runs saved no ckpts):

  fig_extrap  -- unseen-tail extrapolation on CALCE (train first 90% of
                 each cell, evaluate final 10%): truth vs last-value
                 continuation vs free head (z-score) vs rate head + IR.
                 Numbers match tab:extrap (R2 0.374 / 0.775 / -5.28).
  fig_robust  -- drop30 capacity corruption (forward-fill), CALCE:
                 corrupted input, truth, free head, rate head (IR clean).
                 Numbers match tab:robust (drop30 MAE 0.0122 vs 0.0090,
                 AE 23 vs 17).

Protocols replicate exactly:
  - test_physics_extrapolation.py  (free head, z-score, frac=0.9)
  - test_phys_irhead.py            (rate head, absolute, frac=0.9)
  - test_ablation_robust.py        (both heads, clean train, 4 modes)

Outputs paper/figures/fig_extrap.{pdf,png}, fig_robust.{pdf,png};
predictions cached in results/phys_figs.npz so replotting never
retrains.
"""
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

sys.path.insert(0, '.')
from gdn_model import build_gdn_model, masked_mae
from make_figures import load_series
from load_datasets import load_calce_cells_multivar
from test_ablation_robust import corrupt

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
W, BATCH, EPOCHS, SEED = 64, 64, 100, 42
EPS = 1e-6
IR_IDX = 3
OUT = "../paper/figures/"
NPA = "results/phys_figs.npz"

# matplotlib 3.11 tight-bbox bug guard: explicit rcParams after imports
plt.rcParams.update({"savefig.bbox": None, "figure.dpi": 150,
                     "font.size": 8})


def load_data():
    caps, train_cells, test_cell, W, sps, eol_ah = load_series("calce")
    caps_all, feats_all, _ = load_calce_cells_multivar()
    caps = {c: caps_all[c].astype(np.float32) for c in caps_all}
    feats = {c: feats_all[c].astype(np.float32) for c in feats_all}
    all_c = np.concatenate([caps[c] for c in train_cells])
    lo_c, hi_c = float(all_c.min()), float(all_c.max())
    all_i = np.concatenate([feats[c][:, IR_IDX] for c in train_cells])
    lo_i, hi_i = float(all_i.min()), float(all_i.max())
    return caps, feats, train_cells, test_cell, lo_c, hi_c, lo_i, hi_i, eol_ah


def train_model(kind, frac, caps, feats, train_cells,
                lo_c, hi_c, lo_i, hi_i):
    """kind: 'phys_ir' (absolute rate head, IR input) |
    'direct_z' (free head, z-score target). frac: train fraction."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    use_ir = kind == "phys_ir"
    zscore = kind == "direct_z"
    model = build_gdn_model(
        multiscale=False, input_dim=2 if use_ir else 1, window_size=W,
        output_len=1, readout="phys_ir" if use_ir else "last",
    ).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    X, Y = [], []
    for c in train_cells:
        seq = (caps[c] - lo_c) / (hi_c - lo_c + EPS)
        ir = ((feats[c][:, IR_IDX] - lo_i) / (hi_i - lo_i + EPS)
              if use_ir else None)
        cut = int(len(seq) * frac)
        for i in range(W, cut):
            if use_ir:
                X.append(np.stack([seq[i - W:i], ir[i - W:i]], axis=1))
            else:
                X.append(seq[i - W:i, None])
            Y.append(seq[i])
    X = np.stack(X).astype(np.float32)
    Y = np.array(Y, dtype=np.float32)
    N = len(X)
    print(f"training {kind} frac={frac} (N={N}) ...", flush=True)
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(N)
        for s in range(0, N, BATCH):
            idx = perm[s:s + BATCH]
            x = torch.tensor(X[idx]).to(DEV)
            y = torch.tensor(Y[idx]).to(DEV)
            opt.zero_grad()
            pred = model(x).squeeze(-1)
            if zscore:
                wmean = x[:, :, 0].mean(dim=1)
                wstd = x[:, :, 0].std(dim=1) + EPS
                tgt = (y - wmean) / wstd
            else:
                tgt = y
            loss = masked_mae(pred, tgt, torch.ones_like(y))
            loss.backward()
            opt.step()
        if ep % 25 == 0:
            print(f"    ep{ep} loss={loss.item():.4f}", flush=True)
    return model


def predict(model, kind, start, tc, ti):
    """Sliding-window prediction from `start` on capacity series tc
    (possibly corrupted); IR series ti stays clean. Returns absolute
    normalized-capacity predictions (phys_ir outputs absolute;
    direct_z is de-normalized per window)."""
    model.eval()
    use_ir = kind == "phys_ir"
    zscore = kind == "direct_z"
    seg_p = []
    with torch.no_grad():
        for i in range(start, len(tc)):
            if use_ir:
                win = np.stack([tc[i - W:i], ti[i - W:i]], axis=1)
            else:
                win = tc[i - W:i, None]
            cin = torch.tensor(win, dtype=torch.float32).unsqueeze(0).to(DEV)
            if zscore:
                wmean = float(win[:, 0].mean())
                wstd = float(win[:, 0].std()) + EPS
                seg_p.append(model(cin).item() * wstd + wmean)
            else:
                seg_p.append(model(cin).item())
    return np.array(seg_p)


def plot_extrap(x, truth, last, free, rate):
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 3.2))
    ax.axvline(x[0], color="gray", ls="--", lw=0.8)
    ax.plot(x, truth, color="black", lw=1.3, label="true (unseen tail)")
    ax.plot(x, last, color="gray", ls=":", lw=1.0,
            label="last-value continuation")
    ax.plot(x, free, color="tab:red", lw=1.0,
            label="free head (R$^2$=0.374)")
    ax.plot(x, rate, color="tab:green", lw=1.2,
            label="rate head + IR (R$^2$=0.775)")
    ax.set_xlabel("cycle")
    ax.set_ylabel("normalized capacity")
    ax.set_title("CALCE unseen-tail extrapolation (train 90%, eval 10%)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT + "fig_extrap.pdf")
    fig.savefig(OUT + "fig_extrap.png", dpi=150)
    plt.close(fig)
    print("saved fig_extrap", flush=True)


def plot_robust(x, truth, corr, free, rate, eol_th):
    cross = int(np.argmax(truth < eol_th))
    z0, z1 = max(0, cross - 60), min(len(truth), cross + 60)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].plot(x, truth, color="black", lw=1.2, label="true")
    axes[0].plot(x, corr, color="gray", lw=0.7, alpha=0.7,
                 label="corrupted input (drop30)")
    axes[0].plot(x, free, color="tab:red", lw=1.0, alpha=0.9,
                 label="free head (AE 23)")
    axes[0].plot(x, rate, color="tab:green", lw=1.2,
                 label="rate head + IR (AE 17)")
    axes[0].axhline(eol_th, color="red", ls=":", lw=0.7)
    axes[0].set_title("drop30: full trajectory")
    axes[0].set_xlabel("cycle")
    axes[0].set_ylabel("normalized capacity")
    axes[0].legend(fontsize=6)
    axes[1].plot(x[z0:z1], truth[z0:z1], color="black", lw=1.3,
                 label="true", marker="o", ms=2)
    axes[1].plot(x[z0:z1], corr[z0:z1], color="gray", lw=0.7,
                 alpha=0.7, label="corrupted input")
    axes[1].plot(x[z0:z1], free[z0:z1], color="tab:red", lw=1.0,
                 label="free head")
    axes[1].plot(x[z0:z1], rate[z0:z1], color="tab:green", lw=1.2,
                 label="rate head + IR")
    axes[1].axhline(eol_th, color="red", ls=":", lw=0.7)
    axes[1].set_title("drop30: EOL-crossing zoom")
    axes[1].set_xlabel("cycle")
    axes[1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(OUT + "fig_robust.pdf")
    fig.savefig(OUT + "fig_robust.png", dpi=150)
    plt.close(fig)
    print("saved fig_robust", flush=True)


if __name__ == "__main__":
    caps, feats, tr, tc, lc, hc, li, hi, eol = load_data()
    eol_th = (eol - lc) / (hc - lc + EPS)
    if os.path.exists(NPA):
        d = np.load(NPA)
        plot_extrap(d["ext_x"], d["ext_truth"], d["ext_last"],
                    d["ext_free"], d["ext_rate"])
        plot_robust(d["rob_x"], d["rob_truth"], d["rob_corr"],
                    d["rob_free"], d["rob_rate"], eol_th)
    else:
        models = {}
        models["free_frac09"] = train_model(
            "direct_z", 0.9, caps, feats, tr, lc, hc, li, hi)
        models["rate_frac09"] = train_model(
            "phys_ir", 0.9, caps, feats, tr, lc, hc, li, hi)
        models["free_clean"] = train_model(
            "direct_z", 1.0, caps, feats, tr, lc, hc, li, hi)
        models["rate_clean"] = train_model(
            "phys_ir", 1.0, caps, feats, tr, lc, hc, li, hi)
        torch.save({k: v.state_dict() for k, v in models.items()},
                   "../checkpoints/phys_figs_models.pt")

        tc_n = (caps[tc] - lc) / (hc - lc + EPS)
        ti_n = (feats[tc][:, IR_IDX] - li) / (hi - li + EPS)
        cut = int(len(tc_n) * 0.9)
        tc_c = corrupt(tc_n, "drop30")
        np.savez(
            NPA,
            ext_x=np.arange(cut, len(tc_n)),
            ext_truth=tc_n[cut:],
            ext_last=np.full(len(tc_n) - cut, tc_n[cut - 1]),
            ext_free=predict(models["free_frac09"], "direct_z", cut,
                             tc_n, ti_n),
            ext_rate=predict(models["rate_frac09"], "phys_ir", cut,
                             tc_n, ti_n),
            rob_x=np.arange(W, len(tc_n)),
            rob_truth=tc_n[W:],
            rob_corr=tc_c[W:],
            rob_free=predict(models["free_clean"], "direct_z", W,
                             tc_c, ti_n),
            rob_rate=predict(models["rate_clean"], "phys_ir", W,
                             tc_c, ti_n),
        )
        d = np.load(NPA)
        plot_extrap(d["ext_x"], d["ext_truth"], d["ext_last"],
                    d["ext_free"], d["ext_rate"])
        plot_robust(d["rob_x"], d["rob_truth"], d["rob_corr"],
                    d["rob_free"], d["rob_rate"], eol_th)
