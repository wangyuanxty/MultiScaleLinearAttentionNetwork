"""Physics features in training — NASA testbed, n-inclusive.

n (normalized cycle age, the degradation time variable) is the base
physics channel; T_mean and EIS Re/Rct are tested ON TOP of [C,n]:
  [C] | [C,n] | [C,n,T] | [C,n,Re,Rct] | [C,n,T,Re,Rct]
1 seed, 100 epochs, single-branch direct head.
Global per-channel min-max (train lo/hi); per-window z-score target.
"""
import sys
import numpy as np
import torch

sys.path.insert(0, '.')
from gdn_model import build_gdn_model, masked_mae
from make_figures import load_series
from eval_multiseed import true_rul
from load_datasets import load_nasa_multivar

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
W, BATCH, EPOCHS, SEED = 30, 32, 100, 42
EPS = 1e-6

FEAT_COLS = ["T_mean", "Re", "Rct"]


def load_feats(train_cells, test_cell):
    """Per-cell feature matrices, aligned with capacity arrays."""
    feats = {}
    for c in list(train_cells) + [test_cell]:
        d = load_nasa_multivar(c)
        feats[c] = np.stack([d[k].astype(np.float32) for k in FEAT_COLS], axis=1)
    return feats


def chan_values(caps, feats, cell, col):
    """Full-sequence values of one channel for a cell (capacity-normalized)."""
    if col == "n":
        return (np.arange(len(caps[cell])) + 1).astype(np.float32) / len(caps[cell])
    j = FEAT_COLS.index(col)
    return feats[cell][:, j]


def build_windows(caps, feats, cells, los, his, use_cols):
    """X: (N, W, C); Y: (N,) next normalized capacity."""
    X, Y = [], []
    for c in cells:
        seq = (caps[c] - los[0]) / (his[0] - los[0] + EPS)
        for i in range(W, len(seq)):
            win = [seq[i - W:i]]
            for k, col in enumerate(use_cols):
                f = chan_values(caps, feats, c, col)
                fn = (f - los[1 + k]) / (his[1 + k] - los[1 + k] + EPS)
                win.append(fn[i - W:i])
            X.append(np.stack(win, axis=1))
            Y.append(seq[i])
    return np.stack(X).astype(np.float32), np.array(Y, dtype=np.float32)


def run(use_cols):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    caps, train_cells, test_cell, W, sps, eol_ah = load_series("nasa")
    caps = {c: caps[c].astype(np.float32) for c in caps}
    feats = load_feats(train_cells, test_cell)

    # per-channel global lo/hi from train cells only
    los, his = [], []
    all_c = np.concatenate([caps[c] for c in train_cells])
    los.append(float(all_c.min())); his.append(float(all_c.max()))
    for col in use_cols:
        if col == "n":
            los.append(0.0); his.append(1.0)
        else:
            all_f = np.concatenate([chan_values(caps, feats, c, col)
                                    for c in train_cells])
            all_f = all_f[np.isfinite(all_f)]
            los.append(float(all_f.min())); his.append(float(all_f.max()))

    model = build_gdn_model(
        multiscale=False, input_dim=1 + len(use_cols), window_size=W,
        output_len=1, readout="last",
    ).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    X, Y = build_windows(caps, feats, train_cells, los, his, use_cols)
    N = len(X)
    tag = f"[C,{','.join(use_cols)}]" if use_cols else "[C]"
    print(f"training {tag} (N={N}) ...", flush=True)
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(N)
        for s in range(0, N, BATCH):
            idx = perm[s:s + BATCH]
            x = torch.tensor(X[idx]).to(DEV)
            y = torch.tensor(Y[idx]).to(DEV)
            opt.zero_grad()
            pred = model(x).squeeze(-1)
            wmean = x[:, :, 0].mean(dim=1)
            wstd = x[:, :, 0].std(dim=1) + EPS
            tgt_norm = (y - wmean) / wstd
            loss = masked_mae(pred, tgt_norm, torch.ones_like(y))
            loss.backward()
            opt.step()
        if ep % 10 == 0:
            print(f"    ep{ep} loss={loss.item():.4f}", flush=True)

    # ---- standard eval on test cell ----
    model.eval()
    tc = (caps[test_cell] - los[0]) / (his[0] - los[0] + EPS)
    seg_p = []
    with torch.no_grad():
        for i in range(W, len(tc)):
            win = [tc[i - W:i]]
            for k, col in enumerate(use_cols):
                f = chan_values(caps, feats, test_cell, col)
                fn = (f - los[1 + k]) / (his[1 + k] - los[1 + k] + EPS)
                win.append(fn[i - W:i])
            win = np.stack(win, axis=1)
            cin = torch.tensor(win, dtype=torch.float32).unsqueeze(0).to(DEV)
            wmean = float(win[:, 0].mean())
            wstd = float(win[:, 0].std()) + EPS
            seg_p.append(model(cin).item() * wstd + wmean)
    seg_p = np.array(seg_p)
    tv = tc[W:]
    mae = np.mean(np.abs(seg_p - tv))
    r2 = 1 - np.sum((tv - seg_p) ** 2) / (np.sum((tv - tv.mean()) ** 2) + EPS)
    regen = np.mean(np.diff(seg_p) > 0.002)
    th = (eol_ah - los[0]) / (his[0] - los[0] + EPS)
    ae = abs(true_rul(tv, th) - true_rul(seg_p, th))
    print(f"  {tag}: MAE={mae:.4f} R2={r2:.4f} regen={regen:.3f} AE={ae}",
          flush=True)
    return mae, r2, regen, ae


if __name__ == "__main__":
    for use_cols in [[], ["n"], ["n", "T_mean"],
                     ["n", "Re", "Rct"], ["n", "T_mean", "Re", "Rct"]]:
        run(use_cols)
