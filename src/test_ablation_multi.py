"""Multiscale ablation (no physics, direct head, z-score protocol).

B2: multi   — 3 branches (patch 2/4/8), no cross-scale interaction
B3: stage   — 3 branches + StageQuery V3 cross-exchange
Single-branch reference B1 (direct-z): MAE=0.0059 R2=0.9963 AE=2.

CALCE, seed 42, 100 epochs. Per-window z-score targets (direct-head
protocol, PatchFormer-consistent); eval with per-window de-normalization.
"""
import sys
import numpy as np
import torch

sys.path.insert(0, '.')
from gdn_model import build_gdn_model, masked_mae
from make_figures import load_series
from eval_multiseed import true_rul

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
W, BATCH, EPOCHS, SEED = 64, 64, 100, 42
EPS = 1e-6


def build_windows(caps, train_cells, lo, hi):
    X, Y = [], []
    for c in train_cells:
        seq = (caps[c] - lo) / (hi - lo + EPS)
        for i in range(W, len(seq)):
            X.append(seq[i - W:i, None])
            Y.append(seq[i])
    return np.stack(X).astype(np.float32), np.array(Y, dtype=np.float32)


def run(stage_query):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    caps, train_cells, test_cell, W, sps, eol_ah = load_series("calce")
    all_tr = np.concatenate([caps[c] for c in train_cells])
    lo, hi = all_tr.min(), all_tr.max()

    model = build_gdn_model(
        multiscale=True, stage_query=stage_query,
        input_dim=1, window_size=W, output_len=1, readout="last",
    ).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    X, Y = build_windows(caps, train_cells, lo, hi)
    N = len(X)
    tag = "stage" if stage_query else "multi"
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

    model.eval()
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
    mae = np.mean(np.abs(seg_p - tv))
    r2 = 1 - np.sum((tv - seg_p) ** 2) / (np.sum((tv - tv.mean()) ** 2) + EPS)
    regen = np.mean(np.diff(seg_p) > 0.002)
    th = (eol_ah - lo) / (hi - lo + EPS)
    ae = abs(true_rul(tv, th) - true_rul(seg_p, th))
    print(f"  [{tag}] MAE={mae:.4f} R2={r2:.4f} regen={regen:.3f} AE={ae}",
          flush=True)
    return mae, r2, regen, ae


if __name__ == "__main__":
    for sq in [False, True]:
        run(sq)
    print("reference: single direct-z MAE=0.0059 R2=0.9963 AE=2 | "
          "phys_ir single MAE=0.0042", flush=True)
