"""Diagnose the extrapolation divergence: is frac=0.9 the trigger, or luck?

Runs the exact same training loop as test_physics_extrapolation.py for
frac in {0.9, 1.0} x seeds {42, 43}, 40 epochs, and reports loss + grad
norm every 5 epochs. Answers:
  - frac=0.9 diverges reproducibly -> data distribution is the trigger
  - frac=1.0 also diverges -> random init/shuffle luck, needs stabilization
"""
import sys
import numpy as np
import torch

sys.path.insert(0, '.')
from gdn_model import build_gdn_model, masked_mae
from make_figures import load_series

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
W, BATCH, EPOCHS, EPS = 64, 64, 40, 1e-6


def build_windows(caps, train_cells, lo, hi, frac):
    X, Y = [], []
    for c in train_cells:
        seq = (caps[c] - lo) / (hi - lo + EPS)
        cut = int(len(seq) * frac)
        for i in range(W, cut):
            X.append(seq[i - W:i])
            Y.append(seq[i])
    return np.stack(X).astype(np.float32), np.array(Y, dtype=np.float32)


def run(frac, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    caps, train_cells, test_cell, W, sps, eol_ah = load_series("calce")
    all_tr = np.concatenate([caps[c] for c in train_cells])
    lo, hi = all_tr.min(), all_tr.max()
    X, Y = build_windows(caps, train_cells, lo, hi, frac)
    N = len(X)
    model = build_gdn_model(
        multiscale=False, input_dim=1, window_size=W, output_len=1,
        readout="last",
    ).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    diverge_at = None
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(N)
        for s in range(0, N, BATCH):
            idx = perm[s:s + BATCH]
            x = torch.tensor(X[idx]).unsqueeze(-1).to(DEV)
            y = torch.tensor(Y[idx]).to(DEV)
            opt.zero_grad()
            pred = model(x).squeeze(-1)
            wmean = x.squeeze(-1).mean(dim=1)
            wstd = x.squeeze(-1).std(dim=1) + EPS
            tgt_norm = (y - wmean) / wstd
            loss = masked_mae(pred, tgt_norm, torch.ones_like(y))
            loss.backward()
            gnorm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
            opt.step()
            if ep % 5 == 0 and s == 0:
                print(f"  frac={frac} seed={seed} ep{ep} loss={loss.item():.4f} gnorm={gnorm:.3f}", flush=True)
            if torch.isnan(loss) or loss.item() > 100:
                diverge_at = ep
                print(f"  frac={frac} seed={seed} DIVERGED at ep{ep} loss={loss.item():.4f}", flush=True)
                return "diverge"
    return "ok"


if __name__ == "__main__":
    for frac in [0.9, 1.0]:
        for seed in [42, 43]:
            res = run(frac, seed)
            print(f"==> frac={frac} seed={seed}: {res}", flush=True)
