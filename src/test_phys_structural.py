"""Structural-physics smoke test (PiDDM-style readout) vs direct head.

GDN state -> kinetic exponents (a1, a2) -> rate r(n) = -(k1 n^a1 + k2 n^a2)
-> Q_next = Q_last + r. Monotonic fade is structural (no penalty term).

2 readouts x 2 protocols, CALCE single-branch, 1 seed, 100 epochs:
  - readout "last" (direct head) vs "phys" (structural physics)
  - standard (full-window train, full test eval) and extrapolation
    (train first 90%, eval unseen final 10%)
Per-window target z-score, PatchFormer-consistent. n = normalized cycle
age of the prediction point, (i+1)/len(seq).
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


def build_windows(caps, train_cells, lo, hi, frac=1.0):
    """X: (N, W, 1) capacity; Y: (N,) next capacity; Nt: (N,) cycle age."""
    X, Y, Nt = [], [], []
    for c in train_cells:
        seq = (caps[c] - lo) / (hi - lo + EPS)
        cut = int(len(seq) * frac)
        for i in range(W, cut):
            X.append(seq[i - W:i, None])
            Y.append(seq[i])
            Nt.append((i + 1) / len(seq))
    return (np.stack(X).astype(np.float32),
            np.array(Y, dtype=np.float32),
            np.array(Nt, dtype=np.float32))


def train(readout, frac):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    caps, train_cells, test_cell, W, sps, eol_ah = load_series("calce")
    all_tr = np.concatenate([caps[c] for c in train_cells])
    lo, hi = all_tr.min(), all_tr.max()

    model = build_gdn_model(
        multiscale=False, input_dim=1, window_size=W, output_len=1,
        readout=readout,
    ).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    X, Y, Nt = build_windows(caps, train_cells, lo, hi, frac)
    N = len(X)
    print(f"training readout={readout} frac={frac} (N={N}) ...", flush=True)
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(N)
        for s in range(0, N, BATCH):
            idx = perm[s:s + BATCH]
            x = torch.tensor(X[idx]).to(DEV)
            y = torch.tensor(Y[idx]).to(DEV)
            opt.zero_grad()
            if readout == "phys":
                n = torch.tensor(Nt[idx]).to(DEV)
                pred = model(x, n=n).squeeze(-1)
            else:
                pred = model(x).squeeze(-1)
            wmean = x[:, :, 0].mean(dim=1)
            wstd = x[:, :, 0].std(dim=1) + EPS
            tgt_norm = (y - wmean) / wstd
            loss = masked_mae(pred, tgt_norm, torch.ones_like(y))
            loss.backward()
            opt.step()
        if ep % 10 == 0:
            print(f"    ep{ep} loss={loss.item():.4f}", flush=True)
    return model, caps, test_cell, lo, hi, eol_ah


def eval_std(model, caps, test_cell, lo, hi, eol_ah, readout):
    model.eval()
    tc = (caps[test_cell] - lo) / (hi - lo + EPS)
    seg_p = []
    with torch.no_grad():
        for i in range(W, len(tc)):
            win = tc[i - W:i, None]
            cin = torch.tensor(win, dtype=torch.float32).unsqueeze(0).to(DEV)
            wmean = float(win[:, 0].mean())
            wstd = float(win[:, 0].std()) + EPS
            if readout == "phys":
                n = torch.tensor([(i + 1) / len(tc)]).to(DEV)
                p = model(cin, n=n).item()
            else:
                p = model(cin).item()
            seg_p.append(p * wstd + wmean)
    seg_p = np.array(seg_p)
    tv = tc[W:]
    mae = np.mean(np.abs(seg_p - tv))
    r2 = 1 - np.sum((tv - seg_p) ** 2) / (np.sum((tv - tv.mean()) ** 2) + EPS)
    regen = np.mean(np.diff(seg_p) > 0.002)
    th = (eol_ah - lo) / (hi - lo + EPS)
    ae = abs(true_rul(tv, th) - true_rul(seg_p, th))
    print(f"  [std] readout={readout}: MAE={mae:.4f} R2={r2:.4f} "
          f"regen={regen:.3f} AE={ae}", flush=True)
    return mae, r2, regen, ae


def eval_extrap(model, caps, test_cell, lo, hi, eol_ah, readout):
    model.eval()
    tc = (caps[test_cell] - lo) / (hi - lo + EPS)
    cut = int(len(tc) * 0.9)
    seg_t = tc[cut:]
    seg_p = []
    with torch.no_grad():
        for i in range(cut, len(tc)):
            win = tc[i - W:i, None]
            cin = torch.tensor(win, dtype=torch.float32).unsqueeze(0).to(DEV)
            wmean = float(win[:, 0].mean())
            wstd = float(win[:, 0].std()) + EPS
            if readout == "phys":
                n = torch.tensor([(i + 1) / len(tc)]).to(DEV)
                p = model(cin, n=n).item()
            else:
                p = model(cin).item()
            seg_p.append(p * wstd + wmean)
    seg_p = np.array(seg_p)
    mae = np.mean(np.abs(seg_p - seg_t))
    r2 = 1 - np.sum((seg_t - seg_p) ** 2) / (np.sum((seg_t - seg_t.mean()) ** 2) + EPS)
    regen = np.mean(np.diff(seg_p) > 0.002)
    th = (eol_ah - lo) / (hi - lo + EPS)
    ae = abs(true_rul(seg_t, th) - true_rul(seg_p, th))
    print(f"  [extrap] readout={readout}: extR2={r2:.4f} extMAE={mae:.4f} "
          f"regen={regen:.3f} AE={ae}", flush=True)
    return r2, mae, regen, ae


if __name__ == "__main__":
    for readout in ["last", "phys"]:
        m, caps, tc, lo, hi, eol = train(readout, frac=1.0)
        eval_std(m, caps, tc, lo, hi, eol, readout)
        m, caps, tc, lo, hi, eol = train(readout, frac=0.9)
        eval_extrap(m, caps, tc, lo, hi, eol, readout)
