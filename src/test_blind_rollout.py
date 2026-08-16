"""Blind prognostics fair comparison: GDN self-fed rollout vs physics.

The previous extrapolation eval fed TRUE tail values into sliding
windows — not a blind task. Here both contenders run blind from the
90% observation point:

  - GDN: 89-step self-fed rollout (window = own predictions)
  - physics: M1/M2/M3 integrated rate models, fitted on observed 90%

CALCE, [C,n] single-branch, seed 42, 100 epochs (train frac=0.9).
"""
import sys
import numpy as np
import torch

sys.path.insert(0, '.')
from gdn_model import build_gdn_model, masked_mae
from make_figures import load_series
from scipy.optimize import curve_fit

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
W, BATCH, EPOCHS, SEED = 64, 64, 100, 42
EPS = 1e-6


def build_windows(caps, train_cells, lo, hi, frac=0.9):
    X, Y = [], []
    for c in train_cells:
        seq = (caps[c] - lo) / (hi - lo + EPS)
        L = len(seq)
        cut = int(L * frac)
        for i in range(W, cut):
            win_c = seq[i - W:i, None]
            win_n = ((np.arange(i - W, i) + 1) / L)[:, None].astype(np.float32)
            X.append(np.concatenate([win_c, win_n], axis=1))
            Y.append(seq[i])
    return np.stack(X).astype(np.float32), np.array(Y, dtype=np.float32)


def train_model():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    caps, train_cells, test_cell, W, sps, eol_ah = load_series("calce")
    all_tr = np.concatenate([caps[c] for c in train_cells])
    lo, hi = all_tr.min(), all_tr.max()
    model = build_gdn_model(
        multiscale=False, input_dim=2, window_size=W, output_len=1,
        readout="last",
    ).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    X, Y = build_windows(caps, train_cells, lo, hi)
    N = len(X)
    print(f"training [C,n] frac=0.9 (N={N}) ...", flush=True)
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
        if ep % 25 == 0:
            print(f"    ep{ep} loss={loss.item():.4f}", flush=True)
    return model, caps, test_cell, lo, hi, eol_ah


def gdn_blind_rollout(model, tc, L, cut):
    """Self-fed rollout from cut; window = own predictions after warmup."""
    model.eval()
    win = tc[cut - W:cut]  # (W,) last observed
    seg_p = []
    with torch.no_grad():
        for i in range(cut, L):
            n = ((np.arange(i - W, i) + 1) / L)[:, None].astype(np.float32)
            cin = torch.tensor(
                np.concatenate([win[:, None], n], axis=1),
                dtype=torch.float32).unsqueeze(0).to(DEV)
            wmean = float(win.mean())
            wstd = float(win.std()) + EPS
            p = model(cin).item() * wstd + wmean
            seg_p.append(p)
            win = np.concatenate([win[1:], [p]])
    return np.array(seg_p)


def fit_m2(n, q):
    def f(n_, q0, lk1, b1, lk2, b2):
        return q0 - np.exp(lk1) * n_**b1 - np.exp(lk2) * n_**b2
    try:
        (q0, lk1, b1, lk2, b2), _ = curve_fit(
            f, n, q, p0=[q[0], -7.0, 0.5, -7.0, 2.0], maxfev=40000)
        return q0, np.exp(lk1), b1, np.exp(lk2), b2
    except Exception:
        return None


def report(name, q_true, q_pred, cut):
    seg_t = q_true[cut:]
    r2 = 1 - np.sum((seg_t - q_pred) ** 2) / (np.sum((seg_t - seg_t.mean()) ** 2) + EPS)
    mae = np.mean(np.abs(seg_t - q_pred))
    print(f"  {name}: extR2={r2:.4f} extMAE={mae:.5f}", flush=True)
    return r2


def main():
    model, caps, test_cell, lo, hi, eol_ah = train_model()
    q = caps[test_cell].astype(np.float64)
    L = len(q)
    cut = int(L * 0.9)
    tc = (q - lo) / (hi - lo + EPS)  # normalized for GDN
    n_all = (np.arange(L) + 1) / L
    n_obs, q_obs = n_all[:cut], q[:cut]
    n_tail = n_all[cut:]

    # GDN blind rollout (in normalized space, then back to Ah)
    seg_n = gdn_blind_rollout(model, tc, L, cut)
    seg_ah = seg_n * (hi - lo + EPS) + lo
    report("GDN blind rollout", q, seg_ah, cut)

    # naive last-value
    report("last-value", q, np.full(L - cut, q_obs[-1]), cut)

    # physics M2 integrated
    res2 = fit_m2(n_obs, q_obs)
    if res2:
        q0, K1, b1, K2, b2 = res2
        q2 = q0 - K1 * n_tail ** b1 - K2 * n_tail ** b2
        report(f"M2 (b1={b1:.2f}, b2={b2:.2f})", q, q2, cut)

    # physics M1
    def f1(n_, q0, lk, b):
        return q0 - np.exp(lk) * n_**b
    (q0, lk, b), _ = curve_fit(f1, n_obs, q_obs, p0=[q[0], -7.0, 1.5], maxfev=20000)
    q1 = q0 - np.exp(lk) * n_tail ** b
    report(f"M1 (b={b:.2f})", q, q1, cut)


if __name__ == "__main__":
    main()
