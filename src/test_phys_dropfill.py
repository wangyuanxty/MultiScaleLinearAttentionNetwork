"""Physics-assisted corruption handling v3: NASA with T + Re/Rct.

drop30 on the test sequence; dropped segments filled by:
  (a) forward-fill (biased upward on a declining series)
  (b) local-rate fill (constant smoothed rate, no features)
  (c) physics-fill: r = c0 + c1*T + c2*Re + c3*Rct fitted on clean
      history — T/Re/Rct are measured independently of capacity, so
      they survive capacity data loss
Then feed each filled sequence to the trained GDN and also report
physics-takeover (physics predictions during drops, clean data
elsewhere, no GDN).

NASA: train B0006/7/18, test B0005, W=30, seed 42, 100 epochs.
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
SMOOTH = 10
FEAT_COLS = ["T_mean", "Re", "Rct"]


def build_windows(caps, train_cells, lo, hi):
    X, Y = [], []
    for c in train_cells:
        seq = (caps[c] - lo) / (hi - lo + EPS)
        for i in range(W, len(seq)):
            X.append(seq[i - W:i, None])
            Y.append(seq[i])
    return np.stack(X).astype(np.float32), np.array(Y, dtype=np.float32)


def train_model():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    caps, train_cells, test_cell, W, sps, eol_ah = load_series("nasa")
    all_tr = np.concatenate([caps[c] for c in train_cells])
    lo, hi = all_tr.min(), all_tr.max()
    model = build_gdn_model(
        multiscale=False, input_dim=1, window_size=W, output_len=1,
        readout="last",
    ).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    X, Y = build_windows(caps, train_cells, lo, hi)
    N = len(X)
    print(f"training NASA [C] (N={N}) ...", flush=True)
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


def load_feats(test_cell):
    d = load_nasa_multivar(test_cell)
    feats = np.stack([d[k].astype(np.float64) for k in FEAT_COLS], axis=1)
    # guard NaNs from EIS interpolation edges
    for j in range(feats.shape[1]):
        col = feats[:, j]
        mask = np.isnan(col)
        if mask.any():
            idx = np.where(~mask)[0]
            if len(idx):
                col[mask] = np.interp(np.where(mask)[0], idx, col[idx])
            else:
                col[:] = 0.0
    return feats


def drop_mask(n, rng):
    seg = 8
    n_drop = int(n * 0.30 / seg)
    starts = rng.choice(np.arange(0, n - seg, seg), n_drop, replace=False)
    mask = np.zeros(n, dtype=bool)
    for st in starts:
        mask[st:st + seg] = True
    return mask


def smooth_rates(q):
    r = (q[:-5] - q[5:]) / 5.0
    r = np.concatenate([r, np.full(5, r[-1])])
    kernel = np.ones(SMOOTH) / SMOOTH
    return np.convolve(r, kernel, mode="same")


def forward_fill(tc, mask):
    tc = tc.copy()
    for i in range(1, len(tc)):
        if mask[i]:
            tc[i] = tc[i - 1]
    return tc


def rate_fill(tc, mask, rate_fn):
    tc = tc.copy()
    i = 1
    while i < len(tc):
        if mask[i]:
            while i < len(tc) and mask[i]:
                tc[i] = tc[i - 1] - max(rate_fn(i), 0.0)
                i += 1
        else:
            i += 1
    return tc


def eval_seq(model, tc_filled, tc_clean, lo, hi, eol_ah):
    model.eval()
    seg_p = []
    with torch.no_grad():
        for i in range(W, len(tc_filled)):
            win = tc_filled[i - W:i, None]
            cin = torch.tensor(win, dtype=torch.float32).unsqueeze(0).to(DEV)
            wmean = float(win[:, 0].mean())
            wstd = float(win[:, 0].std()) + EPS
            seg_p.append(model(cin).item() * wstd + wmean)
    seg_p = np.array(seg_p)
    tv = tc_clean[W:]
    mae = np.mean(np.abs(seg_p - tv))
    r2 = 1 - np.sum((tv - seg_p) ** 2) / (np.sum((tv - tv.mean()) ** 2) + EPS)
    th = (eol_ah - lo) / (hi - lo + EPS)
    ae = abs(true_rul(tv, th) - true_rul(seg_p, th))
    return mae, r2, ae


def main():
    model, caps, test_cell, lo, hi, eol_ah = train_model()
    tc_clean = (caps[test_cell] - lo) / (hi - lo + EPS)
    feats = load_feats(test_cell)

    rng = np.random.RandomState(SEED)
    mask = drop_mask(len(tc_clean), rng)
    print(f"NASA {test_cell}: dropped {mask.sum()}/{len(mask)} points "
          f"in {mask.sum()//8} segments", flush=True)

    r_all = smooth_rates(tc_clean)
    clean_idx = np.where(~mask)[0]

    # local-rate (no features)
    r_loc = float(np.mean(r_all[clean_idx]))

    # physics: r = c0 + c1*T + c2*Re + c3*Rct (multivariate linear)
    A = np.stack([np.ones_like(clean_idx, dtype=np.float64)]
                 + [feats[clean_idx, j] for j in range(len(FEAT_COLS))], axis=1)
    coef, *_ = np.linalg.lstsq(A, r_all[clean_idx], rcond=None)
    print(f"rate models: local r={r_loc:.5f} | physics coefs="
          f"{[f'{c:.4f}' for c in coef]}", flush=True)

    # (a) forward-fill
    tf = forward_fill(tc_clean, mask)
    mae, r2, ae = eval_seq(model, tf, tc_clean, lo, hi, eol_ah)
    print(f"  forward-fill (GDN): MAE={mae:.4f} R2={r2:.4f} AE={ae}", flush=True)

    # (b) local-rate fill
    tl = rate_fill(tc_clean, mask, lambda i: r_loc)
    mae, r2, ae = eval_seq(model, tl, tc_clean, lo, hi, eol_ah)
    print(f"  local-fill  (GDN): MAE={mae:.4f} R2={r2:.4f} AE={ae}", flush=True)

    # (c) physics-fill (T + Re + Rct)
    def phys_rate(i):
        return coef[0] + sum(coef[j + 1] * feats[i, j]
                             for j in range(len(FEAT_COLS)))
    tp = rate_fill(tc_clean, mask, phys_rate)
    mae, r2, ae = eval_seq(model, tp, tc_clean, lo, hi, eol_ah)
    print(f"  phys-fill   (GDN): MAE={mae:.4f} R2={r2:.4f} AE={ae}", flush=True)

    # physics-takeover: physics predictions during drops, clean elsewhere
    take = tc_clean.copy()
    take[mask] = tp[mask]
    th = (eol_ah - lo) / (hi - lo + EPS)
    ae_t = abs(true_rul(tc_clean[W:], th) - true_rul(take[W:], th))
    mae_t = np.mean(np.abs(take - tc_clean))
    print(f"  phys-takeover (no GDN): MAE={mae_t:.4f} AE={ae_t}", flush=True)


if __name__ == "__main__":
    main()
