"""Physics-feature rate head test: r = softplus(w·h) + softplus(gamma)·IR.

The ONE chosen mechanism for physics-in-model: a rate head inside the
computation graph, driven by the learned representation AND the physics
feature IR (positive gamma — IR only accelerates decay). Structural
monotonicity is free (Q_hat = Q_last - r).

Trained in ABSOLUTE space (protocol-hypothesis fix: no per-window
z-score — the z-target sign mismatch that killed the earlier structural
head does not exist here). Eval output is normalized capacity directly.

CALCE, single-branch, seed 42, 100 epochs; standard + extrapolation.
Reference (direct head, z-score protocol): MAE=0.0059 R2=0.9963 AE=2.
"""
import sys
import numpy as np
import torch

sys.path.insert(0, '.')
from gdn_model import build_gdn_model, masked_mae
from make_figures import load_series
from eval_multiseed import true_rul
from load_datasets import load_calce_cells_multivar

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
W, BATCH, EPOCHS, SEED = 64, 64, 100, 42
EPS = 1e-6
IR_IDX = 3


def build_windows(caps, feats, cells, lo_c, hi_c, lo_i, hi_i, frac=1.0):
    X, Y = [], []
    for c in cells:
        seq = (caps[c] - lo_c) / (hi_c - lo_c + EPS)
        ir = (feats[c][:, IR_IDX] - lo_i) / (hi_i - lo_i + EPS)
        L = len(seq)
        cut = int(L * frac)
        for i in range(W, cut):
            X.append(np.stack([seq[i - W:i], ir[i - W:i]], axis=1))
            Y.append(seq[i])
    return np.stack(X).astype(np.float32), np.array(Y, dtype=np.float32)


def train(frac):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    caps, train_cells, test_cell, W, sps, eol_ah = load_series("calce")
    caps_all, feats_all, _ = load_calce_cells_multivar()
    caps = {c: caps_all[c].astype(np.float32) for c in caps_all}
    feats = {c: feats_all[c].astype(np.float32) for c in feats_all}

    all_c = np.concatenate([caps[c] for c in train_cells])
    lo_c, hi_c = float(all_c.min()), float(all_c.max())
    all_i = np.concatenate([feats[c][:, IR_IDX] for c in train_cells])
    lo_i, hi_i = float(all_i.min()), float(all_i.max())

    model = build_gdn_model(
        multiscale=False, input_dim=2, window_size=W, output_len=1,
        readout="phys_ir",
    ).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    X, Y = build_windows(caps, feats, train_cells,
                         lo_c, hi_c, lo_i, hi_i, frac)
    N = len(X)
    print(f"training phys_ir frac={frac} (N={N}) ...", flush=True)
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(N)
        for s in range(0, N, BATCH):
            idx = perm[s:s + BATCH]
            x = torch.tensor(X[idx]).to(DEV)
            y = torch.tensor(Y[idx]).to(DEV)
            opt.zero_grad()
            pred = model(x).squeeze(-1)  # absolute normalized capacity
            loss = masked_mae(pred, y, torch.ones_like(y))
            loss.backward()
            opt.step()
        if ep % 10 == 0:
            print(f"    ep{ep} loss={loss.item():.4f}", flush=True)
    return model, caps, feats, test_cell, lo_c, hi_c, lo_i, hi_i, eol_ah


def eval_std(model, caps, feats, test_cell, lo_c, hi_c, lo_i, hi_i, eol_ah):
    model.eval()
    tc = (caps[test_cell] - lo_c) / (hi_c - lo_c + EPS)
    ti = (feats[test_cell][:, IR_IDX] - lo_i) / (hi_i - lo_i + EPS)
    seg_p = []
    with torch.no_grad():
        for i in range(W, len(tc)):
            win = np.stack([tc[i - W:i], ti[i - W:i]], axis=1)
            cin = torch.tensor(win, dtype=torch.float32).unsqueeze(0).to(DEV)
            seg_p.append(model(cin).item())  # absolute output, no de-norm
    seg_p = np.array(seg_p)
    tv = tc[W:]
    mae = np.mean(np.abs(seg_p - tv))
    r2 = 1 - np.sum((tv - seg_p) ** 2) / (np.sum((tv - tv.mean()) ** 2) + EPS)
    regen = np.mean(np.diff(seg_p) > 0.002)
    th = (eol_ah - lo_c) / (hi_c - lo_c + EPS)
    ae = abs(true_rul(tv, th) - true_rul(seg_p, th))
    print(f"  [std] phys_ir: MAE={mae:.4f} R2={r2:.4f} "
          f"regen={regen:.3f} AE={ae}", flush=True)
    return mae, r2, regen, ae


def eval_extrap(model, caps, feats, test_cell, lo_c, hi_c, lo_i, hi_i, eol_ah):
    model.eval()
    tc = (caps[test_cell] - lo_c) / (hi_c - lo_c + EPS)
    ti = (feats[test_cell][:, IR_IDX] - lo_i) / (hi_i - lo_i + EPS)
    cut = int(len(tc) * 0.9)
    seg_t = tc[cut:]
    seg_p = []
    with torch.no_grad():
        for i in range(cut, len(tc)):
            win = np.stack([tc[i - W:i], ti[i - W:i]], axis=1)
            cin = torch.tensor(win, dtype=torch.float32).unsqueeze(0).to(DEV)
            seg_p.append(model(cin).item())
    seg_p = np.array(seg_p)
    r2 = 1 - np.sum((seg_t - seg_p) ** 2) / (np.sum((seg_t - seg_t.mean()) ** 2) + EPS)
    mae = np.mean(np.abs(seg_t - seg_p))
    regen = np.mean(np.diff(seg_p) > 0.002)
    print(f"  [extrap] phys_ir: extR2={r2:.4f} extMAE={mae:.4f} "
          f"regen={regen:.3f}", flush=True)
    return r2, mae, regen


if __name__ == "__main__":
    m, caps, feats, tc, lc, hc, li, hi, eol = train(frac=1.0)
    eval_std(m, caps, feats, tc, lc, hc, li, hi, eol)
    m, caps, feats, tc, lc, hc, li, hi, eol = train(frac=0.9)
    eval_extrap(m, caps, feats, tc, lc, hc, li, hi, eol)
    print("reference (direct head, z-score): MAE=0.0059 R2=0.9963 AE=2 "
          "regen=0.248 | extrap n-ch 0.4471", flush=True)
