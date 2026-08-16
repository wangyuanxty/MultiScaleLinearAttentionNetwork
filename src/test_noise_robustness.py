"""Noise-robustness ablation (damaged-test direction): train ONCE per
physics config on clean data, evaluate on three corrupted test
sequences. Does the physics regularizer make the single-branch model
more robust to sensor aging / data loss?

Corruption types (test sequence only):
  - drop 30% segments (power/packet loss)
  - Gaussian white noise (sensor noise)
  - impulse noise (sensor glitches)
CALCE, 1 seed, per-window target normalization pipeline.
"""
import sys
import numpy as np
import torch

sys.path.insert(0, '.')
from gdn_model import build_gdn_model, masked_mae, PhysicsRegularizer
from make_figures import load_series

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
W, BATCH, EPOCHS, SEED = 64, 64, 100, 42
EPS = 1e-6


def build_windows(caps, train_cells, lo, hi, K=1):
    X, Y = [], []
    for c in train_cells:
        seq = (caps[c] - lo) / (hi - lo + EPS)
        for i in range(W, len(seq) - K + 1):
            X.append(seq[i - W:i])
            Y.append(seq[i:i + K])
    return np.stack(X).astype(np.float32), np.stack(Y).astype(np.float32)


def corrupt(tc, mode):
    rng = np.random.RandomState(SEED)
    tc = tc.copy()
    if mode == "drop30":
        n = len(tc)
        seg = 8
        n_drop = int(n * 0.30 / seg)
        starts = rng.choice(np.arange(0, n - seg, seg), n_drop, replace=False)
        for st in starts:
            tc[st:st + seg] = np.nan
        for i in range(1, n):  # forward-fill (BMS holds last reading)
            if np.isnan(tc[i]):
                tc[i] = tc[i - 1]
        if np.isnan(tc[0]):  # dropped segment may start at index 0
            first_ok = int(np.argmax(~np.isnan(tc)))
            tc[:first_ok] = tc[first_ok]
    elif mode == "gauss":
        tc = tc + rng.normal(0, 0.01, size=len(tc))  # ~1% capacity noise
    elif mode == "impulse":
        n = len(tc)
        idx = rng.choice(n, int(n * 0.05), replace=False)  # 5% glitches
        tc[idx] += rng.normal(0, 0.05, size=len(idx))
    return tc


def train_model(use_phys):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    caps, train_cells, test_cell, W, sps, eol_ah = load_series("calce")
    all_tr = np.concatenate([caps[c] for c in train_cells])
    lo, hi = all_tr.min(), all_tr.max()

    model = build_gdn_model(
        multiscale=False, input_dim=1, window_size=W, output_len=1,
        readout="last",
    ).to(DEV)
    phys_reg = PhysicsRegularizer(lambda_=0.1).to(DEV) if use_phys else None
    params = list(model.parameters())
    if phys_reg:
        params += list(phys_reg.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)

    X_tr, Y_tr = build_windows(caps, train_cells, lo, hi)
    N = len(X_tr)
    ir_mean = None
    if use_phys:
        from load_datasets import load_calce_cells_multivar
        _, feats_all, _ = load_calce_cells_multivar()
        ir_vals = [feats_all[c][:, 3].astype(np.float32) for c in train_cells]
        ir_mean = float(np.mean(np.concatenate(ir_vals)))

    print(f"training phys={use_phys} ...", flush=True)
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(N)
        for s in range(0, N, BATCH):
            idx = perm[s:s + BATCH]
            x = torch.tensor(X_tr[idx]).unsqueeze(-1).to(DEV)
            y = torch.tensor(Y_tr[idx]).to(DEV).squeeze(-1)  # (B,)
            opt.zero_grad()
            pred = model(x).squeeze(-1)
            wmean = x.squeeze(-1).mean(dim=1)
            wstd = x.squeeze(-1).std(dim=1) + EPS
            tgt_norm = (y - wmean) / wstd
            loss = masked_mae(pred, tgt_norm, torch.ones_like(y))
            if phys_reg:
                pred_abs = (pred * wstd + wmean).unsqueeze(-1)  # (B,1)
                ir_vals = torch.full_like(pred_abs, ir_mean)
                loss = loss + phys_reg(pred_abs, x[:, :, 0:1], ir_vals)
            loss.backward()
            opt.step()
        if ep % 10 == 0:
            print(f"    ep{ep} loss={loss.item():.4f}", flush=True)
    return model, caps, train_cells, test_cell, lo, hi, eol_ah


def eval_corrupted(model, caps, train_cells, test_cell, lo, hi, eol_ah, mode):
    model.eval()
    tc_clean = (caps[test_cell] - lo) / (hi - lo + EPS)
    tc = corrupt(tc_clean, mode)
    seg_p = []
    with torch.no_grad():
        for i in range(W, len(tc)):
            win = tc[i - W:i]
            wmean = float(win.mean())
            wstd = float(win.std()) + EPS
            cin = torch.tensor(win, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEV)
            seg_p.append(model(cin).item() * wstd + wmean)
    seg_p = np.array(seg_p)
    tv = tc_clean[W:]
    mae = np.mean(np.abs(seg_p - tv))
    r2 = 1 - np.sum((tv - seg_p) ** 2) / (np.sum((tv - tv.mean()) ** 2) + EPS)
    regen = np.mean(np.diff(seg_p) > 0.002)
    print(f"  [{mode}] phys eval: MAE={mae:.4f} R2={r2:.4f} regen={regen:.3f}",
          flush=True)
    return mae, r2, regen


if __name__ == "__main__":
    modes = ["clean", "drop30", "gauss", "impulse"]
    for use_phys in [False, True]:
        model, caps, tr, tc, lo, hi, eol = train_model(use_phys)
        for mode in modes:
            eval_corrupted(model, caps, tr, tc, lo, hi, eol, mode)
