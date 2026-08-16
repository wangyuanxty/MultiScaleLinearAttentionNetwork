"""Physics-features-as-input comparison: [C] vs [C, IR], full matrix.

Standard protocol + noise robustness. Corruption is applied to the
CAPACITY channel only — IR stays clean, testing whether IR earns its
place as an independent corroborating signal when capacity data is
corrupted (the one niche where it could still win).

CALCE, 1 seed, 100 epochs. Inputs globally min-maxed per channel (train
lo/hi); target per-window z-score (PatchFormer-consistent).
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
IR_IDX = 3  # feat_cols: V_mean,V_min,V_max,IR,dvdt,E_d,E_c,AC,Phase,delta_t
MODES = ["clean", "drop30", "gauss", "impulse"]


def build_windows(caps, feats, cells, lo, hi, use_ir):
    """X shape (N, W, C); Y = normalized capacity (N,)."""
    X, Y = [], []
    for c in cells:
        seq = (caps[c] - lo[0]) / (hi[0] - lo[0] + EPS)
        ir = (feats[c][:, IR_IDX] - lo[1]) / (hi[1] - lo[1] + EPS) if use_ir else None
        for i in range(W, len(seq)):
            win = seq[i - W:i]
            if use_ir:
                win = np.stack([win, ir[i - W:i]], axis=1)
            else:
                win = win[:, None]  # (W, 1)
            X.append(win)
            Y.append(seq[i])
    return np.stack(X).astype(np.float32), np.array(Y, dtype=np.float32)


def corrupt(tc, mode):
    """Corrupt the capacity channel only (IR stays clean)."""
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


def train(use_ir):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    caps, train_cells, test_cell, W, sps, eol_ah = load_series("calce")
    caps_all, feats_all, _ = load_calce_cells_multivar()
    caps = {c: caps_all[c].astype(np.float32) for c in caps_all}
    feats = {c: feats_all[c].astype(np.float32) for c in feats_all}

    # per-channel global lo/hi from train cells
    all_tr = np.concatenate([caps[c] for c in train_cells])
    lo, hi = [float(all_tr.min())], [float(all_tr.max())]
    if use_ir:
        all_ir = np.concatenate([feats[c][:, IR_IDX] for c in train_cells])
        lo.append(float(all_ir.min()))
        hi.append(float(all_ir.max()))

    n_in = 2 if use_ir else 1
    model = build_gdn_model(
        multiscale=False, input_dim=n_in, window_size=W, output_len=1,
        readout="last",
    ).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    X, Y = build_windows(caps, feats, train_cells, lo, hi, use_ir)
    N = len(X)
    print(f"training use_ir={use_ir} (N={N}, in_dim={n_in}) ...", flush=True)
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
    return model, caps, feats, test_cell, lo, hi, eol_ah


def eval_one(model, caps, feats, test_cell, lo, hi, eol_ah, use_ir, mode):
    model.eval()
    tc = (caps[test_cell] - lo[0]) / (hi[0] - lo[0] + EPS)
    tc_ir = (feats[test_cell][:, IR_IDX] - lo[1]) / (hi[1] - lo[1] + EPS) if use_ir else None
    tc_cor = corrupt(tc, mode) if mode != "clean" else tc
    seg_p = []
    with torch.no_grad():
        for i in range(W, len(tc_cor)):
            win = tc_cor[i - W:i]
            if use_ir:
                win = np.stack([win, tc_ir[i - W:i]], axis=1)
            else:
                win = win[:, None]
            cin = torch.tensor(win, dtype=torch.float32).unsqueeze(0).to(DEV)
            wmean = float(win[:, 0].mean())
            wstd = float(win[:, 0].std()) + EPS
            seg_p.append(model(cin).item() * wstd + wmean)
    seg_p = np.array(seg_p)
    tv = tc[W:]  # compare against CLEAN capacity
    mae = np.mean(np.abs(seg_p - tv))
    r2 = 1 - np.sum((tv - seg_p) ** 2) / (np.sum((tv - tv.mean()) ** 2) + EPS)
    regen = np.mean(np.diff(seg_p) > 0.002)
    th = (eol_ah - lo[0]) / (hi[0] - lo[0] + EPS)
    ae = abs(true_rul(tv, th) - true_rul(seg_p, th))
    print(f"  [{mode}] use_ir={use_ir}: MAE={mae:.4f} R2={r2:.4f} "
          f"regen={regen:.3f} AE={ae}", flush=True)
    return mae, r2, regen, ae


if __name__ == "__main__":
    for use_ir in [False, True]:
        model, caps, feats, tc, lo, hi, eol = train(use_ir)
        for mode in MODES:
            eval_one(model, caps, feats, tc, lo, hi, eol, use_ir, mode)
