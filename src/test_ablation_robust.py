"""Physics robustness: phys_ir under capacity corruption (IR stays clean).

Train phys_ir once on clean data (absolute space), then corrupt the
test sequence's CAPACITY channel only — IR is measured independently
and stays clean, so the rate head can lean on it during data loss.

Modes: clean / drop30 (forward-fill) / gauss (±1%) / impulse (5% glitches)
CALCE, seed 42, 100 epochs. Output is absolute normalized capacity.
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
MODES = ["clean", "drop30", "gauss", "impulse"]


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
        for i in range(1, n):
            if np.isnan(tc[i]):
                tc[i] = tc[i - 1]
        if np.isnan(tc[0]):
            first_ok = int(np.argmax(~np.isnan(tc)))
            tc[:first_ok] = tc[first_ok]
    elif mode == "gauss":
        tc = tc + rng.normal(0, 0.01, size=len(tc))
    elif mode == "impulse":
        n = len(tc)
        idx = rng.choice(n, int(n * 0.05), replace=False)
        tc[idx] += rng.normal(0, 0.05, size=len(idx))
    return tc


def run(model_kind):
    """model_kind: 'phys_ir' (absolute) or 'direct_z' (z-score control)."""
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

    use_ir = model_kind == "phys_ir"
    zscore = model_kind == "direct_z"
    model = build_gdn_model(
        multiscale=False, input_dim=2 if use_ir else 1, window_size=W,
        output_len=1, readout="phys_ir" if use_ir else "last",
    ).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    X, Y = [], []
    for c in train_cells:
        seq = (caps[c] - lo_c) / (hi_c - lo_c + EPS)
        ir = None
        if use_ir:
            ir = (feats[c][:, IR_IDX] - lo_i) / (hi_i - lo_i + EPS)
        for i in range(W, len(seq)):
            if use_ir:
                X.append(np.stack([seq[i - W:i], ir[i - W:i]], axis=1))
            else:
                X.append(seq[i - W:i, None])
            Y.append(seq[i])
    X = np.stack(X).astype(np.float32)
    Y = np.array(Y, dtype=np.float32)
    N = len(X)
    print(f"training {model_kind} clean (N={N}) ...", flush=True)
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
        if ep % 10 == 0:
            print(f"    ep{ep} loss={loss.item():.4f}", flush=True)

    tc_clean = (caps[test_cell] - lo_c) / (hi_c - lo_c + EPS)
    ti = None
    if use_ir:
        ti = (feats[test_cell][:, IR_IDX] - lo_i) / (hi_i - lo_i + EPS)
    th = (eol_ah - lo_c) / (hi_c - lo_c + EPS)

    model.eval()
    for mode in MODES:
        tc = corrupt(tc_clean, mode) if mode != "clean" else tc_clean
        seg_p = []
        with torch.no_grad():
            for i in range(W, len(tc)):
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
        seg_p = np.array(seg_p)
        tv = tc_clean[W:]
        mae = np.mean(np.abs(seg_p - tv))
        r2 = 1 - np.sum((tv - seg_p) ** 2) / (np.sum((tv - tv.mean()) ** 2) + EPS)
        ae = abs(true_rul(tv, th) - true_rul(seg_p, th))
        print(f"  [{mode}] {model_kind}: MAE={mae:.4f} R2={r2:.4f} AE={ae}",
              flush=True)


if __name__ == "__main__":
    for kind in ["phys_ir", "direct_z"]:
        run(kind)
