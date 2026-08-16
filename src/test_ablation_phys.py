"""Physics ablation: phys_ir vs direct, each in its best objective.

  phys_ir   — [C,IR] input, r = softplus(w·h) + softplus(gamma)·IR_last,
              Q_hat = Q_last - r (structural monotonicity), ABSOLUTE space
  direct_z  — [C] input, free head, z-score targets (PatchFormer-
              consistent protocol) — the fair no-physics baseline
  direct_abs— [C] input, free head, absolute space (attribution row:
              shows the win is NOT from the objective alone)

CALCE, single-branch, seed 42, 100 epochs. Standard protocol eval.
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


def build_windows(caps, feats, cells, lo_c, hi_c, lo_i, hi_i, use_ir):
    X, Y = [], []
    for c in cells:
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
    return np.stack(X).astype(np.float32), np.array(Y, dtype=np.float32)


def run(mode):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    caps, train_cells, test_cell, W, sps, eol_ah = load_series("calce")
    caps_all, feats_all, _ = load_calce_cells_multivar()
    caps = {c: caps_all[c].astype(np.float32) for c in caps_all}
    feats = {c: feats_all[c].astype(np.float32) for c in feats_all}

    all_c = np.concatenate([caps[c] for c in train_cells])
    lo_c, hi_c = float(all_c.min()), float(all_c.max())
    lo_i = hi_i = 0.0
    use_ir = mode == "phys_ir"
    if use_ir:
        all_i = np.concatenate([feats[c][:, IR_IDX] for c in train_cells])
        lo_i, hi_i = float(all_i.min()), float(all_i.max())

    zscore = (mode == "direct_z")  # PatchFormer-consistent target protocol
    model = build_gdn_model(
        multiscale=False, input_dim=2 if use_ir else 1, window_size=W,
        output_len=1, readout="phys_ir" if use_ir else "last",
    ).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    X, Y = build_windows(caps, feats, train_cells,
                         lo_c, hi_c, lo_i, hi_i, use_ir)
    N = len(X)
    print(f"training {mode} (N={N}) ...", flush=True)
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
                tgt = y  # absolute normalized capacity
            loss = masked_mae(pred, tgt, torch.ones_like(y))
            loss.backward()
            opt.step()
        if ep % 10 == 0:
            print(f"    ep{ep} loss={loss.item():.4f}", flush=True)

    model.eval()
    tc = (caps[test_cell] - lo_c) / (hi_c - lo_c + EPS)
    ti = None
    if use_ir:
        ti = (feats[test_cell][:, IR_IDX] - lo_i) / (hi_i - lo_i + EPS)
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
    tv = tc[W:]
    mae = np.mean(np.abs(seg_p - tv))
    r2 = 1 - np.sum((tv - seg_p) ** 2) / (np.sum((tv - tv.mean()) ** 2) + EPS)
    regen = np.mean(np.diff(seg_p) > 0.002)
    th = (eol_ah - lo_c) / (hi_c - lo_c + EPS)
    ae = abs(true_rul(tv, th) - true_rul(seg_p, th))
    print(f"  [{mode}] MAE={mae:.4f} R2={r2:.4f} regen={regen:.3f} AE={ae}",
          flush=True)
    return mae, r2, regen, ae


if __name__ == "__main__":
    for mode in ["phys_ir", "direct_z"]:
        run(mode)
    print("reference: direct_abs 0.0097 (attribution, already measured) | "
          "phys_ir extrap 0.7745 (vs n 0.4471)", flush=True)
