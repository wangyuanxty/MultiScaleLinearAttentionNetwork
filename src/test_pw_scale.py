"""Per-window target scaling experiment (PatchFormer's EncoderNormalizer
semantics) on our single-branch model, CALCE.

Training: input x globally normalized (lo/hi from train cells); target
per-window scaled: t_norm = (t - mean_win)/std_win with mean/std from
the 64-step input window. Inference: predict t_norm, de-normalize with
the SAME window stats, then compare against absolute capacity.

Goal: does per-window scaling change per-SP AE? Compare with the
global-normalization single-branch baseline.
"""
import sys
import numpy as np
import torch

sys.path.insert(0, '.')
from gdn_model import build_gdn_model
from make_figures import load_series
from eval_multiseed import true_rul

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
W, BATCH, EPOCHS, SEEDS = 64, 64, 100, [42, 43, 44]
EPS = 1e-6


def pw_stats(wins):
    """(N, W) -> (N,) mean, (N,) std per window."""
    mean = wins.mean(axis=1)
    std = wins.std(axis=1) + EPS
    return mean, std


def train_eval_pw(seed, ds="calce"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    caps, train_cells, test_cell, W, sps, eol_ah = load_series(ds)
    all_tr = np.concatenate([caps[c] for c in train_cells])
    lo, hi = all_tr.min(), all_tr.max()

    model = build_gdn_model(
        multiscale=False, input_dim=1, window_size=W, output_len=1,
        readout="last",
    ).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # build training windows: (N, W) inputs, (N,) targets
    tr_seqs = [(caps[c] - lo) / (hi - lo + EPS) for c in train_cells]
    X_tr, Y_tr = [], []
    for seq in tr_seqs:
        for i in range(W, len(seq)):
            X_tr.append(seq[i - W:i])
            Y_tr.append(seq[i])
    X_tr = np.stack(X_tr).astype(np.float32)
    Y_tr = np.stack(Y_tr).astype(np.float32)
    N = len(X_tr)

    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(N)
        for s in range(0, N, BATCH):
            idx = perm[s:s + BATCH]
            x = torch.tensor(X_tr[idx]).unsqueeze(-1).to(DEV)
            y = torch.tensor(Y_tr[idx]).to(DEV)
            # per-window target scaling
            mean = x.squeeze(-1).mean(dim=1)
            std = x.squeeze(-1).std(dim=1) + EPS
            y_norm = (y - mean) / std
            pred = model(x).squeeze(-1)
            loss = torch.abs(pred - y_norm).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    tc = (caps[test_cell] - lo) / (hi - lo + EPS)
    wins = np.stack([tc[i - W:i] for i in range(W, len(tc))]).astype(np.float32)
    mean_w, std_w = pw_stats(wins)
    cin = torch.tensor(wins).unsqueeze(-1).to(DEV)
    with torch.no_grad():
        pred_norm = model(cin).cpu().numpy()[:, 0]
    # de-normalize with the same window stats (compare in normalized
    # capacity space against the normalized threshold)
    pred_abs_norm = pred_norm * std_w + mean_w
    th = (eol_ah - lo) / (hi - lo + EPS)
    tc_true = tc[W:]
    aes = []
    for sp in sps:
        seg_p = pred_abs_norm[sp - W:]
        seg_t = tc[sp:]
        n = min(len(seg_p), len(seg_t))
        aes.append(abs(true_rul(seg_t[:n], th) - true_rul(seg_p[:n], th)))
    r2 = 1 - np.sum((tc_true - pred_abs_norm) ** 2) / \
        np.sum((tc_true - tc_true.mean()) ** 2)
    mae_global = np.mean(np.abs(tc_true - pred_abs_norm))
    rmse_global = np.sqrt(np.mean((tc_true - pred_abs_norm) ** 2))
    # per-SP MAE/RMSE/R2
    sp_metrics = []
    for sp in sps:
        seg_p = pred_abs_norm[sp - W:]
        seg_t = tc[sp:]
        n = min(len(seg_p), len(seg_t))
        seg_p, seg_t = seg_p[:n], seg_t[:n]
        mae = np.mean(np.abs(seg_p - seg_t))
        rmse = np.sqrt(np.mean((seg_p - seg_t) ** 2))
        r2sp = 1 - np.sum((seg_t - seg_p) ** 2) / \
            (np.sum((seg_t - seg_t.mean()) ** 2) + EPS)
        sp_metrics.append((mae, rmse, r2sp))
    print(f"  seed{seed}: R2={r2:.4f} MAE={mae_global:.4f} "
          f"RMSE={rmse_global:.4f} AE={aes}", flush=True)
    print(f"          per-SP: " + " | ".join(
        f"SP{sp}: MAE={m:.4f} RMSE={r:.4f} R2={s:.4f}"
        for sp, (m, r, s) in zip(sps, sp_metrics)), flush=True)
    return r2, aes


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "calce"
    print(f"per-window scale single-branch, {ds} (100 epochs):", flush=True)
    for s in SEEDS:
        train_eval_pw(s, ds)
