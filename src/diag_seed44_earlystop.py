"""Diagnose CALCE seed 44 AE=7: is it late-epoch overfitting?

Trains CALCE seed 44 with the EXACT main-model protocol of
test_full_train.py.train_eval (multiscale + stage-query, z-score
target, 100 epochs, seed 44) and, every 5 epochs:

  - logs epoch-MEAN train loss (the retrain log's ep%25==0 values are
    noisy last-batch losses);
  - saves a checkpoint to checkpoints/seed44_epochs/ep{N}.pt;
  - evaluates the test cell (full-sequence AE/MAE/R2, no per-SP).

Question for the user: if AE stays low mid-training and rises to 7 at
the end while train loss keeps falling, late-epoch overfitting on the
flat tail explains AE=7 and early stopping is the fix. If AE=7 appears
early and persists, it is a deterministic flat-tail crossing artifact,
not overfitting.

NOTE on model selection: picking the epoch by TEST-cell AE would be
cherry-picking; this script only gathers the curve so we can decide on
an honest selection rule (e.g. validation cell / train loss) afterwards.
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, '.')
from gdn_model import build_gdn_model, masked_mae
from make_figures import load_series
from eval_multiseed import true_rul

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DS, SEED = "calce", 44
BATCH, EPOCHS, CKPT_EVERY = 64, 100, 5
EPS = 1e-6


def eval_test(model, caps, test_cell, lo, hi, eol_ah, W):
    """Full-sequence test-cell metrics (matches test_full_train)."""
    model.eval()
    tc = (caps[test_cell] - lo) / (hi - lo + EPS)
    seg_p = []
    with torch.no_grad():
        for i in range(W, len(tc)):
            win = tc[i - W:i, None]
            cin = torch.tensor(win, dtype=torch.float32).unsqueeze(0).to(DEV)
            wmean = float(win[:, 0].mean())
            wstd = float(win[:, 0].std()) + EPS
            seg_p.append(model(cin).item() * wstd + wmean)
    seg_p = np.array(seg_p)
    tv = tc[W:]
    mae = np.mean(np.abs(seg_p - tv))
    r2 = 1 - np.sum((tv - seg_p) ** 2) / (np.sum((tv - tv.mean()) ** 2) + EPS)
    th = (eol_ah - lo) / (hi - lo + EPS)
    ae = abs(true_rul(tv, th) - true_rul(seg_p, th))
    return mae, r2, ae


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    caps, train_cells, test_cell, W, sps, eol_ah = load_series(DS)
    caps = {c: caps[c].astype(np.float32) for c in caps}
    all_tr = np.concatenate([caps[c] for c in train_cells])
    lo, hi = float(all_tr.min()), float(all_tr.max())

    model = build_gdn_model(
        multiscale=True, stage_query=True, input_dim=1, window_size=W,
        output_len=1, readout="last",
    ).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    X, Y = [], []
    for c in train_cells:
        seq = (caps[c] - lo) / (hi - lo + EPS)
        for i in range(W, len(seq)):
            X.append(seq[i - W:i, None])
            Y.append(seq[i])
    X = np.stack(X).astype(np.float32)
    Y = np.array(Y, dtype=np.float32)
    N = len(X)
    print(f"training {DS} seed{SEED} (N={N}, W={W}) ...", flush=True)

    os.makedirs("../checkpoints/seed44_epochs", exist_ok=True)
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(N)
        tot, cnt = 0.0, 0
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
            tot += loss.item() * len(idx)
            cnt += len(idx)
        mean_loss = tot / cnt
        if (ep + 1) % CKPT_EVERY == 0:
            torch.save({"state_dict": model.state_dict(), "epoch": ep + 1},
                       f"../checkpoints/seed44_epochs/ep{ep + 1:03d}.pt")
            mae, r2, ae = eval_test(model, caps, test_cell, lo, hi,
                                    eol_ah, W)
            print(f"  ep{ep + 1:3d} mean_train_loss={mean_loss:.4f} "
                  f"test MAE={mae:.4f} R2={r2:.4f} AE={ae}", flush=True)


if __name__ == "__main__":
    main()
