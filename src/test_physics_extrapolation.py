"""Extrapolation ablation for physics-as-input: [C] vs [C, IR].

Train on the first 90% of each cell's life, evaluate on the unseen
final 10% (EOL acceleration). Does the IR input channel help tail
prediction? (Physics-in-objective was already rejected; this run is
pure data vs data+IR.)

CALCE, 1 seed, 100 epochs. Inputs globally min-maxed per channel
(train lo/hi); target per-window z-score (PatchFormer-consistent).
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
W, BATCH, EPOCHS, SEEDS = 64, 64, 100, [42]
EPS = 1e-6
IR_IDX = 3  # feat_cols: V_mean,V_min,V_max,IR,dvdt,E_d,E_c,AC,Phase,delta_t


def build_windows(caps, feats, train_cells, lo, hi, use_ir, frac=0.9):
    X, Y = [], []
    for c in train_cells:
        seq = (caps[c] - lo[0]) / (hi[0] - lo[0] + EPS)
        ir = (feats[c][:, IR_IDX] - lo[1]) / (hi[1] - lo[1] + EPS) if use_ir else None
        cut = int(len(seq) * frac)
        for i in range(W, cut):
            win = seq[i - W:i]
            if use_ir:
                win = np.stack([win, ir[i - W:i]], axis=1)
            else:
                win = win[:, None]  # (W, 1)
            X.append(win)
            Y.append(seq[i])
    return np.stack(X).astype(np.float32), np.array(Y, dtype=np.float32)


def train_eval(seed, use_ir, ds="calce"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    caps, train_cells, test_cell, W, sps, eol_ah = load_series(ds)
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

    X_tr, Y_tr = build_windows(caps, feats, train_cells, lo, hi, use_ir)
    N = len(X_tr)
    print(f"training use_ir={use_ir} (N={N}, in_dim={n_in}) ...", flush=True)
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(N)
        for s in range(0, N, BATCH):
            idx = perm[s:s + BATCH]
            x = torch.tensor(X_tr[idx]).to(DEV)
            y = torch.tensor(Y_tr[idx]).to(DEV)
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

    # ---- extrapolation evaluation: last 10% (unseen) ----
    model.eval()
    tc = (caps[test_cell] - lo[0]) / (hi[0] - lo[0] + EPS)
    tc_ir = (feats[test_cell][:, IR_IDX] - lo[1]) / (hi[1] - lo[1] + EPS) if use_ir else None
    cut = int(len(tc) * 0.9)
    seg_t = tc[cut:]
    seg_p = []
    with torch.no_grad():
        for i in range(cut, len(tc)):
            win = tc[i - W:i]
            if use_ir:
                win = np.stack([win, tc_ir[i - W:i]], axis=1)
            else:
                win = win[:, None]
            cin = torch.tensor(win, dtype=torch.float32).unsqueeze(0).to(DEV)
            wmean = float(win[:, 0].mean())
            wstd = float(win[:, 0].std()) + EPS
            seg_p.append(model(cin).item() * wstd + wmean)
    seg_p = np.array(seg_p)
    mae = np.mean(np.abs(seg_p - seg_t))
    r2 = 1 - np.sum((seg_t - seg_p) ** 2) / (np.sum((seg_t - seg_t.mean()) ** 2) + EPS)
    regen = np.mean(np.diff(seg_p) > 0.002)
    th = (eol_ah - lo[0]) / (hi[0] - lo[0] + EPS)
    ae = abs(true_rul(seg_t, th) - true_rul(seg_p, th))
    print(f"  seed{seed} use_ir={use_ir}: extR2={r2:.4f} extMAE={mae:.4f} "
          f"regen_frac={regen:.3f} AE={ae}", flush=True)
    return r2, mae, regen, ae


if __name__ == "__main__":
    print("CALCE extrapolation ablation, [C] vs [C,IR]:", flush=True)
    for use_ir in [False, True]:
        r2s, maes, regens, aes = [], [], [], []
        for s in SEEDS:
            r2, m, rg, a = train_eval(s, use_ir)
            r2s.append(r2)
            maes.append(m)
            regens.append(rg)
            aes.append(a)
        print(f"=== use_ir={use_ir}: R2={np.mean(r2s):.4f}±{np.std(r2s):.4f} "
              f"MAE={np.mean(maes):.4f}±{np.std(maes):.4f} "
              f"regen={np.mean(regens):.4f} AE={np.mean(aes):.1f}", flush=True)
