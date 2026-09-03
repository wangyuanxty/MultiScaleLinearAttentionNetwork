"""Physics rate-head line, multi-seed runner (CALCE, 10 seeds).

Replicates the paper's physics protocol exactly:

  rate head : single-branch, input [C, IR] (IR normalized from train
              cells), absolute-space MAE objective, 100 epochs,
              readout='phys_ir' (r = softplus(w.h) + softplus(gamma).IR)
  std eval  : full-sequence MAE/R2/regen/AE (absolute normalized space)
  extrap    : same model trained on frac=0.9 -> tail (last 10%) R2/MAE
  control   : same [C,IR] single-branch architecture, readout='last'
              (free head) in ABSOLUTE space -> isolates mechanism
  robustness: capacity channel corrupted (drop30 hold-last-value /
              gauss +/-1% / impulse 5%), IR clean; applied to BOTH
              (a) the z-score free head (reuses existing 10-seed ckpts
                  checkpoints/abl_seed/calce/single/seed{n}.pt)
              (b) the absolute rate head

Outputs:
  checkpoints/phys_ir_seeds/seed{n}_{std,extrap,free}.pt
  results/physics_ir_seeds.json   {seed: {std, extrap, control,
                                         robust_free, robust_rate}}
"""
import argparse
import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, '.')
from gdn_model import build_gdn_model, masked_mae
from make_figures import load_series
from eval_multiseed import true_rul
from load_datasets import load_calce_cells_multivar

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
W, BATCH, EPOCHS, EPS = 64, 64, 100, 1e-6
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


def load_data():
    caps, train_cells, test_cell, W_, sps, eol_ah = load_series("calce")
    caps_all, feats_all, _ = load_calce_cells_multivar()
    caps = {c: caps_all[c].astype(np.float32) for c in caps_all}
    feats = {c: feats_all[c].astype(np.float32) for c in feats_all}
    tr_c = np.concatenate([caps[c] for c in train_cells])
    tr_i = np.concatenate([feats[c][:, IR_IDX] for c in train_cells])
    return caps, feats, train_cells, test_cell, W_, sps, eol_ah, \
        float(tr_c.min()), float(tr_c.max()), \
        float(tr_i.min()), float(tr_i.max())


def train_ir(seed, X, Y, W_, readout):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_gdn_model(
        multiscale=False, input_dim=2, window_size=W_, output_len=1,
        readout=readout).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    N = len(X)
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
    return model


def eval_ir(model, caps, feats, test_cell, lo_c, hi_c, lo_i, hi_i,
            eol_ah, frac=None):
    model.eval()
    tc = (caps[test_cell] - lo_c) / (hi_c - lo_c + EPS)
    ti = (feats[test_cell][:, IR_IDX] - lo_i) / (hi_i - lo_i + EPS)
    start = W
    if frac is not None:  # extrapolation: evaluate final (1-frac)
        cut = int(len(tc) * frac)
        start = max(start, cut)
        tv = tc[cut:]
    else:
        tv = tc[W:]
    seg_p = []
    with torch.no_grad():
        for i in range(start, len(tc)):
            win = np.stack([tc[i - W:i], ti[i - W:i]], axis=1)
            cin = torch.tensor(win, dtype=torch.float32).unsqueeze(0).to(DEV)
            seg_p.append(model(cin).item())
    seg_p = np.array(seg_p)
    n = min(len(tv), len(seg_p))
    tv, seg_p = tv[:n], seg_p[:n]
    mae = float(np.mean(np.abs(seg_p - tv)))
    r2 = float(1 - np.sum((tv - seg_p) ** 2) / (np.sum((tv - tv.mean()) ** 2) + EPS))
    regen = float(np.mean(np.diff(seg_p) > 0.002))
    th = (eol_ah - lo_c) / (hi_c - lo_c + EPS)
    ae = abs(true_rul(tv, th) - true_rul(seg_p, th))
    return {"mae": mae, "r2": r2, "regen": regen, "ae": float(ae)}


def corrupt_capacity(tc_clean, mode, seed):
    tc = tc_clean.copy()
    rng = np.random.RandomState(seed)
    if mode == "drop30":
        n = len(tc)
        seg = 8
        n_drop = int(n * 0.30 / seg)
        starts = rng.choice(np.arange(0, n - seg, seg), n_drop, replace=False)
        tmp = tc.copy()
        for st in starts:
            tmp[st:st + seg] = np.nan
        out = tmp.copy()
        for i in range(1, n):
            if np.isnan(out[i]):
                out[i] = out[i - 1]
        out = np.where(np.isnan(out), 0.0, out)
        return out
    elif mode == "gauss":
        return tc_clean + rng.normal(0, 0.01, size=len(tc_clean))
    elif mode == "impulse":
        n = len(tc_clean)
        idx = rng.choice(n, int(n * 0.05), replace=False)
        tc[idx] += rng.normal(0, 0.05, size=len(idx))
        return tc
    return tc  # clean


def eval_ir_corrupted(model, caps, feats, test_cell, lo_c, hi_c, lo_i, hi_i,
                      eol_ah, mode, seed):
    model.eval()
    tc_clean = (caps[test_cell] - lo_c) / (hi_c - lo_c + EPS)
    ti = (feats[test_cell][:, IR_IDX] - lo_i) / (hi_i - lo_i + EPS)
    tc = corrupt_capacity(tc_clean, mode, seed)
    tv = tc_clean[W:]
    seg_p = []
    with torch.no_grad():
        for i in range(W, len(tc)):
            win = np.stack([tc[i - W:i], ti[i - W:i]], axis=1)
            cin = torch.tensor(win, dtype=torch.float32).unsqueeze(0).to(DEV)
            seg_p.append(model(cin).item())
    seg_p = np.array(seg_p)
    n = min(len(tv), len(seg_p))
    tv, seg_p = tv[:n], seg_p[:n]
    mae = float(np.mean(np.abs(seg_p - tv)))
    r2 = float(1 - np.sum((tv - seg_p) ** 2) / (np.sum((tv - tv.mean()) ** 2) + EPS))
    th = (eol_ah - lo_c) / (hi_c - lo_c + EPS)
    ae = float(abs(true_rul(tv, th) - true_rul(seg_p, th)))
    return {"mae": mae, "r2": r2, "ae": ae}


def eval_z_corrupted(zck, caps, test_cell, eol_ah, mode, seed, W_):
    torch.manual_seed(0)
    model = build_gdn_model(
        multiscale=False, input_dim=1, window_size=W_, output_len=1,
        readout='last').to(DEV)
    model.load_state_dict(zck["state_dict"])
    model.eval()
    lo, hi = zck["lo"], zck["hi"]
    tc_clean = (caps[test_cell] - lo) / (hi - lo + EPS)
    tc = corrupt_capacity(tc_clean, mode, seed)
    tv = tc_clean[W_:]
    seg_p = []
    with torch.no_grad():
        for i in range(W_, len(tc)):
            win = tc[i - W_:i]
            wmean = float(win.mean())
            wstd = float(win.std()) + EPS
            cin = torch.tensor(win, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEV)
            seg_p.append(model(cin).item() * wstd + wmean)
    seg_p = np.array(seg_p)
    n = min(len(tv), len(seg_p))
    tv, seg_p = tv[:n], seg_p[:n]
    mae = float(np.mean(np.abs(seg_p - tv)))
    th = (eol_ah - lo) / (hi - lo + EPS)
    ae = float(abs(true_rul(tv, th) - true_rul(seg_p, th)))
    return {"mae": mae, "ae": ae}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--start-seed", type=int, default=1)
    args = ap.parse_args()

    caps, feats, train_cells, test_cell, W_, sps, eol_ah, \
        lo_c, hi_c, lo_i, hi_i = load_data()
    X1, Y1 = build_windows(caps, feats, train_cells, lo_c, hi_c, lo_i, hi_i, 1.0)
    X9, Y9 = build_windows(caps, feats, train_cells, lo_c, hi_c, lo_i, hi_i, 0.9)
    os.makedirs("../checkpoints/phys_ir_seeds", exist_ok=True)

    out_path = "results/physics_ir_seeds.json"
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}
    modes = ["clean", "drop30", "gauss", "impulse"]

    for seed in range(args.start_seed, args.start_seed + args.seeds):
        skey = str(seed)
        if skey in out:
            print(f"seed{seed}: SKIP", flush=True)
            continue
        rec = {}
        ck = f"../checkpoints/phys_ir_seeds/seed{seed}_std.pt"
        if os.path.exists(ck):
            m = train_ir(seed, X1, Y1, W_, "phys_ir")
            m.load_state_dict(torch.load(ck, map_location=DEV)["state_dict"])
        else:
            m = train_ir(seed, X1, Y1, W_, "phys_ir")
            torch.save({"state_dict": m.state_dict(), "seed": seed}, ck)
        rec["std"] = eval_ir(m, caps, feats, test_cell, lo_c, hi_c, lo_i,
                             hi_i, eol_ah)
        print(f"seed{seed} std: {rec['std']}", flush=True)
        ck9 = f"../checkpoints/phys_ir_seeds/seed{seed}_extrap.pt"
        if os.path.exists(ck9):
            m9 = train_ir(seed, X9, Y9, W_, "phys_ir")
            m9.load_state_dict(torch.load(ck9, map_location=DEV)["state_dict"])
        else:
            m9 = train_ir(seed, X9, Y9, W_, "phys_ir")
            torch.save({"state_dict": m9.state_dict(), "seed": seed}, ck9)
        rec["extrap"] = eval_ir(m9, caps, feats, test_cell, lo_c, hi_c,
                                lo_i, hi_i, eol_ah, frac=0.9)
        print(f"seed{seed} extrap: {rec['extrap']}", flush=True)
        ckf = f"../checkpoints/phys_ir_seeds/seed{seed}_free.pt"
        if os.path.exists(ckf):
            mf = train_ir(seed, X1, Y1, W_, "last")
            mf.load_state_dict(torch.load(ckf, map_location=DEV)["state_dict"])
        else:
            mf = train_ir(seed, X1, Y1, W_, "last")
            torch.save({"state_dict": mf.state_dict(), "seed": seed}, ckf)
        rec["control"] = eval_ir(mf, caps, feats, test_cell, lo_c, hi_c,
                                 lo_i, hi_i, eol_ah)
        print(f"seed{seed} control: {rec['control']}", flush=True)
        rec["robust_rate"] = {mode: eval_ir_corrupted(
            m, caps, feats, test_cell, lo_c, hi_c, lo_i, hi_i, eol_ah,
            mode, seed) for mode in modes}
        print(f"seed{seed} robust_rate: {rec['robust_rate']}", flush=True)
        zck = None
        for cand in (f"../checkpoints/abl_seed/calce/single/seed{seed}.pt",
                     f"../checkpoints/abl_seed/single/seed{seed}.pt"):
            if os.path.exists(cand):
                zck = torch.load(cand, map_location=DEV)
                break
        if zck is None:
            print(f"WARN: z-score single ckpt missing for seed{seed} "
                  "(skipping robust_free)", flush=True)
            rec["robust_free"] = {}
        else:
            rec["robust_free"] = {mode: eval_z_corrupted(
                zck, caps, test_cell, eol_ah, mode, seed, W_) for mode in modes}
            print(f"seed{seed} robust_free: {rec['robust_free']}", flush=True)
        out[skey] = rec
        with open(out_path, "w") as fp:
            json.dump(out, fp, indent=2)
            fp.flush()
            os.fsync(fp.fileno())


if __name__ == "__main__":
    main()
