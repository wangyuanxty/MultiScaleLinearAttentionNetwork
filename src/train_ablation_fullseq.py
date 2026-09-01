"""Full-sequence ablation with multiple seeds.

Protocol = the paper's old ablation (test_ablation_multi.py), user
decision 2026-08-29: keep full-sequence single-value metrics, extend
single seed 42 to seeds 1..10 (first run 1..3 to sanity-check).

  - train: one model per (config, seed), 100 epochs, train = non-test
    cells' FULL sequences, min--max from train cells, per-window
    z-score targets, last-token readout
  - eval : FULL-SEQUENCE single-value MAE/R2/regen/AE on the test cell
    (whole trajectory sliding window, per-window de-normalization)
    NO per-SP slicing — identical to test_ablation_multi.py

Configs (same as Table 4 / run_ablation.py):
  single : multiscale=False                 (single branch)
  multi  : multiscale=True,  stage_query=False (3 branches, no xchg)
  xchg   : multiscale=True,  stage_query=True  (main model)

Outputs (never touch old ablation_*.json / abl_calce_*.pt):
  checkpoints/abl_seed/{config}/seed{seed}.pt
  results/ablation_fullseq_calce.json   {"config": {"seed": {...}}}
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
from train_per_sp import build_windows

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH, EPOCHS = 64, 100
EPS = 1e-6

CONFIGS = {
    "single": {"multiscale": False, "stage_query": False},
    "multi": {"multiscale": True, "stage_query": False},
    "xchg": {"multiscale": True, "stage_query": True},
}


def train_one(seed, X, Y, W, cfg, schedule="const", lr=1e-3, epochs=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_gdn_model(
        multiscale=cfg["multiscale"], stage_query=cfg["stage_query"],
        input_dim=1, window_size=W, output_len=1, readout="last",
    ).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_epochs = epochs or EPOCHS
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_epochs, eta_min=1e-6) if schedule == "cosine" else None)
    N = len(X)
    for ep in range(n_epochs):
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
            tgt = (y - wmean) / wstd
            loss = masked_mae(pred, tgt, torch.ones_like(y))
            loss.backward()
            opt.step()
        if sched is not None and ep % 10 == 0:
            print(f"  ep{ep} loss={loss.item():.4f} lr={sched.get_last_lr()[0]:.2e}",
                  flush=True)
        elif ep % 10 == 0:
            print(f"  ep{ep} loss={loss.item():.4f}", flush=True)
        if sched is not None:
            sched.step()
    return model


def eval_fullseq(model, caps, test_cell, W, lo, hi, eol_ah):
    """Full-sequence single-value MAE/R2/regen/AE (old ablation protocol)."""
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
    regen = np.mean(np.diff(seg_p) > 0.002)
    th = (eol_ah - lo) / (hi - lo + EPS)
    ae = abs(true_rul(tv, th) - true_rul(seg_p, th))
    return {"mae": float(mae), "r2": float(r2),
            "regen": float(regen), "ae": float(ae)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["calce", "nasa"], default="calce")
    ap.add_argument("--config", nargs="+", required=True, choices=list(CONFIGS))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--start-seed", type=int, default=1)
    ap.add_argument("--repro42", action="store_true",
                    help="Also run seed 42 (checks old table-4 numbers reproduce)")
    ap.add_argument("--lr-schedule", choices=["const", "cosine"], default="const")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--tag", default="",
                    help="Suffix for output JSON/ckpt paths (isolation)")
    args = ap.parse_args()

    caps, train_cells, test_cell, W, sps, eol_ah = load_series(args.dataset)
    caps = {c: caps[c].astype(np.float32) for c in caps}

    all_tr = np.concatenate([caps[c] for c in train_cells])
    lo, hi = float(all_tr.min()), float(all_tr.max())
    X, Y = build_windows(caps, train_cells, lo, hi, W)

    tag = f"_{args.tag}" if args.tag else ""
    out_path = f"results/ablation_fullseq_{args.dataset}{tag}.json"
    out = {}
    if os.path.exists(out_path):
        try:
            out = json.load(open(out_path)) or {}
        except json.JSONDecodeError:
            print("WARN: results JSON 为空/损坏,按空结果处理", flush=True)
    os.makedirs(f"../checkpoints/abl_seed/{args.dataset}{tag}", exist_ok=True)

    seed_list = list(range(args.start_seed, args.start_seed + args.seeds))
    if args.repro42:
        seed_list.append(42)

    for cfg_name in args.config:
        cfg = CONFIGS[cfg_name]
        ckpt_dir = f"../checkpoints/abl_seed/{args.dataset}{tag}/{cfg_name}"
        os.makedirs(ckpt_dir, exist_ok=True)
        for seed in seed_list:
            skey = str(seed)
            if skey in out.setdefault(cfg_name, {}):
                print(f"[{cfg_name}] seed{seed}: SKIP (saved)", flush=True)
                continue
            ckpt_path = f"{ckpt_dir}/seed{seed}.pt"
            if os.path.exists(ckpt_path):
                # 中断恢复:复用已存 ckpt,跳过训练,重算指标
                ckpt = torch.load(ckpt_path, map_location=DEV)
                model = build_gdn_model(
                    multiscale=cfg["multiscale"], stage_query=cfg["stage_query"],
                    input_dim=1, window_size=W, output_len=1, readout="last",
                ).to(DEV)
                model.load_state_dict(ckpt["state_dict"])
                print(f"[{cfg_name}] seed{seed}: 复用 ckpt(跳过训练)", flush=True)
            else:
                model = train_one(seed, X, Y, W, cfg, schedule=args.lr_schedule,
                                  lr=args.lr, epochs=args.epochs)
                torch.save({"state_dict": model.state_dict(), "seed": seed,
                            "lo": lo, "hi": hi, "W": W, "eol_ah": eol_ah,
                            "config": cfg_name}, ckpt_path)
            r = eval_fullseq(model, caps, test_cell, W, lo, hi, eol_ah)
            out[cfg_name][skey] = r
            print(f"[{cfg_name}] seed{seed}: MAE={r['mae']:.4f} "
                  f"R2={r['r2']:.4f} regen={r['regen']:.3f} AE={r['ae']}",
                  flush=True)
            with open(out_path, "w") as fp:
                json.dump(out, fp, indent=2)
                fp.flush()
                os.fsync(fp.fileno())

    # summary mean±std over the requested seeds (1..N)
    print("=== full-seq mean+/-std ===", flush=True)
    for cfg_name in args.config:
        rows = [out[cfg_name][str(s)] for s in range(
            args.start_seed, args.start_seed + args.seeds)]
        print(f"{cfg_name}: MAE={np.mean([r['mae'] for r in rows]):.4f}"
              f"±{np.std([r['mae'] for r in rows]):.4f} "
              f"R2={np.mean([r['r2'] for r in rows]):.4f}"
              f"±{np.std([r['r2'] for r in rows]):.4f} "
              f"regen={np.mean([r['regen'] for r in rows]):.3f} "
              f"AE={np.mean([r['ae'] for r in rows]):.1f}", flush=True)


if __name__ == "__main__":
    main()
