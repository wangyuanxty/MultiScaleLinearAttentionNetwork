"""Unified ablation runner — one config table, one eval protocol, results to JSON.

Protocol:
  single-step: rolling prediction, EOL crossing AE@300/400/500, R2
  multi-step : one-shot K=32 trajectory MAE/RMSE per SP
  fixed seed, results appended to results/ablation.json

Usage: python src/run_ablation.py [--only NAME] [--epochs 100]
"""
import sys, json, time, argparse, numpy as np, torch, torch.nn as nn
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from gdn_v2 import GDN2Block
from gdn_model import build_gdn_model, masked_mae, PhysicsRegularizer
from load_datasets import load_calce_cells_multivar, load_nasa_multivar, load_nasa_capacity
from data_pipeline import Seq2VecDataset, collate_seq2vec
from torch.utils.data import DataLoader
import logging; logging.basicConfig(level=logging.WARNING)

DEV = torch.device("cuda")
BATCH, SEED = 64, 42

# ─── ablation config table ────────────────────────────────────────
CONFIGS = [
    # scale x features x physics x K  (all last-token readout)
    # pure-capacity input for ALL configs (physics enters loss only)
    {"name": "single-cap",        "multiscale": False, "physics": False, "K": 1},
    {"name": "multi-cap",         "multiscale": True,  "physics": False, "K": 1},
    {"name": "multi-cap-xchg",    "multiscale": True,  "physics": False, "K": 1, "stage_query": True},
    {"name": "multi-cap-physics", "multiscale": True,  "physics": True,  "K": 1},
    {"name": "multi-cap-traj32",  "multiscale": True,  "physics": False, "K": 32},
]


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def make_model(cfg, window_size):
    # Pure-capacity input; physics via PhysicsRegularizer in loss only.
    # xchg config -> stage-query attention exchange (V3)
    return build_gdn_model(
        multiscale=cfg["multiscale"],
        cross_exchange=cfg.get("cross", False),
        stage_query=cfg.get("stage_query", False),
        input_dim=1, window_size=window_size,
        output_len=cfg["K"], use_physics=False, ir_ch=None, t_ch=None,
        readout="last",
    )


def eval_table_a(model, tc, W, sps, eol_ah=None, lo=None, hi=None):
    """Table A: non-recursive trajectory — per-SP R2/MAE/RMSE (literature protocol).
    If eol_ah given, also compute AE per SP (PatchFormer/RUL-Mamba rul_value_error):
      y_true = tc[sp:], y_pred = pv[sp:], find first threshold crossing in each."""
    preds = []
    with torch.no_grad():
        for i in range(W, len(tc)):
            win = tc[i - W:i]
            wmean = float(win.mean()); wstd = float(win.std()) + 1e-6
            cin = torch.tensor(win, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEV)
            p_norm = model(cin).item()
            preds.append(p_norm * wstd + wmean)  # per-window de-normalize
    pv = np.array(preds)[:len(tc) - W]; tv = tc[W:]
    r2_global = 1 - np.sum((tv - pv) ** 2) / np.sum((tv - tv.mean()) ** 2)
    out = {"R2": round(float(r2_global), 4)}

    if eol_ah is not None:
        threshold_n = (eol_ah - lo) / (hi - lo + 1e-8)

    for sp in sps:
        seg = pv[sp - W:]; seg_t = tc[sp:]
        n = min(len(seg), len(seg_t))
        seg_p, seg_t = seg[:n], seg_t[:n]
        out[f"traj_MAE_sp{sp}"] = round(float(np.mean(np.abs(seg_p - seg_t))), 4)
        out[f"traj_RMSE_sp{sp}"] = round(float(np.sqrt(np.mean((seg_p - seg_t) ** 2))), 4)
        r2_sp = 1 - np.sum((seg_t - seg_p)**2) / (np.sum((seg_t - seg_t.mean())**2) + 1e-8)
        out[f"traj_R2_sp{sp}"] = round(float(r2_sp), 4)
        # AE: unified crossing definition (seg[i] >= th > seg[i+1])
        if eol_ah is not None:
            true_re, pred_re = len(seg_t), 0
            for i in range(len(seg_t) - 1):
                if seg_t[i] >= threshold_n > seg_t[i + 1]:
                    true_re = i
                    break
            for i in range(len(seg_p) - 1):
                if seg_p[i] >= threshold_n > seg_p[i + 1]:
                    pred_re = i
                    break
            out[f"AE_sp{sp}"] = round(float(abs(true_re - pred_re)), 1)
    return out


def eval_traj(model, tc, W, K, sps):
    out = {}
    with torch.no_grad():
        for sp in sps:
            tgt = tc[sp:sp + K]
            cin = torch.tensor(tc[sp - W:sp], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEV)
            pred = model(cin).squeeze(0).cpu().numpy()
            mae = np.mean(np.abs(pred - tgt))
            rmse = np.sqrt(np.mean((pred - tgt) ** 2))
            r2 = 1 - np.sum((tgt - pred) ** 2) / (np.sum((tgt - tgt.mean()) ** 2) + 1e-8)
            out[f"sp{sp}"] = {"mae": round(float(mae), 4), "rmse": round(float(rmse), 4), "r2": round(float(r2), 3)}
    return out


def eval_rul(model, tc, W, K, sps, eol_ah, raw_cap, lo, hi):
    """Deployment-consistent RUL: seq2vec chunked rollout to EOL crossing.
    eol_ah: EOL threshold in original capacity units (e.g. 1.40 Ah for NASA).
    raw_cap: original (unnormalized) test capacity series — true EOL from it.
    RUL_MAE = |true_EOL - predicted_EOL| in cycles."""
    eol_n = (eol_ah - lo) / (hi - lo + 1e-8)  # normalized EOL threshold
    te = int(np.argmax(raw_cap < eol_ah)) if (raw_cap < eol_ah).any() else len(raw_cap)
    out = {}
    with torch.no_grad():
        for sp in sps:
            window = tc[sp - W:sp].copy()
            traj = []
            while len(traj) < 800:
                cin = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEV)
                pred = model(cin).squeeze(0).cpu().numpy()
                traj.extend(pred.tolist())
                window = np.concatenate([window[K:], pred])
                if pred[-1] < eol_n:
                    break
            traj = np.array(traj)
            pe = -1
            for j in range(len(traj) - 1):
                if traj[j] >= eol_n > traj[j + 1]:
                    pe = sp + j + (eol_n - traj[j]) / (traj[j + 1] - traj[j] + 1e-8)
                    break
                elif traj[j] < eol_n:
                    pe = sp + j
                    break
            out[f"RUL_sp{sp}"] = {"rul_mae": round(float(abs(te - pe)), 1) if pe >= 0 else None,
                                   "true_eol": te, "pred_eol": round(float(pe), 1) if pe >= 0 else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--dataset", default="calce", choices=["calce", "nasa", "mit", "mit", "panasonic", "tju"])
    args = ap.parse_args()

    if args.dataset == "calce":
        DS = {
            "W": 64, "train": ["CS2_36","CS2_37","CS2_38"], "test": "CS2_35",
            "sps": [300, 400, 500], "ir_ch": 0, "eol_ah": 0.77,  # rated 1.1 × 0.70
        }
        caps_all, feats_all, _ = load_calce_cells_multivar()
        caps = {c: caps_all[c].copy() for c in caps_all}
        # physics loss only: keep IR channel
        feats = {c: feats_all[c][:, 3:4].copy() for c in feats_all}  # col 3 = IR
    elif args.dataset == "nasa":
        DS = {
            "W": 30, "train": ["B0006","B0007","B0018"], "test": "B0005",
            "sps": [50, 70, 90], "ir_ch": 0, "eol_ah": 1.40,
        }
        batteries = DS["train"] + [DS["test"]]
        caps, feats = {}, {}
        for bat in batteries:
            data = load_nasa_multivar(bat)
            caps[bat] = data['capacity'].astype(np.float32)
            # physics loss only: [IR(Re), T(T_mean)]
            feats[bat] = np.stack([data['Re'], data['T_mean']], axis=-1).astype(np.float32)
    elif args.dataset == "mit":
        DS = {"W": 64, "sps": [200, 300, 400], "ir_ch": 0, "eol_ah": 0.86}
        from load_datasets import load_mit_stanford_multivar, MIT_TRAIN_CELLS, MIT_TEST_CELLS
        caps_all, feats_all, fd = load_mit_stanford_multivar()
        # 80/20 full-dataset split: 35 train / 8 test cells
        train_cells = [c for c in MIT_TRAIN_CELLS if c in caps_all]
        test_cells = [c for c in MIT_TEST_CELLS if c in caps_all]
        caps = {c: caps_all[c].copy().astype(np.float32)
                for c in train_cells + test_cells}
        # Physics loss only: [IR, Tavg] — feature cols 0 (IR) and 2 (Tavg)
        feats = {c: feats_all[c][:, [0, 2]].copy().astype(np.float32)
                 for c in train_cells + test_cells}
        DS["train"], DS["test"] = train_cells, test_cells[0]
    elif args.dataset == "panasonic":
        DS = {"W": 30, "sps": [300, 500, 700], "ir_ch": None, "eol_ah": 2.12}  # rated 3.03 × 0.70
        from load_datasets import load_panasonic_cells
        caps_all = load_panasonic_cells()
        caps = {c: caps_all[c].copy().astype(np.float32) for c in caps_all}
        cells = sorted(caps.keys())
        DS["train"], DS["test"] = cells[:-1], cells[-1]
        feats = None
    elif args.dataset == "tju":
        DS = {"W": 64, "sps": [200, 300, 400], "ir_ch": None, "eol_ah": 1.75}  # rated 2.5 × 0.70
        from load_datasets import load_tju_cells
        caps_all = load_tju_cells()
        caps = {c: caps_all[c].copy().astype(np.float32) for c in caps_all}
        DS["train"], DS["test"] = ['CY25_2','CY25_3'], 'CY25_1'
        feats = None
    else:  # gotion
        DS = {"W": 30, "sps": [500, 800, 1100], "ir_ch": None, "eol_ah": 21.60}  # rated 27 × 0.80
        from load_datasets import load_gotion_cells
        caps_all = load_gotion_cells()
        caps = {c: caps_all[c].copy().astype(np.float32) for c in caps_all}
        cells = sorted(caps.keys())
        DS["train"], DS["test"] = cells[:-1], cells[-1]
        feats = None

    W = DS["W"]
    train, test = DS["train"], DS["test"]

    all_tr = np.concatenate([caps[c] for c in train])
    lo, hi = all_tr.min(), all_tr.max()

    def scale(seqs):
        return [(s - lo) / (hi - lo) for s in seqs]

    results_file = Path(__file__).parent.parent / "results" / f"ablation_{args.dataset}.json"
    results_file.parent.mkdir(exist_ok=True)
    results = json.loads(results_file.read_text()) if results_file.exists() else {}

    IR_CH = DS["ir_ch"]

    for cfg in CONFIGS:
        if args.only and args.only != cfg["name"]:
            continue
        # physics configs need IR/T features — skip on datasets without them
        if cfg["physics"] and feats is None:
            print(f"  skip {cfg['name']} (no physics features on {args.dataset})", flush=True)
            continue
        set_seed(SEED)
        print(f"\n=== [{args.dataset}] {cfg['name']} (K={cfg['K']}) ===", flush=True)
        feat_list = [feats[c] for c in train] if cfg["physics"] else None
        tr = Seq2VecDataset(scale([caps[c] for c in train]), W, cfg["K"], 1, feat_list)
        ld = DataLoader(tr, BATCH, shuffle=True, collate_fn=collate_seq2vec)
        model = make_model(cfg, W).to(DEV)
        phys_reg = PhysicsRegularizer(lambda_=0.1).to(DEV) if cfg["physics"] else None
        params = list(model.parameters())
        if phys_reg: params += list(phys_reg.parameters())
        opt = torch.optim.Adam(params, lr=1e-3)
        n_params = sum(p.numel() for p in model.parameters())
        for ep in range(args.epochs):
            model.train()
            for cap, feat, tgt, msk in ld:
                cap, tgt, msk = cap.to(DEV), tgt.to(DEV), msk.to(DEV)
                opt.zero_grad()
                pred = model(cap)  # pure-capacity input
                # per-window target scaling (PatchFormer-style)
                wmean = cap.squeeze(-1).mean(dim=1, keepdim=True)
                wstd = cap.squeeze(-1).std(dim=1, keepdim=True) + 1e-6
                tgt_norm = (tgt - wmean) / wstd
                loss = masked_mae(pred, tgt_norm, msk)
                if phys_reg and feat is not None:
                    # physics features from training data — loss only
                    fdev = feat.to(DEV)
                    phys_last = fdev[:, -1, IR_CH:IR_CH+1]
                    pred_abs = pred * wstd + wmean  # de-normalized
                    loss = loss + phys_reg(pred_abs[:, 0:1], cap[:, :, 0:1], phys_last)
                loss.backward()
                opt.step()
            if ep % 25 == 0:
                print(f"  E{ep} L={loss.item():.4f}", flush=True)

        model.eval()
        tc = scale([caps[test]])[0]
        entry = {"config": cfg, "epochs": args.epochs, "seed": SEED, "params": n_params,
                 "train_loss": round(float(loss.item()), 4), "timestamp": time.strftime("%Y-%m-%d %H:%M")}
        if phys_reg:
            entry["physics"] = {
                "lambda": 0.1,
                "gamma_ir": round(float(nn.functional.softplus(phys_reg.gamma_ir).item()), 6),
                "gamma_t": round(float(nn.functional.softplus(phys_reg.gamma_t).item()), 6),
                "Ea_J_per_mol": round(float(torch.exp(phys_reg.Ea_log).item() * 1000), 1),
                "base": round(float(phys_reg.base.item()), 6),
            }
        # smoke guard: short runs (<10 epochs) never overwrite real ckpts
        smoke = "_smoke" if args.epochs < 10 else ""
        ckpt_path = Path(__file__).parent.parent / "checkpoints" / \
            f"abl_{args.dataset}_{cfg['name']}{smoke}.pt"
        ckpt_path.parent.mkdir(exist_ok=True)
        torch.save(model.state_dict(), str(ckpt_path))
        if cfg["K"] == 1:
            entry["table_a"] = eval_table_a(model, tc, W, DS["sps"], DS["eol_ah"], lo, hi)
            # K=1 single-step autoregressive rollout — I4 contrast baseline
            entry["rul"] = eval_rul(model, tc, W, 1, DS["sps"], DS["eol_ah"], caps[test], lo, hi)
        else:
            entry["traj"] = eval_traj(model, tc, W, cfg["K"], DS["sps"])
            entry["rul"] = eval_rul(model, tc, W, cfg["K"], DS["sps"], DS["eol_ah"], caps[test], lo, hi)
        results[cfg["name"]] = entry
        results_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"  -> saved to results/ablation.json: {cfg['name']}", flush=True)

    print("\n=== SUMMARY ===")
    sps = DS["sps"]
    for name, e in results.items():
        ta = e.get("table_a", {})
        tr = e.get("traj", {})
        ru = e.get("rul", {})
        ph = e.get("physics", {})
        extra = f" g_ir={ph.get('gamma_ir','?'):.4f}" if ph else ""
        if ta:
            parts = []
            for sp in sps:
                v = ta.get(f'traj_MAE_sp{sp}', None)
                parts.append(f'{v:.4f}' if isinstance(v, (int,float)) else '?')
            mae_str = "/".join(parts)
            print(f"{name:<24} R2={ta['R2']:.4f} MAE={mae_str}{extra}")
        elif tr:
            traj_mae = tr.get(f"sp{sps[0]}",{}).get("mae","?")
            rul_mae = "/".join(f"{ru.get(f'RUL_sp{sp}',{}).get('rul_mae','?'):.1f}" for sp in sps)
            print(f"{name:<24} traj_MAE={traj_mae:.4f} RUL_MAE={rul_mae}")


if __name__ == "__main__":
    main()
