"""Unified multi-dataset trainer — K=1 SOH + K=32 RUL.
Usage: python test_unified_train.py --dataset calce [--epochs 100] [--k 1|32|both]
Output: results/unified_train.json"""
import sys, json, time, argparse, numpy as np, torch, torch.nn as nn
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from gdn_model import build_gdn_model, masked_mae, PhysicsRegularizer
from load_datasets import (load_calce_cells_multivar, load_nasa_multivar,
                           load_mit_stanford, load_panasonic_cells, load_gotion_cells, load_tju_cells,
                           load_nasa_capacity)
from data_pipeline import Seq2VecDataset, collate_seq2vec
from torch.utils.data import DataLoader
import logging; logging.basicConfig(level=logging.WARNING)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH, SEED = 64, 42

DATASETS = {
    "calce": {"train": ["CS2_36","CS2_37","CS2_38"], "test": "CS2_35",
              "W": 64, "sps": [300,400,500], "eol_ah": 0.77,  # rated 1.1 × 0.70
              "feat_dim": 5, "ir_ch": 3},
    "nasa":  {"train": ["B0006","B0007","B0018"], "test": "B0005",
              "W": 30, "sps": [50,70,90], "eol_ah": 1.40,  # rated 2.0 × 0.70
              "feat_dim": 6, "ir_ch": 5},
    "mit":   {"W": 64, "sps": [200, 300, 400], "eol_ah": 0.86,  # rated 1.074 × 0.80
              "feat_dim": None, "ir_ch": None},
    "mit": {"W": 64, "sps": [200, 300, 400], "eol_ah": 0.86,
                   "feat_dim": None, "ir_ch": None},
    "panasonic": {"W": 30, "sps": [300, 500, 700], "eol_ah": 2.12,  # rated 3.03 × 0.70
                  "feat_dim": None, "ir_ch": None},
    "gotion": {"W": 30, "sps": [500, 800, 1100], "eol_ah": 21.60,  # rated 27 × 0.80
               "feat_dim": None, "ir_ch": None},
    "tju":   {"W": 64, "sps": [200, 300, 400], "eol_ah": 1.75,  # rated 2.5 × 0.70
              "feat_dim": None, "ir_ch": None},
}


def load_dataset(name):
    cfg = DATASETS[name]
    if name == "calce":
        caps_all, feats_all, _ = load_calce_cells_multivar()
        caps = {c: caps_all[c].copy().astype(np.float32) for c in caps_all}
        feats = {c: feats_all[c][:, 3:4].copy().astype(np.float32) for c in feats_all}  # IR
        return caps, feats, cfg["train"], cfg["test"], cfg
    elif name == "nasa":
        bats = cfg["train"] + [cfg["test"]]
        caps, feats = {}, {}
        for bat in bats:
            data = load_nasa_multivar(bat)
            caps[bat] = data['capacity'].astype(np.float32)
            # physics loss only: [IR(Re), T(T_mean)]
            feats[bat] = np.stack([data['Re'], data['T_mean']], axis=-1).astype(np.float32)
        return caps, feats, cfg["train"], cfg["test"], cfg
    elif name == "mit":
        caps = {k: v.astype(np.float32) for k, v in load_mit_stanford().items()}
        cells = sorted(caps.keys())
        return caps, None, cells[:-1], cells[-1], cfg
    elif name == "mit":
        from load_datasets import MIT_TRAIN_CELLS, MIT_TEST_CELLS
        caps_all = {k: v.astype(np.float32) for k, v in load_mit_stanford().items()}
        train_cells = MIT_TRAIN_CELLS  # 35 cells (80/20 full dataset)
        # caps include the first test cell for the in-training sanity
        # eval; the paper numbers come from eval_multiseed (8 cells).
        caps = {c: caps_all[c] for c in train_cells + [MIT_TEST_CELLS[0]]}
        return caps, None, train_cells, MIT_TEST_CELLS[0], cfg
    elif name == "panasonic":
        caps = {k: v.astype(np.float32) for k, v in load_panasonic_cells().items()}
        cells = sorted(caps.keys())
        return caps, None, cells[:-1], cells[-1], cfg
    elif name == "gotion":
        # Cell01 only reaches EOL (21.6Ah); Cell02/03 never dip below — align with OmniTIEFormer
        caps_all = {k: v.astype(np.float32) for k, v in load_gotion_cells().items()}
        train_cells, test_cell = ['Cell02', 'Cell03'], 'Cell01'
        caps = {c: caps_all[c] for c in train_cells + [test_cell]}
        return caps, None, train_cells, test_cell, cfg
    elif name == "tju":
        caps_all = load_tju_cells()
        train_cells, test_cell = ['CY25_2', 'CY25_3'], 'CY25_1'
        caps = {c: caps_all[c] for c in train_cells + [test_cell]}
        return caps, None, train_cells, test_cell, cfg
    raise ValueError(f"Unknown: {name}")


def eval_table_a(model, tc, W, sps, eol_ah=None, lo=None, hi=None):
    preds = []
    with torch.no_grad():
        for i in range(W, len(tc)):
            win = tc[i-W:i]
            wmean = float(win.mean()); wstd = float(win.std()) + 1e-6
            cin = torch.tensor(win, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEV)
            p_norm = model(cin).item()
            preds.append(p_norm * wstd + wmean)  # per-window de-normalize
    pv = np.array(preds)[:len(tc)-W]; tv = tc[W:]
    r2_global = 1 - np.sum((tv-pv)**2) / np.sum((tv-tv.mean())**2)
    out = {"R2": round(float(r2_global), 4)}
    if eol_ah is not None:
        threshold_n = (eol_ah - lo) / (hi - lo + 1e-8)
    for sp in (sps or []):
        seg = pv[sp-W:]; seg_t = tc[sp:]
        n = min(len(seg), len(seg_t))
        seg_p, seg_t = seg[:n], seg_t[:n]
        out[f"traj_MAE_sp{sp}"] = round(float(np.mean(np.abs(seg_p-seg_t))), 4)
        out[f"traj_RMSE_sp{sp}"] = round(float(np.sqrt(np.mean((seg_p-seg_t)**2))), 4)
        r2_sp = 1 - np.sum((seg_t-seg_p)**2) / (np.sum((seg_t-seg_t.mean())**2) + 1e-8)
        out[f"traj_R2_sp{sp}"] = round(float(r2_sp), 4)
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


def eval_rul(model, tc, W, K, sps, eol_ah, raw_cap, lo, hi):
    eol_n = (eol_ah - lo) / (hi - lo + 1e-8)
    te = int(np.argmax(raw_cap < eol_ah)) if (raw_cap < eol_ah).any() else len(raw_cap)
    out = {}
    with torch.no_grad():
        for sp in (sps or []):
            if sp >= te: continue
            window = tc[sp-W:sp].copy()
            traj = []
            while len(traj) < 800:
                cin = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEV)
                pred = model(cin).squeeze(0).cpu().numpy()
                traj.extend(pred.tolist())
                window = np.concatenate([window[K:], pred])
                if pred[-1] < eol_n: break
            traj = np.array(traj)
            pe = -1
            for j in range(len(traj)-1):
                if traj[j] >= eol_n > traj[j+1]:
                    pe = sp + j + (eol_n - traj[j]) / (traj[j+1] - traj[j] + 1e-8)
                    break
                elif traj[j] < eol_n:
                    pe = sp + j; break
            out[f"RUL_sp{sp}"] = {"rul_mae": round(float(abs(te-pe)),1) if pe>=0 else None,
                                   "true_eol": te, "pred_eol": round(float(pe),1) if pe>=0 else None}
    return out


def train_one(model, phys_reg, ld, epochs, ds_cfg):
    params = list(model.parameters())
    if phys_reg: params += list(phys_reg.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    for ep in range(epochs):
        model.train()
        for cap, feat, tgt, msk in ld:
            cap, tgt, msk = cap.to(DEV), tgt.to(DEV), msk.to(DEV)
            opt.zero_grad()
            pred = model(cap)  # pure-capacity input
            # per-window target scaling (PatchFormer-style): target is
            # normalized by the input window's mean/std
            wmean = cap.squeeze(-1).mean(dim=1, keepdim=True)
            wstd = cap.squeeze(-1).std(dim=1, keepdim=True) + 1e-6
            tgt_norm = (tgt - wmean) / wstd
            loss = masked_mae(pred, tgt_norm, msk)
            if phys_reg and feat is not None and ds_cfg.get("ir_ch") is not None:
                fdev = feat.to(DEV)
                # physics on DE-normalized prediction (absolute capacity)
                pred_abs = pred * wstd + wmean
                loss = loss + phys_reg(pred_abs[:,0:1], cap[:,:,0:1], fdev[:,-1, ds_cfg["ir_ch"]:ds_cfg["ir_ch"]+1])
            loss.backward(); opt.step()
        if ep % 25 == 0:
            print(f"  E{ep} L={loss.item():.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--k", default="both", choices=["1","8","16","32","64","both"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--physics", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip training if the seed checkpoint already exists")
    args = ap.parse_args()
    global SEED
    SEED = args.seed

    # fast skip BEFORE data loading (CALCE xlsx takes ~100 s)
    ckpt_dir = Path(__file__).parent.parent / "checkpoints"
    prefix0 = f"{args.dataset}" + ("_phys" if args.physics else "")
    ks0 = [1, 32] if args.k == "both" else [int(args.k)]
    smoke0 = "_smoke" if args.epochs < 10 else ""  # never overwrite real ckpts
    if args.skip_existing and all(
        (ckpt_dir / f"unified_{prefix0}_K{K}_seed{SEED}{smoke0}.pt").exists()
        for K in ks0
    ):
        print(f"skip: all checkpoints for {args.dataset} seed {SEED} exist")
        return

    caps, feats, train_cells, test_cell, ds_cfg = load_dataset(args.dataset)
    W, sps, eol_ah = ds_cfg["W"], ds_cfg["sps"], ds_cfg["eol_ah"]
    has_phys = feats is not None  # physics features exist (loss only)

    all_tr = np.concatenate([caps[c] for c in train_cells])
    lo, hi = all_tr.min(), all_tr.max()
    def scale(seqs): return [(s-lo)/(hi-lo) for s in seqs]

    tc_test = scale([caps[test_cell]])[0]

    out_file = Path(__file__).parent.parent / "results" / "unified_train.json"
    out_file.parent.mkdir(exist_ok=True)
    results = json.loads(out_file.read_text()) if out_file.exists() else {}
    prefix = f"{args.dataset}" + ("_phys" if args.physics else "")

    ckpt_dir = Path(__file__).parent.parent / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    smoke = "_smoke" if args.epochs < 10 else ""  # never overwrite real ckpts
    for K in ([1, 32] if args.k == "both" else [int(args.k)]):
        torch.manual_seed(SEED); np.random.seed(SEED)
        ckpt_path = ckpt_dir / f"unified_{prefix}_K{K}_seed{SEED}{smoke}.pt"
        if args.skip_existing and ckpt_path.exists():
            print(f"  skip: {ckpt_path.name} exists", flush=True)
            continue
        print(f"\n=== [{args.dataset}] K={K} ===", flush=True)
        # pure-capacity input for all; feats only feed physics loss
        feat_list = [feats[c] for c in train_cells] if has_phys else None
        tr = Seq2VecDataset(scale([caps[c] for c in train_cells]), W, K, 1, feat_list)
        ld = DataLoader(tr, BATCH, shuffle=True, collate_fn=collate_seq2vec)

        use_phys = args.physics and has_phys and ds_cfg.get("ir_ch") is not None
        # Final architecture: multi-scale + stage-query attention exchange (V3)
        model = build_gdn_model(
            multiscale=True, cross_exchange=False, stage_query=True,
            input_dim=1, window_size=W, output_len=K, use_physics=False,
            ir_ch=None, t_ch=None, readout="last",
        ).to(DEV)
        phys_reg = PhysicsRegularizer(lambda_=0.1).to(DEV) if use_phys else None
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  params={n_params:,} physics={use_phys}", flush=True)

        train_one(model, phys_reg, ld, args.epochs, ds_cfg)
        model.eval()

        key = f"{prefix}_K{K}"
        entry = {"dataset": args.dataset, "K": K, "epochs": args.epochs,
                 "seed": SEED, "params": n_params, "physics": use_phys,
                 "timestamp": time.strftime("%Y-%m-%d %H:%M")}

        if phys_reg:
            entry["physics_params"] = {
                "gamma_ir": round(float(nn.functional.softplus(phys_reg.gamma_ir).item()), 6),
                "gamma_t": round(float(nn.functional.softplus(phys_reg.gamma_t).item()), 6),
                "Ea_J_per_mol": round(float(torch.exp(phys_reg.Ea_log).item()*1000), 1),
                "base": round(float(phys_reg.base.item()), 6),
            }

        ckpt_dir = Path(__file__).parent.parent / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        torch.save(model.state_dict(),
                   str(ckpt_dir / f"unified_{prefix}_K{K}_seed{SEED}.pt"))
        if K == 1:
            entry["table_a"] = eval_table_a(model, tc_test, W, sps, eol_ah, lo, hi)
            # K=1 single-step AR rollout — I4 contrast baseline
            entry["rul"] = eval_rul(model, tc_test, W, 1, sps, eol_ah, caps[test_cell], lo, hi)
        else:
            entry["rul"] = eval_rul(model, tc_test, W, K, sps, eol_ah, caps[test_cell], lo, hi)

        results[key] = entry
        out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        tag = "table_a" if K==1 else "rul"
        print(f"  -> {key}: {json.dumps(entry[tag])}", flush=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
