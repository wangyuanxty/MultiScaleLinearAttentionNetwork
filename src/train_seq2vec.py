import argparse, json, logging
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(r"D:/research/degradation_prognostics/Transformer_and_Multi_Scale_Models/src")))
from data_pipeline import SlidingWindowBuilder, collate_seq2vec
from gdn_model import build_gdn_model, PinballLoss, masked_mae
from load_datasets import (
    load_nasa_cells_multivar, load_calce_cells_multivar,
    load_mit_stanford_multivar, load_panasonic_cells, load_gotion_cells
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = Path(r"D:/research/degradation_prognostics/Transformer_and_Multi_Scale_Models/checkpoints")
CKPT.mkdir(exist_ok=True)

CFG = {
    "nasa": {"window": 30, "output_len": 17, "stride": 8, "rated_cap": 2.0, "eol_pct": 0.70},
    "calce": {"window": 64, "output_len": 120, "stride": 8, "rated_cap": 1.1, "eol_pct": 0.70},
    "panasonic": {"window": 64, "output_len": 107, "stride": 8, "rated_cap": None, "eol_pct": 0.70},
    "gotion": {"window": 64, "output_len": 169, "stride": 8, "rated_cap": None, "eol_pct": 0.80},
    "mit": {"window": 64, "output_len": 100, "stride": 8, "rated_cap": 1.1, "eol_pct": 0.80},
}
print("Config OK")

def load_dataset(name):
    c = CFG[name]
    if name == "nasa":
        raw = load_nasa_cells_multivar()
        caps = {b: raw[b]["capacity"] for b in raw}
        feats = {}
        for b in raw:
            r = raw[b]
            feats[b] = np.stack([r[k] for k in ["V_mean","I_mean","T_mean","T_max","Re","Rct","delta_t"]], axis=-1).astype(np.float32)
        return caps, feats, 7
    elif name == "calce":
        caps, feats, fd = load_calce_cells_multivar()
        return caps, feats, fd
    elif name == "mit":
        caps, feats, fd = load_mit_stanford_multivar()
        return caps, feats, fd
    elif name == "panasonic":
        return load_panasonic_cells(), None, 0
    elif name == "gotion":
        return load_gotion_cells(), None, 0
    raise ValueError(name)

def build_input(cap, feat):
    if feat is not None and feat.shape[-1] > 1:
        return torch.cat([cap, feat], dim=-1)
    return cap

def train_epoch(model, loader, criterion, opt):
    model.train(); tot, n = 0.0, 0
    for cap, feat, tgt, msk in loader:
        cap, tgt, msk = cap.to(DEVICE), tgt.to(DEVICE), msk.to(DEVICE)
        feat = feat.to(DEVICE) if feat is not None and feat.shape[-1] > 1 else None
        x = build_input(cap, feat); opt.zero_grad()
        pred = model(x)
        if pred.dim() == 3:
            loss = criterion(pred, tgt.unsqueeze(1).expand(-1, pred.size(1), -1), msk.unsqueeze(1).expand(-1, pred.size(1), -1))
        else:
            loss = masked_mae(pred, tgt, msk)
        loss.backward(); opt.step()
        tot += loss.item() * cap.size(0); n += cap.size(0)
    return tot / n

@torch.no_grad()
def validate(model, loader, criterion):
    model.eval(); tot, n = 0.0, 0
    for cap, feat, tgt, msk in loader:
        cap, tgt, msk = cap.to(DEVICE), tgt.to(DEVICE), msk.to(DEVICE)
        feat = feat.to(DEVICE) if feat is not None and feat.shape[-1] > 1 else None
        x = build_input(cap, feat)
        pred = model(x)
        if pred.dim() == 3:
            loss = criterion(pred, tgt.unsqueeze(1).expand(-1, pred.size(1), -1), msk.unsqueeze(1).expand(-1, pred.size(1), -1))
        else:
            loss = masked_mae(pred, tgt, msk)
        tot += loss.item() * cap.size(0); n += cap.size(0)
    return tot / n

@torch.no_grad()
def eval_rul(model, caps_norm, feats_norm, window, out_len, stride, eol_thr):
    model.eval()
    caps = np.asarray(caps_norm, dtype=np.float32); n = len(caps)
    true_eol = int(np.argmax(caps < eol_thr))
    if true_eol == 0 and caps[0] >= eol_thr: return None
    results = []
    for sp in range(window, n - 10, max(1, (n - window) // 15)):
        cin = torch.tensor(caps[sp - window:sp], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEVICE)
        if feats_norm is not None:
            fin = torch.tensor(feats_norm[sp - window:sp], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            x = torch.cat([cin, fin], dim=-1)
        else: x = cin
        pred = model(x)
        if pred.dim() == 3: pred = pred[:, 1, :]
        pred = pred.squeeze(0).cpu().numpy()
        eol_pred = -1
        for j in range(len(pred) - 1):
            if pred[j] >= eol_thr > pred[j + 1]:
                frac = (eol_thr - pred[j]) / (pred[j + 1] - pred[j] + 1e-8)
                eol_pred = sp + (j + frac) * stride; break
            elif pred[j] < eol_thr: eol_pred = sp + j * stride; break
        if eol_pred >= 0: results.append((abs(true_eol - eol_pred), abs(true_eol - eol_pred) / true_eol))
    if not results: return None
    aes, res = zip(*results)
    return {"eol_mae": float(np.mean(aes)), "eol_rmse": float(np.sqrt(np.mean(np.array(aes)**2))), "eol_mre": float(np.mean(res)), "sp_count": len(results)}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="nasa", choices=["nasa","calce","panasonic","gotion","mit"])
    p.add_argument("--test_cell", default=None)
    p.add_argument("--multiscale", action="store_true")
    p.add_argument("--cross_exchange", action="store_true")
    p.add_argument("--use_features", action="store_true")
    p.add_argument("--num_quantiles", type=int, default=1, choices=[1, 3])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=25)
    args = p.parse_args()

    cfg = CFG[args.dataset]; tc = args.test_cell
    uf = args.use_features and args.dataset in ("nasa","calce","mit")
    caps, feats, fd = load_dataset(args.dataset)
    if tc is None: tc = sorted(caps.keys())[0]
    tc_raw = caps[tc].copy(); tf_raw = feats[tc].copy() if feats else None

    builder = SlidingWindowBuilder(cfg["window"], cfg["output_len"], cfg["stride"])
    tr, va, te = builder.build_cell_disjoint(caps, tc, cell_features=feats)
    tr_ld = DataLoader(tr, args.batch_size, shuffle=True, collate_fn=collate_seq2vec)
    va_ld = DataLoader(va, args.batch_size, collate_fn=collate_seq2vec)
    te_ld = DataLoader(te, args.batch_size, collate_fn=collate_seq2vec)

    rc = cfg["rated_cap"] or tc_raw[0]
    all_tr = np.concatenate([caps[c] for c in caps if c != tc])
    lo, hi = all_tr.min(), all_tr.max()
    eol_n = (rc * cfg["eol_pct"] - lo) / (hi - lo)
    tc_norm = (tc_raw - lo) / (hi - lo) if hi > lo else tc_raw

    in_dim = 1 + (fd if uf else 0)
    model = build_gdn_model(
        multiscale=args.multiscale, cross_exchange=args.cross_exchange,
        num_quantiles=args.num_quantiles,
        window_size=cfg["window"], output_len=cfg["output_len"],
        input_dim=in_dim,
    ).to(DEVICE)
    logger.info(f"{args.dataset}/{tc} dim={in_dim} params={sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Train={len(tr)} Val={len(va)} Test={len(te)} EOL={eol_n:.4f}")

    crit = PinballLoss() if args.num_quantiles == 3 else None
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best_vl, pat = float("inf"), 0

    for ep in range(args.epochs):
        tl = train_epoch(model, tr_ld, crit, opt)
        vl = validate(model, va_ld, crit); sch.step()
        if vl < best_vl: best_vl = vl; pat = 0; torch.save(model.state_dict(), CKPT / f"best_{args.dataset}_{tc}.pt")
        else: pat += 1
        if ep % 10 == 0: logger.info(f"E{ep}: train={tl:.4f} val={vl:.4f}")
        if pat >= args.patience: logger.info(f"Early stop @ {ep}"); break

    model.load_state_dict(torch.load(CKPT / f"best_{args.dataset}_{tc}.pt"))
    tst = validate(model, te_ld, crit)
    logger.info(f"Test loss: {tst:.6f}")

    rul = eval_rul(model, tc_norm, tf_raw, cfg["window"], cfg["output_len"], cfg["stride"], eol_n)
    if rul: logger.info(f"RUL: MAE={rul['eol_mae']:.1f} RMSE={rul['eol_rmse']:.1f} MRE={rul['eol_mre']:.4f}")
    else: logger.info("RUL: N/A")

    res = {"dataset": args.dataset, "test_cell": tc, "test_loss": float(tst),
           "eol_mae": rul["eol_mae"] if rul else -1, "eol_rmse": rul["eol_rmse"] if rul else -1,
           "params": sum(p.numel() for p in model.parameters()),
           "multiscale": args.multiscale, "cross_exchange": args.cross_exchange,
           "use_features": uf, "num_quantiles": args.num_quantiles,
           "timestamp": datetime.now().isoformat()}
    with open(CKPT / f"results_{args.dataset}_{tc}.json", "w") as f: json.dump(res, f, indent=2)
    logger.info(f"Saved results_{args.dataset}_{tc}.json")

if __name__ == "__main__": main()
