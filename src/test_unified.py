"""Unified S_L readout: one 32-query model, evaluated in both single-step and multi-step modes.

Single-step: take query[0] output only, rolling prediction + EOL crossing
Multi-step: all 32 queries, trajectory MAE/RMSE/R2

Target: one model replaces both single-step and multi-step - truly unified.
"""
import sys, numpy as np, torch, torch.nn as nn
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from gdn_v2 import GDN2Block
from load_datasets import load_calce_cells_multivar
from data_pipeline import Seq2VecDataset
from torch.utils.data import DataLoader
import logging; logging.basicConfig(level=logging.WARNING)

DEV = torch.device("cuda")
W, OUT, BATCH, EPOCHS, STRIDE = 64, 32, 64, 100, 1
TRAIN = ["CS2_36", "CS2_37", "CS2_38"]
TEST = "CS2_35"

src = open(Path(__file__).parent / "test_sl_readout.py", encoding="utf-8").read()
cls_end = src.index("def masked_mae")
exec(src[:cls_end])


def masked_mae(pred, target, mask=None):
    err = (pred - target).abs()
    if mask is not None:
        err = err * mask
        return err.sum() / (mask.sum() + 1e-8)
    return err.mean()


def traj_metrics(pred, tgt):
    mae = np.mean(np.abs(pred - tgt))
    rmse = np.sqrt(np.mean((pred - tgt) ** 2))
    r2 = 1 - np.sum((tgt - pred) ** 2) / (np.sum((tgt - tgt.mean()) ** 2) + 1e-8)
    return mae, rmse, r2


print("Loading...")
caps_all, _, _ = load_calce_cells_multivar()
caps = {c: caps_all[c].copy() for c in caps_all}
all_tr = np.concatenate([caps[c] for c in TRAIN])
lo, hi = all_tr.min(), all_tr.max()


def scale(seqs):
    return [(s - lo) / (hi - lo) for s in seqs]


tr = Seq2VecDataset(scale([caps[c] for c in TRAIN]), W, OUT, STRIDE)
ld = DataLoader(tr, BATCH, shuffle=True)
model = SLReadoutSeq2Vec(out_len=OUT).to(DEV)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
print(f"[unified] {len(tr)} samples, {sum(p.numel() for p in model.parameters()):,} params", flush=True)
for ep in range(EPOCHS):
    model.train()
    for cap, feat, tgt, msk in ld:
        cap, tgt, msk = cap.to(DEV), tgt.to(DEV), msk.to(DEV)
        opt.zero_grad()
        loss = masked_mae(model(cap), tgt, msk)
        loss.backward()
        opt.step()
    if ep % 25 == 0:
        print(f"  E{ep} L={loss.item():.4f}", flush=True)

model.eval()
tc = scale([caps[TEST]])[0]
eol_n = (0.77 - lo) / (hi - lo)
te = int(np.argmax(tc < eol_n))

# --- single-step: query[0] only, rolling + EOL ---
print("\n=== single-step (query[0] only, rolling) ===")
preds = []
with torch.no_grad():
    for i in range(W, len(tc)):
        cin = torch.tensor(tc[i - W:i], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEV)
        preds.append(model(cin).squeeze(0)[0].item())
preds = np.array(preds); tv = tc[W:]; pv = preds[:len(tv)]
r2_1 = 1 - np.sum((tv - pv) ** 2) / np.sum((tv - tv.mean()) ** 2)
aes = []
for sp in [300, 400, 500]:
    sp_preds = preds[sp - W:]; pe = -1
    for j in range(len(sp_preds) - 1):
        if sp_preds[j] >= eol_n > sp_preds[j + 1]:
            pe = sp + j + (eol_n - sp_preds[j]) / (sp_preds[j + 1] - sp_preds[j] + 1e-8); break
        elif sp_preds[j] < eol_n: pe = sp + j; break
    aes.append(abs(te - pe) if pe >= 0 else -1)
print(f"R2={r2_1:.4f} AE={aes[0]:.0f}/{aes[1]:.0f}/{aes[2]:.0f}")
print(f"baseline: last-token AE=2/2/2 | S_L 1query AE=10/10/10")

# --- multi-step: all 32 queries, trajectory ---
print("\n=== multi-step (all 32 queries, trajectory) ===")
print(f"{'SP':>4} {'MAE':>8} {'RMSE':>8} {'R2':>8}  first6 pred vs targ")
with torch.no_grad():
    for sp in [300, 400, 500]:
        tgt = tc[sp:sp + OUT]
        cin = torch.tensor(tc[sp - W:sp], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEV)
        pred = model(cin).squeeze(0).cpu().numpy()
        mae, rmse, r2 = traj_metrics(pred, tgt)
        print(f"{sp:>4} {mae:>8.4f} {rmse:>8.4f} {r2:>8.3f}  "
              f"{np.round(pred[:6], 3)} vs {np.round(tgt[:6], 3)}")
print("done")
