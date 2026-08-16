"""MIT subset validation: 8 cells, 7 train / 1 test, same protocol as CALCE.

Verifies pipeline learnability + eval protocol before full 43-cell run.
MIT: LFP, features [IR, QCharge, Tavg, Tmax, Tmin, chargetime] (6ch), EOL=0.8.
"""
import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from gdn_model import build_gdn_model, masked_mae
from load_datasets import load_mit_stanford_multivar
from data_pipeline import Seq2VecDataset
from torch.utils.data import DataLoader
import logging; logging.basicConfig(level=logging.WARNING)

DEV = torch.device("cuda")
W, OUT, BATCH, EPOCHS = 64, 1, 64, 80
SEED = 42
EOL_PCT = 0.80

torch.manual_seed(SEED)
np.random.seed(SEED)

print("Loading MIT...")
caps, feats, fd = load_mit_stanford_multivar()
names = sorted(caps.keys())
rng = np.random.default_rng(SEED)
subset = rng.choice(names, size=8, replace=False)
subset = sorted(subset)
TEST = subset[-1]
TRAIN = [c for c in subset if c != TEST]
print(f"subset: {TRAIN} -> test {TEST}")

all_tr = np.concatenate([caps[c] for c in TRAIN])
lo, hi = all_tr.min(), all_tr.max()


def scale(seqs):
    return [(s - lo) / (hi - lo) for s in seqs]


tr = Seq2VecDataset(scale([caps[c] for c in TRAIN]), W, OUT, 1, [feats[c] for c in TRAIN])
ld = DataLoader(tr, BATCH, shuffle=True)
model = build_gdn_model(
    multiscale=True, input_dim=1 + fd, window_size=W, output_len=OUT,
    readout="last",
).to(DEV)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
print(f"[MIT subset] {len(tr)} samples, {sum(p.numel() for p in model.parameters()):,} params", flush=True)
for ep in range(EPOCHS):
    model.train()
    for cap, feat, tgt, msk in ld:
        cap, tgt, msk = cap.to(DEV), tgt.to(DEV), msk.to(DEV)
        x = torch.cat([cap, feat.to(DEV)], dim=-1)
        opt.zero_grad()
        loss = masked_mae(model(x), tgt, msk)
        loss.backward()
        opt.step()
    if ep % 20 == 0:
        print(f"  E{ep} L={loss.item():.4f}", flush=True)

model.eval()
tc = scale([caps[TEST]])[0]
eol_n = (EOL_PCT * caps[TEST][0] - lo) / (hi - lo)  # EOL at 80% of initial
te = int(np.argmax(tc < eol_n))
ftest = feats[TEST]
preds = []
with torch.no_grad():
    for i in range(W, len(tc)):
        cin = torch.tensor(tc[i - W:i], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEV)
        fin = torch.tensor(ftest[i - W:i], dtype=torch.float32).unsqueeze(0).to(DEV)
        preds.append(model(torch.cat([cin, fin], dim=-1)).item())
preds = np.array(preds); tv = tc[W:]; pv = preds[:len(tv)]
r2 = 1 - np.sum((tv - pv) ** 2) / np.sum((tv - tv.mean()) ** 2)
aes = []
for sp in [int(len(tc) * 0.5), int(len(tc) * 0.6), int(len(tc) * 0.7)]:
    sp_preds = pv[sp - W:]; pe = -1
    for j in range(len(sp_preds) - 1):
        if sp_preds[j] >= eol_n > sp_preds[j + 1]:
            pe = sp + j + (eol_n - sp_preds[j]) / (sp_preds[j + 1] - sp_preds[j] + 1e-8); break
        elif sp_preds[j] < eol_n: pe = sp + j; break
    aes.append(abs(te - pe) if pe >= 0 else -1)
print(f"\n[MIT subset] test={TEST} true_EOL={te} R2={float(r2):.4f}")
# Table A: trajectory MAE/RMSE from SP-64 onwards
for sp in [int(len(tc) * 0.5), int(len(tc) * 0.6), int(len(tc) * 0.7)]:
    seg = pv[sp - W:]; seg_t = tc[sp:]
    n = min(len(seg), len(seg_t))
    mae = np.mean(np.abs(seg[:n] - seg_t[:n]))
    rmse = np.sqrt(np.mean((seg[:n] - seg_t[:n]) ** 2))
    print(f"  SP~{sp}: traj_MAE={float(mae):.4f} traj_RMSE={float(rmse):.4f}")
print("done")
