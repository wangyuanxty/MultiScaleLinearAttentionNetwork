"""CALCE: quantile prediction (I4) — P10/P50/P90 with Pinball loss.

Main line + num_quantiles=3. Metrics:
  - P50 AE (single-step rolling, EOL crossing) — should match point model (~6-7)
  - P10/P90 coverage: fraction of true values inside [P10, P90] — should be ~80%
  - interval width: uncertainty magnitude
"""
import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from gdn_model import build_gdn_model, PinballLoss
from load_datasets import load_calce_cells_multivar
from data_pipeline import Seq2VecDataset
from torch.utils.data import DataLoader
import logging; logging.basicConfig(level=logging.WARNING)

DEV = torch.device("cuda")
W, OUT, BATCH, EPOCHS = 64, 1, 64, 100
TRAIN = ["CS2_36", "CS2_37", "CS2_38"]
TEST = "CS2_35"
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

print("Loading...")
caps_all, feats_all, _ = load_calce_cells_multivar()
caps = {c: caps_all[c].copy() for c in caps_all}
keep = [0, 1, 2, 3, 9]  # V_mean, V_min, V_max, IR, dt
feats = {c: feats_all[c][:, keep].copy() for c in feats_all}

all_tr = np.concatenate([caps[c] for c in TRAIN])
lo, hi = all_tr.min(), all_tr.max()


def scale(seqs):
    return [(s - lo) / (hi - lo) for s in seqs]


tr = Seq2VecDataset(scale([caps[c] for c in TRAIN]), W, OUT, 1, [feats[c] for c in TRAIN])
ld = DataLoader(tr, BATCH, shuffle=True)
model = build_gdn_model(
    multiscale=True, input_dim=6, window_size=W, output_len=OUT,
    readout="last", num_quantiles=3,
).to(DEV)
crit = PinballLoss(quantiles=(0.1, 0.5, 0.9)).to(DEV)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
print(f"[quantile P10/50/90] {len(tr)} samples, {sum(p.numel() for p in model.parameters()):,} params", flush=True)
for ep in range(EPOCHS):
    model.train()
    for cap, feat, tgt, msk in ld:
        cap, tgt, msk = cap.to(DEV), tgt.to(DEV), msk.to(DEV)
        x = torch.cat([cap, feat.to(DEV)], dim=-1)
        opt.zero_grad()
        loss = crit(model(x), tgt, msk)  # (B,3,K) vs (B,K)
        loss.backward()
        opt.step()
    if ep % 25 == 0:
        print(f"  E{ep} L={loss.item():.4f}", flush=True)

model.eval()
tc = scale([caps[TEST]])[0]
eol_n = (0.77 - lo) / (hi - lo)
te = int(np.argmax(tc < eol_n))
ftest = feats[TEST]
q_preds = []
with torch.no_grad():
    for i in range(W, len(tc)):
        cin = torch.tensor(tc[i - W:i], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEV)
        fin = torch.tensor(ftest[i - W:i], dtype=torch.float32).unsqueeze(0).to(DEV)
        q_preds.append(model(torch.cat([cin, fin], dim=-1)).squeeze(0).squeeze(-1).cpu().numpy())
q_preds = np.array(q_preds)  # (N, 3): P10, P50, P90
tv = tc[W:]
N = len(tv)

# P50 accuracy (single-step rolling EOL crossing)
p50 = q_preds[:N, 1]
r2 = 1 - np.sum((tv - p50) ** 2) / np.sum((tv - tv.mean()) ** 2)
aes = []
for sp in [300, 400, 500]:
    sp_preds = p50[sp - W:]; pe = -1
    for j in range(len(sp_preds) - 1):
        if sp_preds[j] >= eol_n > sp_preds[j + 1]:
            pe = sp + j + (eol_n - sp_preds[j]) / (sp_preds[j + 1] - sp_preds[j] + 1e-8); break
        elif sp_preds[j] < eol_n: pe = sp + j; break
    aes.append(abs(te - pe) if pe >= 0 else -1)

# Coverage: true inside [P10, P90]
p10 = q_preds[:N, 0]; p90 = q_preds[:N, 2]
cov = np.mean((tv >= p10) & (tv <= p90))
width = np.mean(p90 - p10)
thirds = np.array_split(np.arange(N), 3)
covs = [np.mean((tv[i] >= p10[i]) & (tv[i] <= p90[i])) for i in thirds]

print(f"\nP50: R2={float(r2):.4f} AE={int(aes[0]):d}/{int(aes[1]):d}/{int(aes[2]):d}")
print(f"P10-P90 coverage: {float(cov):.3f} (target 0.80)")
print(f"interval width: {float(width):.4f} (normalized capacity)")
print(f"coverage by third: {[f'{float(c):.2f}' for c in covs]}")
print(f"baseline point model: AE=2/2/2 (last-token) or 6-7 (multi-scale+5ch)")
print("done")
