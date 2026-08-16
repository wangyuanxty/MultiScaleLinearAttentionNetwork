"""Our model on the RUL-Mamba/PatchFormer NASA protocol: SP=50/70/90, window 30.

Fair 3-way NASA comparison: RUL-Mamba (their script, reported AE 0.8-2.5),
PatchFormer (reported AE 5.1-7.9), Ours (this script).
"""
import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from gdn_model import build_gdn_model, masked_mae
from load_datasets import load_nasa_capacity
from data_pipeline import Seq2VecDataset
from torch.utils.data import DataLoader
import logging; logging.basicConfig(level=logging.WARNING)

DEV = torch.device("cuda")
W, OUT, BATCH, EPOCHS = 30, 1, 64, 100
TRAIN = ["B0006", "B0007", "B0018"]
TEST = "B0005"
EOL_AH = 1.4
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

print("Loading...")
caps = {b: load_nasa_capacity(b) for b in TRAIN + [TEST]}
all_tr = np.concatenate([caps[c] for c in TRAIN])
lo, hi = all_tr.min(), all_tr.max()
scale = lambda s: [(x - lo) / (hi - lo) for x in s]

tr = Seq2VecDataset(scale([caps[c] for c in TRAIN]), W, OUT, 1)
ld = DataLoader(tr, BATCH, shuffle=True)
model = build_gdn_model(multiscale=False, input_dim=1, window_size=W,
                        output_len=OUT, readout="last").to(DEV)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
print(f"[ours on NASA protocol] {len(tr)} samples, {sum(p.numel() for p in model.parameters()):,} params", flush=True)
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
eol_n = (EOL_AH - lo) / (hi - lo)
te = int(np.argmax(tc < eol_n))

# Aligned protocol: from each SP, roll with feedback (autoregressive) until EOL crossing
aes = []
for sp in [50, 70, 90]:
    window = tc[sp - W:sp].copy()
    n_steps = 0
    with torch.no_grad():
        while n_steps < 600:
            cin = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEV)
            p = model(cin).item()
            n_steps += 1
            window = np.concatenate([window[1:], [p]])
            if p < eol_n:
                break
    pe = sp + n_steps
    ae = abs(te - pe) if n_steps < 600 else -1
    aes.append(ae)
    print(f"  SP={sp}: rolled {n_steps} steps, pred EOL={pe}, AE={ae}", flush=True)
print(f"\n[ours NASA protocol, aligned] true_EOL={te} AE={aes[0]}/{aes[1]}/{aes[2]}")
print(f"RUL-Mamba (reported): AE=0.8/0.9/2.5 | PatchFormer (reported): AE=5.1/5.6/7.9")
print("done")
