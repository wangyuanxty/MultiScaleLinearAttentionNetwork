"""RUL-Mamba (core module) on OUR CALCE protocol: window 64, rolling single-step, EOL AE.

Uses the pure-PyTorch RULMamba module (no pytorch_forecasting dependency),
trained and evaluated with our exact protocol for a fair comparison.
"""
import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "reference_repos/ref_rul_mamba"))
sys.path.insert(0, str(Path(__file__).parent.parent / "reference_repos/ref_rul_mamba" / "Models"))
from RULMamba import RULMamba  # noqa: E402
from load_datasets import load_calce_cells_multivar  # noqa: E402
from data_pipeline import Seq2VecDataset  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
import logging; logging.basicConfig(level=logging.WARNING)

DEV = torch.device("cuda")
W, OUT, BATCH, EPOCHS = 30, 1, 64, 150
TRAIN = ["CS2_36", "CS2_37", "CS2_38"]
TEST = "CS2_35"
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

print("Loading...")
caps_all, _, _ = load_calce_cells_multivar()
caps = {c: caps_all[c].copy() for c in caps_all}
all_tr = np.concatenate([caps[c] for c in TRAIN])
lo, hi = all_tr.min(), all_tr.max()


def scale(seqs):
    return [(s - lo) / (hi - lo) for s in seqs]


tr = Seq2VecDataset(scale([caps[c] for c in TRAIN]), W, OUT, 1)
ld = DataLoader(tr, BATCH, shuffle=True)
model = RULMamba(enc_in=1, d_model=48, n_dec_layer=2, dropout=0.0615).to(DEV)
opt = torch.optim.Adam(model.parameters(), lr=2.2e-3)
print(f"[RULMamba on our protocol] {len(tr)} samples, {sum(p.numel() for p in model.parameters()):,} params", flush=True)
for ep in range(EPOCHS):
    model.train()
    for cap, feat, tgt, msk in ld:
        cap, tgt, msk = cap.to(DEV), tgt.to(DEV), msk.to(DEV)
        opt.zero_grad()
        pred = model(x_enc=cap, x_dec=None)  # (B, 1, 1)
        # SMAPE loss (their loss function)
        denom = (pred.squeeze(-1).abs() + tgt.unsqueeze(-1).abs()) / 2 + 1e-8
        loss = ((pred.squeeze(-1) - tgt.unsqueeze(-1)).abs() / denom * msk.unsqueeze(-1)).sum() / (msk.sum() + 1e-8)
        loss.backward()
        opt.step()
    if ep % 25 == 0:
        print(f"  E{ep} L={loss.item():.4f}", flush=True)

model.eval()
tc = scale([caps[TEST]])[0]
eol_n = (0.77 - lo) / (hi - lo)
te = int(np.argmax(tc < eol_n))
preds = []
with torch.no_grad():
    for i in range(W, len(tc)):
        cin = torch.tensor(tc[i - W:i], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEV)
        preds.append(model(x_enc=cin, x_dec=None).squeeze().item())
preds = np.array(preds); tv = tc[W:]; pv = preds[:len(tv)]
r2 = 1 - np.sum((tv - pv) ** 2) / np.sum((tv - tv.mean()) ** 2)
aes = []
for sp in [300, 400, 500]:
    sp_preds = pv[sp - W:]; pe = -1
    for j in range(len(sp_preds) - 1):
        if sp_preds[j] >= eol_n > sp_preds[j + 1]:
            pe = sp + j + (eol_n - sp_preds[j]) / (sp_preds[j + 1] - sp_preds[j] + 1e-8); break
        elif sp_preds[j] < eol_n: pe = sp + j; break
    aes.append(abs(te - pe) if pe >= 0 else -1)
print(f"\n[RULMamba] R2={float(r2):.4f} AE={aes[0]}/{aes[1]}/{aes[2]}")
print("ours (GDN-2 last-token): AE=2/2/2 | PatchFormer: AE=0.8-1.1")
print("done")
