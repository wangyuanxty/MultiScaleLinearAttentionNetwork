"""CALCE: S_L direct readout with length=1 — standard single-step eval.

Same model as test_sl_readout.py but OUT=1, evaluated like test_calce_multiscale
(rolling prediction, EOL crossing, AE@300/400/500). Isolates architecture health
from multi-step task difficulty. Baseline: last-token AE=2/2/2.
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
W, OUT, BATCH, EPOCHS, STRIDE = 64, 1, 64, 100, 1
TRAIN = ["CS2_36", "CS2_37", "CS2_38"]
TEST = "CS2_35"


class SLReadout1Step(nn.Module):
    """S_L direct readout, one query vector -> one capacity value."""

    def __init__(self, d_model=64, num_layers=2, out_len=1, num_heads=4,
                 head_dim=16, expand_v=2.0, conv_size=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, W, d_model) * 0.02)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'gdn': GDN2Block(d_model, head_dim, num_heads, expand_v, conv_size, dropout),
                'norm': nn.RMSNorm(d_model),
            }) for _ in range(num_layers)
        ])
        self.queries = nn.Parameter(torch.randn(out_len, num_heads, head_dim) * 0.02)
        self.head = nn.Sequential(
            nn.Dropout(0.1), nn.Linear(num_heads * int(head_dim * expand_v), 1),
        )

    def forward(self, x):
        B = x.size(0)
        h = self.input_proj(x) + self.pos_embed
        for i, l in enumerate(self.layers):
            out, S = l['gdn'](h, return_state=True)
            h = l['norm'](out + h)
        o = torch.einsum('bhds,khd->kbhs', S, self.queries)  # (K, B, H, Dv)
        o = o.transpose(0, 1).reshape(B, OUT, -1)            # (B, K, H*Dv)
        return self.head(o).squeeze(-1)                      # (B, K)


def masked_mae(pred, target, mask=None):
    err = (pred - target).abs()
    if mask is not None:
        err = err * mask
        return err.sum() / (mask.sum() + 1e-8)
    return err.mean()


print("Loading...")
caps_all, _, _ = load_calce_cells_multivar()
caps = {c: caps_all[c].copy() for c in caps_all}
all_tr = np.concatenate([caps[c] for c in TRAIN])
lo, hi = all_tr.min(), all_tr.max()


def scale(seqs):
    return [(s - lo) / (hi - lo) for s in seqs]


tr = Seq2VecDataset(scale([caps[c] for c in TRAIN]), W, OUT, STRIDE)
ld = DataLoader(tr, BATCH, shuffle=True)
model = SLReadout1Step(out_len=OUT).to(DEV)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
print(f"[S_L len=1] {len(tr)} samples, {sum(p.numel() for p in model.parameters()):,} params", flush=True)
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
preds = []
with torch.no_grad():
    for i in range(W, len(tc)):
        cin = torch.tensor(tc[i - W:i], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEV)
        preds.append(model(cin).item())
preds = np.array(preds); tv = tc[W:]; pv = preds[:len(tv)]
r2 = 1 - np.sum((tv - pv) ** 2) / np.sum((tv - tv.mean()) ** 2)
aes = []
for sp in [300, 400, 500]:
    sp_preds = preds[sp - W:]; pe = -1
    for j in range(len(sp_preds) - 1):
        if sp_preds[j] >= eol_n > sp_preds[j + 1]:
            pe = sp + j + (eol_n - sp_preds[j]) / (sp_preds[j + 1] - sp_preds[j] + 1e-8); break
        elif sp_preds[j] < eol_n: pe = sp + j; break
    aes.append(abs(te - pe) if pe >= 0 else -1)
print(f"true EOL={te}")
print(f"R2={r2:.4f} AE={aes[0]:.0f}/{aes[1]:.0f}/{aes[2]:.0f} "
      f"pred_std={pv.std():.4f} min={pv.min():.4f} max={pv.max():.4f}")
print("done")
