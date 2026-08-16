"""CALCE: mean-pool vs last-token head — final decision, 100 epochs each.

Same encoder (2× GDN2Block + pos_embed), same head, standard CALCE eval.
Settles the readout question with full training, not 15-epoch snapshots.
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
W, OUT, BATCH, EPOCHS = 64, 1, 64, 100
TRAIN = ["CS2_36", "CS2_37", "CS2_38"]
TEST = "CS2_35"


class GDNEncoder(nn.Module):
    def __init__(self, d_model=64, num_layers=2, num_heads=4, head_dim=16,
                 expand_v=2.0, conv_size=4, dropout=0.1, input_dim=1, window_size=64):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, window_size, d_model) * 0.02)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'gdn': GDN2Block(d_model, head_dim, num_heads, expand_v, conv_size, dropout),
                'norm': nn.RMSNorm(d_model),
            }) for _ in range(num_layers)
        ])

    def forward(self, x):
        B, L, _ = x.shape
        h = self.input_proj(x) + self.pos_embed[:, :L, :]
        for l in self.layers:
            h = l['norm'](l['gdn'](h) + h)
        return h


class PoolHead(nn.Module):
    """head on (B, D) vector — pooling strategy chosen by mode."""

    def __init__(self, enc, mode, d_model=64, dropout=0.1):
        super().__init__()
        self.enc = enc
        self.mode = mode
        hdim = 2 * d_model if mode == "concat" else d_model
        self.head = nn.Sequential(
            nn.RMSNorm(hdim), nn.Linear(hdim, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        h = self.enc(x)
        if self.mode == "mean":
            v = h.mean(dim=1)
        elif self.mode == "last":
            v = h[:, -1, :]
        elif self.mode == "concat":
            v = torch.cat([h.mean(dim=1), h[:, -1, :]], dim=-1)
        return self.head(v)


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


tr = Seq2VecDataset(scale([caps[c] for c in TRAIN]), W, OUT, 1)
ld = DataLoader(tr, BATCH, shuffle=True)


def run(mode):
    model = PoolHead(GDNEncoder(input_dim=1, window_size=W), mode).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(f"[{mode}] {sum(p.numel() for p in model.parameters()):,} params", flush=True)
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
    print(f"[{mode}] R2={r2:.4f} AE={aes[0]:.0f}/{aes[1]:.0f}/{aes[2]:.0f} "
          f"pred_std={pv.std():.4f}", flush=True)


for mode in ["concat"]:
    run(mode)
print("done")
