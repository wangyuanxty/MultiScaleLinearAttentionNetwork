"""CALCE: encoder-decoder with output_len=1 — round 2.

Round 1 verdict: model collapsed to a constant (pred std=0.0001, loss flat).
Cause hypothesis: last-token ctx o_L is a projection for next-token prediction,
not a sequence summary → info/gradient too weak → shared query degenerates.

Round 2: same encoder, two configs:
  A. mean-ctx decoder (mean-pool ctx + 1 query + 1 GDN decoder layer)
  B. mean-pool baseline (same encoder + mean-pool head — should recover AE~3)
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
W, OUT, BATCH, EPOCHS = 64, 1, 64, 60
TRAIN = ["CS2_36", "CS2_37", "CS2_38"]
TEST = "CS2_35"


class GDNEncoder(nn.Module):
    """Shared encoder: input_proj + pos_embed + 2× GDN2Block."""

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


class MeanCtxDecoder(nn.Module):
    """Encoder + mean-pool context + 1 query token + GDN decoder."""

    def __init__(self, enc, d_model=64, num_heads=4, head_dim=16, expand_v=2.0,
                 conv_size=4, dropout=0.1):
        super().__init__()
        self.enc = enc
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.dec_gdn = GDN2Block(d_model, head_dim, num_heads, expand_v, conv_size, dropout)
        self.dec_norm = nn.RMSNorm(d_model)
        self.head = nn.Sequential(
            nn.RMSNorm(d_model), nn.Linear(d_model, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        h = self.enc(x)                            # (B, W, D)
        ctx = h.mean(dim=1, keepdim=True)          # (B, 1, D) mean context
        q = self.query.expand(h.size(0), -1, -1)   # (B, 1, D)
        xd = torch.cat([ctx, q], dim=1)            # (B, 2, D)
        xd = self.dec_norm(self.dec_gdn(xd) + xd)
        return self.head(xd[:, -1, :])             # (B, 1)


class MeanPoolBaseline(nn.Module):
    """Same encoder + mean-pool head (control: encoder must be learnable)."""

    def __init__(self, enc, d_model=64, dropout=0.1):
        super().__init__()
        self.enc = enc
        self.head = nn.Sequential(
            nn.RMSNorm(d_model), nn.Linear(d_model, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        h = self.enc(x)
        return self.head(h.mean(dim=1))            # (B, 1)


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


def eval_model(model, label):
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
    print(f"[{label}] R2={r2:.4f} AE={aes[0]:.0f}/{aes[1]:.0f}/{aes[2]:.0f} "
          f"pred_std={pv.std():.4f} pred_min={pv.min():.4f} pred_max={pv.max():.4f}", flush=True)


def run(label, build):
    model = build().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(f"[{label}] {sum(p.numel() for p in model.parameters()):,} params", flush=True)
    for ep in range(EPOCHS):
        model.train()
        for cap, feat, tgt, msk in ld:
            cap, tgt, msk = cap.to(DEV), tgt.to(DEV), msk.to(DEV)
            opt.zero_grad()
            loss = masked_mae(model(cap), tgt, msk)
            loss.backward()
            opt.step()
        if ep % 10 == 0:
            print(f"  E{ep} L={loss.item():.4f}", flush=True)
    eval_model(model, label)


enc = GDNEncoder(input_dim=1, window_size=W)
run("mean-ctx decoder", lambda: MeanCtxDecoder(GDNEncoder(input_dim=1, window_size=W)))
run("mean-pool baseline", lambda: MeanPoolBaseline(GDNEncoder(input_dim=1, window_size=W)))
print("done")
