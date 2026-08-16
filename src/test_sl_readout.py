"""CALCE: S_L direct readout — 32 query vectors read the final GDN state.

Structure:
  h = proj(x) + pos_embed                      # window 64 tokens
  h = 2x GDN layers (scan stops at window end)
  S_L = final GDN state (B, H, Dk, Dv)
  q_k = 32 learned query vectors (one per future step)
  o_k = S_L^T @ q_k                            # direct read, no extra scan
  pred = Linear(o_k)

Evaluated with trajectory metrics (MAE/RMSE/R2 per SP), same as seq2vec-vs-AR.
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
W, OUT, BATCH, EPOCHS, STRIDE = 64, 32, 64, 60, 1
TRAIN = ["CS2_36", "CS2_37", "CS2_38"]
TEST = "CS2_35"


class SLReadoutSeq2Vec(nn.Module):
    """32 query vectors read the final GDN state directly (no extra scan)."""

    def __init__(self, d_model=64, num_layers=2, out_len=32, num_heads=4,
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
        # one query vector per future step: (K, H, Dk)
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
        # S: (B, H, Dk, Dv) final state of last layer
        o = torch.einsum('bhds,khd->kbhs', S, self.queries)  # (K, B, H, Dv)
        o = o.transpose(0, 1).reshape(B, OUT, -1)            # (B, K, H*Dv)
        return self.head(o).squeeze(-1)                      # (B, K)


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
print(f"[S_L readout] {len(tr)} samples, {sum(p.numel() for p in model.parameters()):,} params", flush=True)
for ep in range(EPOCHS):
    model.train()
    for cap, feat, tgt, msk in ld:
        cap, tgt, msk = cap.to(DEV), tgt.to(DEV), msk.to(DEV)
        opt.zero_grad()
        loss = masked_mae(model(cap), tgt, msk)
        loss.backward()
        opt.step()
    if ep % 15 == 0:
        print(f"  E{ep} L={loss.item():.4f}", flush=True)

model.eval()
tc = scale([caps[TEST]])[0]
print("\n--- 32-step trajectory (S_L direct readout) ---")
print(f"{'SP':>4} {'MAE':>8} {'RMSE':>8} {'R2':>8}  pred first6/last6 vs targ")
with torch.no_grad():
    for sp in [300, 400, 500]:
        tgt = tc[sp:sp + OUT]
        cin = torch.tensor(tc[sp - W:sp], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEV)
        pred = model(cin).squeeze(0).cpu().numpy()
        mae, rmse, r2 = traj_metrics(pred, tgt)
        print(f"{sp:>4} {mae:>8.4f} {rmse:>8.4f} {r2:>8.3f}  "
              f"{np.round(pred[:6], 3)}...{np.round(pred[-6:], 3)} vs {np.round(tgt[:6], 3)}")
print("done")
