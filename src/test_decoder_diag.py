"""Gradient diagnostic: where is learning breaking?

Configs (15 epochs each, loss + grad-norm per epoch):
  C. GDNBatteryModel single-branch patch=2  — known to converge (AE~3 control)
  B. GDNEncoder + mean-pool head            — is my encoder learnable?
  A. GDNEncoder + mean-ctx decoder          — is the decoder chain the problem?
"""
import sys, numpy as np, torch, torch.nn as nn
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from gdn_v2 import GDN2Block
from gdn_model import build_gdn_model
from load_datasets import load_calce_cells_multivar
from data_pipeline import Seq2VecDataset
from torch.utils.data import DataLoader
import logging; logging.basicConfig(level=logging.WARNING)

DEV = torch.device("cuda")
W, OUT, BATCH, EPOCHS = 64, 1, 64, 15
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


class MeanCtxDecoder(nn.Module):
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
        h = self.enc(x)
        ctx = h.mean(dim=1, keepdim=True)
        q = self.query.expand(h.size(0), -1, -1)
        xd = torch.cat([ctx, q], dim=1)
        xd = self.dec_norm(self.dec_gdn(xd) + xd)
        return self.head(xd[:, -1, :])


class MeanPoolBaseline(nn.Module):
    def __init__(self, enc, d_model=64, dropout=0.1):
        super().__init__()
        self.enc = enc
        self.head = nn.Sequential(
            nn.RMSNorm(d_model), nn.Linear(d_model, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.head(self.enc(x).mean(dim=1))


def masked_mae(pred, target, mask=None):
    err = (pred - target).abs()
    if mask is not None:
        err = err * mask
        return err.sum() / (mask.sum() + 1e-8)
    return err.mean()


def grad_norms(model):
    tot = enc = other = 0.0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        n = p.grad.norm().item()
        tot += n * n
        if name.startswith('enc'):
            enc += n * n
        else:
            other += n * n
    return np.sqrt(tot), np.sqrt(enc), np.sqrt(other)


print("Loading...")
caps_all, _, _ = load_calce_cells_multivar()
caps = {c: caps_all[c].copy() for c in caps_all}
all_tr = np.concatenate([caps[c] for c in TRAIN])
lo, hi = all_tr.min(), all_tr.max()


def scale(seqs):
    return [(s - lo) / (hi - lo) for s in seqs]


tr = Seq2VecDataset(scale([caps[c] for c in TRAIN]), W, OUT, 1)
ld = DataLoader(tr, BATCH, shuffle=True)


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
            g_tot, g_enc, g_oth = grad_norms(model)
            opt.step()
        print(f"  E{ep} L={loss.item():.4f} grad_tot={g_tot:.3f} grad_enc={g_enc:.3f} "
              f"grad_oth={g_oth:.3f}", flush=True)
    model.eval()
    tc = scale([caps[TEST]])[0]
    preds = []
    with torch.no_grad():
        for i in range(W, len(tc)):
            cin = torch.tensor(tc[i - W:i], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEV)
            preds.append(model(cin).item())
    pv = np.array(preds)[:len(tc) - W]
    print(f"  [{label}] pred_std={pv.std():.4f} min={pv.min():.4f} max={pv.max():.4f}", flush=True)


run("C: GDNBatteryModel control", lambda: build_gdn_model(patch_size=2, input_dim=1,
                                                          window_size=W, output_len=OUT))
run("B: enc+meanpool", lambda: MeanPoolBaseline(GDNEncoder(input_dim=1, window_size=W)))
run("A: enc+meanctx-decoder", lambda: MeanCtxDecoder(GDNEncoder(input_dim=1, window_size=W)))
print("done")
