"""CALCE: seq2vec out=32, two readout designs (60 epochs each).

A. LastTokenSeq2Vec — last-token readout + head → 32 values (proven readout, never tested at K=32)
B. UnifiedGDNSeq2Vec — one GDN backbone scans [window 64 + query 32] continuously,
   loss on query positions only (user's design: GDN is a backbone, not encoder-decoder)

Diagnostics per SP: prediction vs target curves (is output constant? where does 641 come from?).
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

# encoder from pooling test
src = open(Path(__file__).parent / "test_pooling_final.py", encoding="utf-8").read()
enc_code = src[src.index("class GDNEncoder"):src.index("class PoolHead")]
exec(enc_code)


class LastTokenSeq2Vec(nn.Module):
    def __init__(self, d_model=64, out_len=32, dropout=0.1):
        super().__init__()
        self.enc = GDNEncoder(input_dim=1, window_size=64)
        self.head = nn.Sequential(
            nn.RMSNorm(d_model), nn.Linear(d_model, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, out_len),
        )

    def forward(self, x):
        h = self.enc(x)
        return self.head(h[:, -1, :])


class UnifiedGDNSeq2Vec(nn.Module):
    """One GDN backbone scans [window 64 + query 32] continuously; loss on query positions only."""

    def __init__(self, d_model=64, num_layers=2, out_len=32, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)
        self.queries = nn.Parameter(torch.randn(1, out_len, d_model) * 0.02)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'gdn': GDN2Block(d_model, 16, 4, 2.0, 4, dropout),
                'norm': nn.RMSNorm(d_model),
            }) for _ in range(num_layers)
        ])
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        B = x.size(0)
        h = self.input_proj(x) + self.pos_embed
        q = self.queries.expand(B, -1, -1)
        hq = torch.cat([h, q], dim=1)
        for l in self.layers:
            hq = l['norm'](l['gdn'](hq) + hq)
        return self.head(hq[:, 64:, :]).squeeze(-1)


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


def run(label, model):
    model = model.to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(f"[{label}] {len(tr)} samples, {sum(p.numel() for p in model.parameters()):,} params", flush=True)
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
    eol_n = (0.77 - lo) / (hi - lo)
    te = int(np.argmax(tc < eol_n))
    print(f"  true EOL={te} eol_n={eol_n:.4f}", flush=True)
    with torch.no_grad():
        for sp in [300, 400, 500]:
            cin = torch.tensor(tc[sp - W:sp], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEV)
            pred = model(cin).squeeze(0).cpu().numpy()
            tgt = tc[sp:sp + OUT]
            pe = -1
            for j in range(len(pred) - 1):
                if pred[j] >= eol_n > pred[j + 1]:
                    pe = sp + j + (eol_n - pred[j]) / (pred[j + 1] - pred[j] + 1e-8); break
                elif pred[j] < eol_n:
                    pe = sp + j; break
            print(f"  SP={sp}: pred first6={np.round(pred[:6], 3)} last6={np.round(pred[-6:], 3)}", flush=True)
            print(f"          targ first6={np.round(tgt[:6], 3)} last6={np.round(tgt[-6:], 3)}")
            print(f"          pred_std={pred.std():.4f} crossing={pe:.1f} AE={abs(te - pe):.0f}", flush=True)


run("A: last-token head out=32", LastTokenSeq2Vec(out_len=OUT))
run("B: unified GDN [win+queries]", UnifiedGDNSeq2Vec(out_len=OUT))
print("done")
