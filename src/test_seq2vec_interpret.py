"""CALCE: last-token → Linear → 32-dim, with weight interpretability analysis.

After training, extracts W matrix (32 x 128) from head[-1].weight and computes:
  row_norms[k]: ||W[k,:]|| — how much "signal" goes to step k (expect decay with k)
  row_sim[k]: cosine(W[k,:], W[k+1,:]) — temporal smoothness
  sv_ratios: singular values of W — intrinsic dimensionality of trajectory manifold
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
W, OUT, BATCH, EPOCHS, STRIDE = 64, 32, 64, 100, 1
TRAIN = ["CS2_36", "CS2_37", "CS2_38"]
TEST = "CS2_35"

src = open(Path(__file__).parent / "test_pooling_final.py", encoding="utf-8").read()
enc_code = src[src.index("class GDNEncoder"):src.index("class PoolHead")]
exec(enc_code)


class LastTokenSeq2Vec(nn.Module):
    """last-token → Linear → K values. Head[-1] is the (K, D) weight matrix."""

    def __init__(self, d_model=64, out_len=32, dropout=0.1):
        super().__init__()
        self.enc = GDNEncoder(input_dim=1, window_size=64)
        self.head = nn.Sequential(
            nn.RMSNorm(d_model), nn.Linear(d_model, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, out_len),
        )

    def forward(self, x):
        h = self.enc(x)
        return self.head(h[:, -1, :])  # (B, K)


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
model = LastTokenSeq2Vec(out_len=OUT).to(DEV)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
print(f"[last-token->32dim] {len(tr)} samples, {sum(p.numel() for p in model.parameters()):,} params", flush=True)
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

# --- trajectory eval ---
print("\n--- multi-step trajectory ---")
print(f"{'SP':>4} {'MAE':>8} {'RMSE':>8} {'R2':>8}  first6 pred vs targ")
with torch.no_grad():
    for sp in [300, 400, 500]:
        tgt = tc[sp:sp + OUT]
        cin = torch.tensor(tc[sp - W:sp], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEV)
        pred = model(cin).squeeze(0).cpu().numpy()
        mae, rmse, r2 = traj_metrics(pred, tgt)
        print(f"{sp:>4} {mae:>8.4f} {rmse:>8.4f} {r2:>8.3f}  "
              f"{np.round(pred[:6], 3)} vs {np.round(tgt[:6], 3)}")

# --- interpretability: W matrix analysis ---
W_final = model.head[-1].weight.data.cpu().numpy()  # (32, 128)

row_norms = np.linalg.norm(W_final, axis=1)
row_norms /= row_norms.max()

row_sims = []
for k in range(OUT - 1):
    cos = np.dot(W_final[k], W_final[k + 1]) / \
          (np.linalg.norm(W_final[k]) * np.linalg.norm(W_final[k + 1]) + 1e-12)
    row_sims.append(cos)

U, S, Vh = np.linalg.svd(W_final, full_matrices=False)
sv_ratios = S / S.sum()

print("\n--- interpretability: W(32 x 128) readout matrix ---")
print(f"row_norms (step 1-32):  {np.round(row_norms[:8], 3)}...{np.round(row_norms[-4:], 3)}")
print(f"  -> norm decay step1->step32: {row_norms[0]:.3f} -> {row_norms[-1]:.3f}")
print(f"row_sim (adjacent steps): mean={np.mean(row_sims):.3f} "
      f"first4={np.round(row_sims[:4], 3)} last3={np.round(row_sims[-3:], 3)}")
print(f"  -> temporal smoothness: {'smooth' if np.mean(row_sims) > 0.7 else 'diverse'}")
print(f"svd top5 ratios: {np.round(sv_ratios[:5], 4)} cumsum={np.cumsum(sv_ratios)[:5].round(3)}")
eff_dim = np.sum(np.cumsum(sv_ratios) < 0.95).item() + 1
print(f"  -> intrinsic dim (95%): {eff_dim}")
print("done")
