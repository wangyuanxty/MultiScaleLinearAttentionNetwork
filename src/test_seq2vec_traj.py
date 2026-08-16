"""CALCE: seq2vec trajectory evaluation — non-autoregressive vs autoregressive.

Evaluates 32-step trajectory quality (MAE/RMSE/R2 per SP), NOT EOL crossing.
  A. seq2vec: last-token readout → 32 values in one forward pass (non-autoregressive)
  R. autoregressive: single-step last-token model rolled 32 steps (error accumulation)

This is the paper-level comparison: one-shot vs rolling.
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

src = open(Path(__file__).parent / "test_pooling_final.py", encoding="utf-8").read()
enc_code = src[src.index("class GDNEncoder"):src.index("class PoolHead")]
exec(enc_code)


class LastTokenSeq2Vec(nn.Module):
    """Non-autoregressive: last-token readout → K values in one pass."""

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


class LastToken1Step(nn.Module):
    """Single-step: same encoder, last-token readout → 1 value (roll for AR)."""

    def __init__(self, d_model=64, dropout=0.1):
        super().__init__()
        self.enc = GDNEncoder(input_dim=1, window_size=64)
        self.head = nn.Sequential(
            nn.RMSNorm(d_model), nn.Linear(d_model, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        h = self.enc(x)
        return self.head(h[:, -1, :])


def masked_mae(pred, target, mask=None):
    err = (pred - target).abs()
    if mask is not None:
        err = err * mask
        return err.sum() / (mask.sum() + 1e-8)
    return err.mean()


def traj_metrics(pred, tgt):
    """Trajectory metrics for one predicted sequence."""
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


def train(model, label):
    model = model.to(DEV)
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
        if ep % 20 == 0:
            print(f"  E{ep} L={loss.item():.4f}", flush=True)
    return model


seq2vec_model = train(LastTokenSeq2Vec(out_len=OUT), "seq2vec non-AR")
onestep_model = train(LastToken1Step(), "single-step AR baseline")

seq2vec_model.eval()
onestep_model.eval()
tc = scale([caps[TEST]])[0]

print("\n--- 32-step trajectory comparison (SP=300/400/500) ---")
print(f"{'SP':>4} {'method':<12} {'MAE':>8} {'RMSE':>8} {'R2':>8}  first6 pred vs targ")
all_rows = []
with torch.no_grad():
    for sp in [300, 400, 500]:
        tgt = tc[sp:sp + OUT]
        # non-AR: one forward pass
        cin = torch.tensor(tc[sp - W:sp], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEV)
        pred_s2v = seq2vec_model(cin).squeeze(0).cpu().numpy()
        mae_s, rmse_s, r2_s = traj_metrics(pred_s2v, tgt)
        # AR: roll single-step model, feeding predicted values back
        window = tc[sp - W:sp].copy()
        preds_ar = []
        for _ in range(OUT):
            cin = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEV)
            p = onestep_model(cin).item()
            preds_ar.append(p)
            window = np.concatenate([window[1:], [p]])
        pred_ar = np.array(preds_ar)
        mae_r, rmse_r, r2_r = traj_metrics(pred_ar, tgt)
        print(f"{sp:>4} {'seq2vec':<12} {mae_s:>8.4f} {rmse_s:>8.4f} {r2_s:>8.3f}  "
              f"{np.round(pred_s2v[:6], 3)} vs {np.round(tgt[:6], 3)}")
        print(f"{sp:>4} {'AR-rolled':<12} {mae_r:>8.4f} {rmse_r:>8.4f} {r2_r:>8.3f}  "
              f"{np.round(pred_ar[:6], 3)} vs {np.round(tgt[:6], 3)}")
        print(f"      targ    {'':12} {'':8} {'':8} {'':8}  {np.round(tgt[:6], 3)}")
        all_rows.append((sp, mae_s, rmse_s, r2_s, mae_r, rmse_r, r2_r))

print("\n--- summary ---")
print(f"{'SP':>4} {'seq2vec MAE':>12} {'AR MAE':>10} {'seq2vec R2':>11} {'AR R2':>8}")
for sp, mae_s, _, r2_s, mae_r, _, r2_r in all_rows:
    print(f"{sp:>4} {mae_s:>12.4f} {mae_r:>10.4f} {r2_s:>11.3f} {r2_r:>8.3f}")
print("done")
