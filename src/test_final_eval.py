"""Final evaluation protocol (unified). CALCE + NASA.

Table 2 (non-recursive, comparable to PatchFormer reports):
  Single-step model: R2, trajectory MAE, trajectory RMSE (per SP, from SP-64)
  Uses real window at each step (same as their predict()).

Table 1 (deployment-consistent, only pre-SP info):
  Seq2vec K=32 model: chunked rollout to EOL, report RUL MAE, RUL RMSE.
  RUL = |true_EOL - predicted_EOL| from pre-SP info only.
"""
import sys, numpy as np, torch, torch.nn as nn
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from gdn_v2 import GDN2Block
from gdn_model import masked_mae
from load_datasets import load_calce_cells_multivar, load_nasa_capacity
from data_pipeline import Seq2VecDataset
from torch.utils.data import DataLoader
import logging; logging.basicConfig(level=logging.WARNING)

DEV = torch.device("cuda")
W_CAL, W_NASA = 64, 30
SEED = 42

src = open(Path(__file__).parent / "test_pooling_final.py", encoding="utf-8").read()
enc_code = src[src.index("class GDNEncoder"):src.index("class PoolHead")]
exec(enc_code)


class SingleStepModel(nn.Module):
    def __init__(self, input_dim=1, window_size=64, dropout=0.1):
        super().__init__()
        self.enc = GDNEncoder(input_dim=input_dim, window_size=window_size)
        self.head = nn.Sequential(
            nn.RMSNorm(64), nn.Linear(64, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.head(self.enc(x)[:, -1, :])


class Seq2VecModel(nn.Module):
    def __init__(self, out_len=32, input_dim=1, window_size=64, dropout=0.1):
        super().__init__()
        self.enc = GDNEncoder(input_dim=input_dim, window_size=window_size)
        self.head = nn.Sequential(
            nn.RMSNorm(64), nn.Linear(64, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, out_len),
        )

    def forward(self, x):
        return self.head(self.enc(x)[:, -1, :])


def eval_table2(model, tc, lo, hi, W, sps, eol_ratio):
    """Table 2: non-recursive trajectory metrics (comparable to PatchFormer reports)."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(W, len(tc)):
            cin = torch.tensor(tc[i - W:i], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEV)
            preds.append(model(cin).item())
    pv = np.array(preds)[:len(tc) - W]; tv = tc[W:]
    r2 = 1 - np.sum((tv - pv) ** 2) / np.sum((tv - tv.mean()) ** 2)
    results = {"R2": round(float(r2), 4)}
    for sp in sps:
        seg = pv[sp - W:]; seg_t = tc[sp:]
        n = min(len(seg), len(seg_t))
        mae = np.mean(np.abs(seg[:n] - seg_t[:n]))
        rmse = np.sqrt(np.mean((seg[:n] - seg_t[:n]) ** 2))
        results[f"traj_MAE_sp{sp}"] = round(float(mae), 4)
        results[f"traj_RMSE_sp{sp}"] = round(float(rmse), 4)
    return results


def eval_table1(model, tc, lo, hi, W, K, sps, eol_ratio):
    """Table 1: deployment-consistent RUL (only pre-SP info). Seq2vec chunked rollout."""
    eol_n = (eol_ratio - lo) / (hi - lo)
    te = int(np.argmax(tc < eol_n))
    model.eval()
    results = {}
    for sp in sps:
        window = tc[sp - W:sp].copy()
        traj = []
        with torch.no_grad():
            while len(traj) < 800:
                cin = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEV)
                pred = model(cin).squeeze(0).cpu().numpy()
                traj.extend(pred.tolist())
                window = np.concatenate([window[K:], pred])
                if pred[-1] < eol_n:
                    break
        traj = np.array(traj)
        pe = -1
        for j in range(len(traj) - 1):
            if traj[j] >= eol_n > traj[j + 1]:
                pe = sp + j + (eol_n - traj[j]) / (traj[j + 1] - traj[j] + 1e-8)
                break
            elif traj[j] < eol_n:
                pe = sp + j
                break
        if pe < 0:
            continue
        abs_err = abs(te - pe)
        results[f"RUL_sp{sp}"] = {"rul_mae": round(float(abs_err), 1)}
    return results


# ─── CALCE (Table 2) ──────────────────────────────────────────────
print("=== CALCE Table 2: non-recursive trajectory ===")
caps_all, _, _ = load_calce_cells_multivar()
caps_c = {c: caps_all[c].copy() for c in caps_all}
TRAIN_C = ["CS2_36", "CS2_37", "CS2_38"]
TEST_C = "CS2_35"
all_tr_c = np.concatenate([caps_c[c] for c in TRAIN_C])
lo_c, hi_c = all_tr_c.min(), all_tr_c.max()
scale_c = lambda s: [(x - lo_c) / (hi_c - lo_c) for x in s]

torch.manual_seed(SEED); np.random.seed(SEED)
tr = Seq2VecDataset(scale_c([caps_c[c] for c in TRAIN_C]), W_CAL, 1, 1)
ld = DataLoader(tr, 64, shuffle=True)
m1 = SingleStepModel(window_size=W_CAL).to(DEV)
opt = torch.optim.Adam(m1.parameters(), lr=1e-3)
print(f"[single-step CALCE] {sum(p.numel() for p in m1.parameters()):,} params", flush=True)
for ep in range(100):
    m1.train()
    for cap, feat, tgt, msk in ld:
        cap, tgt, msk = cap.to(DEV), tgt.to(DEV), msk.to(DEV)
        opt.zero_grad()
        loss = masked_mae(m1(cap), tgt, msk)
        loss.backward(); opt.step()
    if ep % 50 == 0:
        print(f"  E{ep} L={loss.item():.4f}", flush=True)
res_c = eval_table2(m1, scale_c([caps_c[TEST_C]])[0], lo_c, hi_c, W_CAL, [300, 400, 500], 0.77)
print(f"CALCE Table2: R2={res_c['R2']}")
for sp in [300, 400, 500]:
    print(f"  SP={sp}: traj_MAE={res_c[f'traj_MAE_sp{sp}']:.4f} "
          f"traj_RMSE={res_c[f'traj_RMSE_sp{sp}']:.4f}")

# ─── CALCE (Table 1) ──────────────────────────────────────────────
print("\n=== CALCE Table 1: deployment-consistent RUL (seq2vec) ===")
K = 32
torch.manual_seed(SEED); np.random.seed(SEED)
tr2 = Seq2VecDataset(scale_c([caps_c[c] for c in TRAIN_C]), W_CAL, K, 1)
ld2 = DataLoader(tr2, 64, shuffle=True)
m2 = Seq2VecModel(out_len=K, window_size=W_CAL).to(DEV)
opt2 = torch.optim.Adam(m2.parameters(), lr=1e-3)
print(f"[seq2vec K={K}] {sum(p.numel() for p in m2.parameters()):,} params", flush=True)
for ep in range(100):
    m2.train()
    for cap, feat, tgt, msk in ld2:
        cap, tgt, msk = cap.to(DEV), tgt.to(DEV), msk.to(DEV)
        opt2.zero_grad()
        loss = masked_mae(m2(cap), tgt, msk)
        loss.backward(); opt2.step()
    if ep % 50 == 0:
        print(f"  E{ep} L={loss.item():.4f}", flush=True)
res_r = eval_table1(m2, scale_c([caps_c[TEST_C]])[0], lo_c, hi_c, W_CAL, K, [300, 400, 500], 0.77)
for sp in [300, 400, 500]:
    r = res_r.get(f"RUL_sp{sp}", {})
    print(f"  SP={sp}: RUL_MAE={r.get('rul_mae', '?')} cycles")

# ─── NASA (Table 2) ───────────────────────────────────────────────
print("\n=== NASA Table 2: non-recursive trajectory ===")
TRAIN_N = ["B0006", "B0007", "B0018"]
TEST_N = "B0005"
caps_n = {b: load_nasa_capacity(b) for b in TRAIN_N + [TEST_N]}
all_tr_n = np.concatenate([caps_n[c] for c in TRAIN_N])
lo_n, hi_n = all_tr_n.min(), all_tr_n.max()
scale_n = lambda s: [(x - lo_n) / (hi_n - lo_n) for x in s]

torch.manual_seed(SEED); np.random.seed(SEED)
tr_n = Seq2VecDataset(scale_n([caps_n[c] for c in TRAIN_N]), W_NASA, 1, 1)
ld_n = DataLoader(tr_n, 64, shuffle=True)
m1n = SingleStepModel(window_size=W_NASA).to(DEV)
opt_n = torch.optim.Adam(m1n.parameters(), lr=1e-3)
print(f"[single-step NASA] {sum(p.numel() for p in m1n.parameters()):,} params", flush=True)
for ep in range(100):
    m1n.train()
    for cap, feat, tgt, msk in ld_n:
        cap, tgt, msk = cap.to(DEV), tgt.to(DEV), msk.to(DEV)
        opt_n.zero_grad()
        loss = masked_mae(m1n(cap), tgt, msk)
        loss.backward(); opt_n.step()
    if ep % 50 == 0:
        print(f"  E{ep} L={loss.item():.4f}", flush=True)
res_n = eval_table2(m1n, scale_n([caps_n[TEST_N]])[0], lo_n, hi_n, W_NASA, [50, 70, 90], 1.4)
print(f"NASA Table2: R2={res_n['R2']}")
for sp in [50, 70, 90]:
    print(f"  SP={sp}: traj_MAE={res_n[f'traj_MAE_sp{sp}']:.4f} "
          f"traj_RMSE={res_n[f'traj_RMSE_sp{sp}']:.4f}")

print("\n=== FINAL SUMMARY ===")
print(f"CALCE: R2={res_c['R2']} | traj_MAE(300/400/500)={res_c['traj_MAE_sp300']}/{res_c['traj_MAE_sp400']}/{res_c['traj_MAE_sp500']}")
print(f"       RUL_MAE(300/400/500)={res_r.get('RUL_sp300',{}).get('rul_mae','?')}/{res_r.get('RUL_sp400',{}).get('rul_mae','?')}/{res_r.get('RUL_sp500',{}).get('rul_mae','?')}")
print(f"NASA:  R2={res_n['R2']} | traj_MAE(50/70/90)={res_n['traj_MAE_sp50']}/{res_n['traj_MAE_sp70']}/{res_n['traj_MAE_sp90']}")
print("done")
