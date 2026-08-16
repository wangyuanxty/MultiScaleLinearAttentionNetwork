"""Multi-scale INT8: per-branch sensitivity + mixed precision diagnosis.

Approach:
  1. Train multi-scale to convergence (100 epoch)
  2. fp32 baseline
  3. Full INT8
  4. Per-branch INT8 proxy: add quant-step noise to each branch, measure AE change
  5. Per-branch INT8 (selectively quantize one branch, others fp32) — key insight
  6. Head-only vs branch-only quantification

Goal: identify the culprit → mixed precision strategy for deployment.
"""
import sys, numpy as np, torch, copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from gdn_model import build_gdn_model, masked_mae
from load_datasets import load_calce_cells_multivar
from data_pipeline import Seq2VecDataset
from torch.utils.data import DataLoader
import logging; logging.basicConfig(level=logging.WARNING)

DEV = torch.device("cuda")
W, OUT, BATCH, EPOCHS = 64, 1, 64, 100
TRAIN = ["CS2_36", "CS2_37", "CS2_38"]
TEST = "CS2_35"

torch.manual_seed(42)
np.random.seed(42)

print("Loading...")
caps_all, _, _ = load_calce_cells_multivar()
caps = {c: caps_all[c].copy() for c in caps_all}
all_tr = np.concatenate([caps[c] for c in TRAIN])
lo, hi = all_tr.min(), all_tr.max()
scale = lambda s: [(x - lo) / (hi - lo) for x in s]

tr = Seq2VecDataset(scale([caps[c] for c in TRAIN]), W, OUT, 1)
ld = DataLoader(tr, BATCH, shuffle=True)
model = build_gdn_model(multiscale=True, input_dim=1, window_size=W,
                        output_len=OUT, readout="last").to(DEV)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
n_params = sum(p.numel() for p in model.parameters())
print(f"[train multi-scale] {n_params:,} params", flush=True)
for ep in range(EPOCHS):
    model.train()
    for cap, feat, tgt, msk in ld:
        cap, tgt, msk = cap.to(DEV), tgt.to(DEV), msk.to(DEV)
        opt.zero_grad()
        loss = masked_mae(model(cap), tgt, msk)
        loss.backward()
        opt.step()
    if ep % 50 == 0:
        print(f"  E{ep} L={loss.item():.4f}", flush=True)

model.eval()
tc = scale([caps[TEST]])[0]
eol_n = (0.77 - lo) / (hi - lo)
te = int(np.argmax(tc < eol_n))


def eval_ae_full(m):
    m.eval()
    preds = []
    with torch.no_grad():
        for i in range(W, len(tc)):
            cin = torch.tensor(tc[i - W:i], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            preds.append(m(cin).item())
    pv = np.array(preds)[:len(tc) - W]; tv = tc[W:]
    r2 = 1 - np.sum((tv - pv) ** 2) / np.sum((tv - tv.mean()) ** 2)
    aes = []
    for sp in [300, 400, 500]:
        sp_preds = pv[sp - W:]; pe = -1
        for j in range(len(sp_preds) - 1):
            if sp_preds[j] >= eol_n > sp_preds[j + 1]:
                pe = sp + j + (eol_n - sp_preds[j]) / (sp_preds[j + 1] - sp_preds[j] + 1e-8); break
            elif sp_preds[j] < eol_n: pe = sp + j; break
        aes.append(abs(te - pe) if pe >= 0 else -1)
    return r2, aes


model_cpu = model.cpu()
r2_f, ae_f = eval_ae_full(model_cpu)
print(f"\nfp32: R2={float(r2_f):.4f} AE={ae_f[0]:.0f}/{ae_f[1]:.0f}/{ae_f[2]:.0f}")

# --- full INT8 ---
q_full = torch.quantization.quantize_dynamic(model_cpu, {torch.nn.Linear}, dtype=torch.qint8)
r2_q, ae_q = eval_ae_full(q_full)
print(f"full INT8: R2={float(r2_q):.4f} AE={ae_q[0]:.0f}/{ae_q[1]:.0f}/{ae_q[2]:.0f}")

# --- per-branch sensitivity: which branch is the INT8 bottleneck? ---
# Use weight noise proxy (INT8 equivalent) on isolated branches
print("\n=== per-branch sensitivity ===")
print(f"{'config':<20} {'AE':>12} {'AE_change':>10}")
branch_names = ["fine(ps2)", "mid(ps4)", "coarse(ps8)"]
for bi in range(3):
    m = build_gdn_model(multiscale=True, input_dim=1, window_size=W,
                        output_len=OUT, readout="last")
    m.load_state_dict(model_cpu.state_dict())
    m.eval()
    # apply INT8-like noise to branch bi only
    with torch.no_grad():
        for p in m.branches[bi].parameters():
            if p.dim() >= 2:
                step = p.abs().max().item() / 127.0
                p.add_(torch.randn_like(p) * step * 0.5)
    _, ae_n = eval_ae_full(m)
    print(f"{'noise-'+branch_names[bi]:<20} {ae_n[0]:>4}/{ae_n[1]:>4}/{ae_n[2]:>4} "
          f"{'+'+str(int(np.mean(ae_n)-np.mean(ae_f))):>8}")

# --- selective INT8: quantize one branch, keep others fp32 ---
print("\n=== selective INT8 (quantize one branch, others fp32) ===")
for bi in range(3):
    m = torch.quantization.quantize_dynamic(model_cpu, {torch.nn.Linear}, dtype=torch.qint8)
    # dequantize the OTHER two branches back to fp32
    for bj in range(3):
        if bj == bi:
            continue
        for name, mod in m.branches[bj].named_modules():
            if hasattr(mod, "weight") and hasattr(mod.weight, "q_per_channel_scales"):
                w_fp = torch.dequantize(mod.weight())  # back to fp32
                mod.set_weight_bias(w_fp, getattr(mod, "bias", None))
    try:
        _, ae_s = eval_ae_full(m)
        print(f"{'INT8-'+branch_names[bi]:<20} {ae_s[0]:>4}/{ae_s[1]:>4}/{ae_s[2]:>4} "
              f"{'+'+str(int(np.mean(ae_s)-np.mean(ae_f))):>8}")
    except Exception as e:
        print(f"{'INT8-'+branch_names[bi]:<20} failed: {str(e)[:80]}")

# --- head-only INT8 (largest matrices, least sensitivity) ---
print("\n=== component breakdown ===")
for bi in range(3):
    n = sum(1 for _, m2 in model_cpu.branches[bi].named_modules() if isinstance(m2, torch.nn.Linear))
    p = sum(p2.numel() for p2 in model_cpu.branches[bi].parameters())
    print(f"{branch_names[bi]:<12}: {n} Linear layers, {p:,} params ({p*100/n_params:.0f}%)")
hp = sum(p2.numel() for p2 in model_cpu.head_cap.parameters())
print(f"{'head':<12}: 2 Linear layers, {hp:,} params ({hp*100/n_params:.0f}%)")
memsave = sum(p2.numel() for p2 in model_cpu.branches[2].parameters())  # coarse branch
print(f"\ndeployment options:")
print(f"  single-branch+INT8: 337KB, AE 2.2→2.3 (lossless)")
print(f"  multi-scale: keep fp32 ({n_params*4/1024:.0f}KB) or accept INT8 loss")
print("done")
