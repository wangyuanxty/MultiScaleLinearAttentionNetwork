"""Diagnose INT8 quantization degradation: is AE 6->16 a real limitation or a bug?

1. Compare fp32 vs INT8 outputs on the same windows (per-sample error)
2. Per-layer quantization error (weight range vs quant step)
3. Check if some specific layer dominates the loss
"""
import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from gdn_model import build_gdn_model, masked_mae
from load_datasets import load_calce_cells_multivar
from data_pipeline import Seq2VecDataset
from torch.utils.data import DataLoader
import logging; logging.basicConfig(level=logging.WARNING)

DEV = torch.device("cuda")
W, OUT, BATCH, EPOCHS = 64, 1, 64, 30
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
model = build_gdn_model(multiscale=False, input_dim=1, window_size=W,
                        output_len=OUT, readout="last").to(DEV)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
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
model_cpu = model.cpu()
qmodel = torch.quantization.quantize_dynamic(model_cpu, {torch.nn.Linear}, dtype=torch.qint8)

print("\n=== fp32 vs INT8 output comparison (10 windows) ===")
diffs = []
with torch.no_grad():
    for i in range(W, W + 10):
        cin = torch.tensor(tc[i - W:i], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        p_f = model_cpu(cin).item()
        p_q = qmodel(cin).item()
        diffs.append(abs(p_f - p_q))
        print(f"win@{i}: fp32={p_f:.6f} int8={p_q:.6f} diff={abs(p_f-p_q):.6f}")
print(f"mean output diff: {np.mean(diffs):.6f} (capacity scale ~0-1)")

print("\n=== per-Linear quantization error ===")
print(f"{'layer':<40} {'weight_range':>12} {'abs_err':>12} {'rel_err':>10}")
total = 0
for name, mod in qmodel.named_modules():
    if hasattr(mod, "weight") and hasattr(mod.weight, "q_per_channel_scales"):
        wq = torch.dequantize(mod.weight())
        wf = None
        for n2, m2 in model_cpu.named_modules():
            if n2 == name:
                wf = m2.weight.detach()
        if wf is None:
            continue
        err = (wq - wf).abs().mean().item()
        rng = wf.abs().max().item() - wf.abs().min().item()
        print(f"{name:<40} {rng:>12.5f} {err:>12.2e} {err/max(rng,1e-9):>10.4f}")
        total += err
print(f"total mean abs weight error: {total:.4f}")
print("done")
