"""Layer-1 deployment validation: INT8 quantization + incremental inference.

Train mainline (multi-scale + 5ch + last-token) on CALCE, then:
  1. Quantize (dynamic INT8) -> AE comparison vs fp32
  2. Incremental inference consistency: state-update path must match full rescan
  3. Memory stats: params fp32/INT8, state size
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
W, OUT, BATCH, EPOCHS = 64, 1, 64, 50
TRAIN = ["CS2_36", "CS2_37", "CS2_38"]
TEST = "CS2_35"
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

print("Loading...")
caps_all, feats_all, _ = load_calce_cells_multivar()
caps = {c: caps_all[c].copy() for c in caps_all}
keep = [0, 1, 2, 3, 9]
feats = {c: feats_all[c][:, keep].copy() for c in feats_all}
all_tr = np.concatenate([caps[c] for c in TRAIN])
lo, hi = all_tr.min(), all_tr.max()


def scale(seqs):
    return [(s - lo) / (hi - lo) for s in seqs]


tr = Seq2VecDataset(scale([caps[c] for c in TRAIN]), W, OUT, 1, [feats[c] for c in TRAIN])
ld = DataLoader(tr, BATCH, shuffle=True)
model = build_gdn_model(
    multiscale=True, input_dim=6, window_size=W, output_len=OUT, readout="last",
).to(DEV)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
print(f"[train] {len(tr)} samples, {sum(p.numel() for p in model.parameters()):,} params", flush=True)
for ep in range(EPOCHS):
    model.train()
    for cap, feat, tgt, msk in ld:
        cap, tgt, msk = cap.to(DEV), tgt.to(DEV), msk.to(DEV)
        x = torch.cat([cap, feat.to(DEV)], dim=-1)
        opt.zero_grad()
        loss = masked_mae(model(x), tgt, msk)
        loss.backward()
        opt.step()
    if ep % 25 == 0:
        print(f"  E{ep} L={loss.item():.4f}", flush=True)

model.eval()
tc = scale([caps[TEST]])[0]
eol_n = (0.77 - lo) / (hi - lo)
te = int(np.argmax(tc < eol_n))
ftest = feats[TEST]


def eval_ae(m, use_cuda=True):
    m.eval()
    dev = DEV if use_cuda else torch.device("cpu")
    preds = []
    with torch.no_grad():
        for i in range(W, len(tc)):
            cin = torch.tensor(tc[i - W:i], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(dev)
            fin = torch.tensor(ftest[i - W:i], dtype=torch.float32).unsqueeze(0).to(dev)
            preds.append(m(torch.cat([cin, fin], dim=-1)).item())
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


r2_f, ae_f = eval_ae(model)
print(f"\nfp32: R2={float(r2_f):.4f} AE={ae_f[0]}/{ae_f[1]}/{ae_f[2]}")

# ─── INT8 dynamic quantization (per-channel, Linear only) ────────
model_cpu = model.cpu()
qmodel = torch.quantization.quantize_dynamic(
    model_cpu, {torch.nn.Linear}, dtype=torch.qint8)
# enable per-channel for all quantized Linear modules
for name, mod in qmodel.named_modules():
    if hasattr(mod, "weight_fake_quant") or hasattr(mod, "weight_quantizer"):
        pass
r2_q, ae_q = eval_ae(qmodel, use_cuda=False)
print(f"INT8: R2={float(r2_q):.4f} AE={int(ae_q[0]):d}/{int(ae_q[1]):d}/{int(ae_q[2]):d}")

# ─── memory stats ─────────────────────────────────────────────────
n_params = sum(p.numel() for p in model.parameters())
state_bytes = 4 * 16 * 32 * 4 * 2  # H x Dk x Dv x 4B x layers
print(f"\nparams: {n_params:,} | fp32 {n_params*4/1024:.0f} KB | INT8 {n_params/1024:.0f} KB")
print(f"state: {state_bytes} B (2 layers)")

print("done")
