"""Locate which multi-scale branch is quantization-sensitive.

Quantize each branch individually (fine/mid/coarse) and measure INT8 AE.
If one branch dominates the loss, deployment can keep it fp32 (mixed precision).
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
W, OUT, BATCH, EPOCHS = 64, 1, 64, 40
TRAIN = ["CS2_36", "CS2_37", "CS2_38"]
TEST = "CS2_35"

torch.manual_seed(42)
np.random.seed(42)

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
for ep in range(EPOCHS):
    model.train()
    for cap, feat, tgt, msk in ld:
        cap, tgt, msk = cap.to(DEV), tgt.to(DEV), msk.to(DEV)
        opt.zero_grad()
        loss = masked_mae(model(cap), tgt, msk)
        loss.backward()
        opt.step()

model.eval()
tc = scale([caps[TEST]])[0]
eol_n = (0.77 - lo) / (hi - lo)
te = int(np.argmax(tc < eol_n))


def eval_ae(m):
    m.eval()
    preds = []
    with torch.no_grad():
        for i in range(W, len(tc)):
            cin = torch.tensor(tc[i - W:i], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            preds.append(m(cin).item())
    pv = np.array(preds)[:len(tc) - W]; tv = tc[W:]
    aes = []
    for sp in [300, 400, 500]:
        sp_preds = pv[sp - W:]; pe = -1
        for j in range(len(sp_preds) - 1):
            if sp_preds[j] >= eol_n > sp_preds[j + 1]:
                pe = sp + j + (eol_n - sp_preds[j]) / (sp_preds[j + 1] - sp_preds[j] + 1e-8); break
            elif sp_preds[j] < eol_n: pe = sp + j; break
        aes.append(abs(te - pe) if pe >= 0 else -1)
    return aes


model_cpu = model.cpu()
print("fp32:", eval_ae(model_cpu))
q_full = torch.quantization.quantize_dynamic(model_cpu, {torch.nn.Linear}, dtype=torch.qint8)
print("full INT8:", eval_ae(q_full))

# branch sensitivity proxy: quant-step noise on each branch's weights
print("\n=== branch sensitivity proxy (weight noise ~ INT8 step) ===")
branch_names = ["fine(ps2)", "mid(ps4)", "coarse(ps8)"]
for bi in range(3):
    m = build_gdn_model(multiscale=True, input_dim=1, window_size=W,
                        output_len=OUT, readout="last")
    m.load_state_dict(model_cpu.state_dict())
    m.eval()
    with torch.no_grad():
        for p in m.branches[bi].parameters():
            step = p.abs().max().item() / 127.0
            p.add_(torch.randn_like(p) * step * 0.5)
    aes = eval_ae(m)
    print(f"{branch_names[bi]}: noise AE={aes[0]}/{aes[1]}/{aes[2]}")
print("done")
