"""CALCE: single-step vs seq2vec (multi-scale cap-only)."""
import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from gdn_model import build_gdn_model, masked_mae
from load_datasets import load_calce_cells_multivar
from data_pipeline import Seq2VecDataset
from torch.utils.data import DataLoader
import logging; logging.basicConfig(level=logging.WARNING)

DEV = torch.device("cuda")
W, EPOCHS, BATCH, LR = 64, 100, 64, 1e-3
TRAIN = ["CS2_36", "CS2_37", "CS2_38"]
TEST = "CS2_35"

print("Loading...")
caps_all, _, _ = load_calce_cells_multivar()
caps = {c: caps_all[c].copy() for c in caps_all}
all_tr = np.concatenate([caps[c] for c in TRAIN])
lo, hi = all_tr.min(), all_tr.max()

def scale(seqs):
    return [(s - lo) / (hi - lo) for s in seqs]

def run(label, out_len, stride):
    tr = Seq2VecDataset(scale([caps[c] for c in TRAIN]), W, out_len, stride)
    ld = DataLoader(tr, BATCH, shuffle=True)
    model = build_gdn_model(multiscale=True, input_dim=1,
                             window_size=W, output_len=out_len).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"[{label}] out={out_len} stride={stride} {len(tr)} samples {sum(p.numel() for p in model.parameters()):,} params")
    for ep in range(EPOCHS):
        model.train()
        for cap, feat, tgt, msk in ld:
            cap, tgt, msk = cap.to(DEV), tgt.to(DEV), msk.to(DEV)
            pred = model(cap)
            loss = masked_mae(pred, tgt, msk)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 25 == 0: print(f"  E{ep}", flush=True)

    model.eval()
    tc = scale([caps[TEST]])[0]
    eol_n = (0.77 - lo) / (hi - lo)
    te = int(np.argmax(tc < eol_n))
    preds_all = []
    if out_len == 1:
        with torch.no_grad():
            for i in range(W, len(tc)):
                cin = torch.tensor(tc[i - W:i], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEV)
                preds_all.append(model(cin).item())
        preds_all = np.array(preds_all)
    else:
        # Seq2vec: one forward pass per sample, stitch trajectory
        pass  # evaluated per SP below

    tv = tc[W:]
    if out_len == 1:
        pv = preds_all[:len(tv)]
    r2 = 1 - np.sum((tv - pv) ** 2) / np.sum((tv - tv.mean()) ** 2) if out_len == 1 else 0

    aes = []
    for sp in [300, 400, 500]:
        if out_len == 1:
            sp_preds = preds_all[sp - W:]
            pe = -1
            for j in range(len(sp_preds) - 1):
                if sp_preds[j] >= eol_n > sp_preds[j + 1]:
                    pe = sp + j + (eol_n - sp_preds[j]) / (sp_preds[j + 1] - sp_preds[j] + 1e-8); break
                elif sp_preds[j] < eol_n: pe = sp + j; break
        else:
            cin = torch.tensor(tc[sp - W:sp], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEV)
            pred = model(cin).detach().squeeze(0).cpu().numpy()
            pe = -1
            for j in range(len(pred) - 1):
                if pred[j] >= eol_n > pred[j + 1]:
                    pe = sp + (j + (eol_n - pred[j]) / (pred[j + 1] - pred[j] + 1e-8)) * stride; break
                elif pred[j] < eol_n: pe = sp + j * stride; break
        aes.append(abs(te - pe) if pe >= 0 else -1)

    print(f"[{label}] R2={r2:.4f} AE={aes[0]:.0f}/{aes[1]:.0f}/{aes[2]:.0f}")
    return r2, aes

print("=" * 50)
r2_s, ae_s = run("single-step", out_len=1, stride=1)
r2_v, ae_v = run("seq2vec", out_len=32, stride=6)
print("=" * 50)
print(f"single-step:  R2={r2_s:.4f}  AE={ae_s[0]:.0f}/{ae_s[1]:.0f}/{ae_s[2]:.0f}")
print(f"seq2vec:      R2={r2_v:.4f}  AE={ae_v[0]:.0f}/{ae_v[1]:.0f}/{ae_v[2]:.0f}")
