"""CALCE multi-variable comparison: cap-only vs multi-var."""
import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from gdn_model import build_gdn_model, masked_mae
from load_datasets import load_calce_cells_multivar
from data_pipeline import Seq2VecDataset
from torch.utils.data import DataLoader
import logging; logging.basicConfig(level=logging.WARNING)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
W = 64
OUT = 1
TRAIN_CELLS = ["CS2_36", "CS2_37", "CS2_38"]
TEST_CELL = "CS2_35"
EPOCHS = 100
LR = 1e-3
BATCH = 64

print(f"Device: {DEVICE} | Loading CALCE...")

caps, feats, fd = load_calce_cells_multivar()
caps = {c: caps[c].copy() for c in caps}
feats = {c: feats[c].copy() for c in feats}
fd = feats[TEST_CELL].shape[1]
print(f"feat_dim={fd} cycles={ {k: len(v) for k, v in caps.items()} }")

all_tr = np.concatenate([caps[c] for c in TRAIN_CELLS])
lo, hi = all_tr.min(), all_tr.max()

def scale(seqs):
    return [(s - lo) / (hi - lo) for s in seqs]

def train_and_eval(label, use_feat):
    in_dim = 1 + (fd if use_feat else 0)
    feat_list = [feats[c] for c in TRAIN_CELLS] if use_feat else None
    tr = Seq2VecDataset(scale([caps[c] for c in TRAIN_CELLS]), W, OUT, 1, feat_list)
    ld = DataLoader(tr, BATCH, shuffle=True)
    model = build_gdn_model(patch_size=2, input_dim=in_dim,
                             window_size=W, output_len=OUT).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"[{label}] {len(tr)} samples {sum(p.numel() for p in model.parameters()):,} params")
    for ep in range(EPOCHS):
        model.train()
        for cap, feat, tgt, msk in ld:
            cap, tgt, msk = cap.to(DEVICE), tgt.to(DEVICE), msk.to(DEVICE)
            x = torch.cat([cap, feat.to(DEVICE)], dim=-1) if (use_feat and feat.shape[-1] > 1) else cap
            opt.zero_grad()
            loss = masked_mae(model(x), tgt, msk)
            loss.backward()
            opt.step()
        if ep % 25 == 0:
            print(f"  E{ep}", flush=True)

    model.eval()
    tc = scale([caps[TEST_CELL]])[0]
    eol_n = (0.77 - lo) / (hi - lo)
    te = int(np.argmax(tc < eol_n))
    preds = []
    with torch.no_grad():
        for i in range(W, len(tc)):
            cin = torch.tensor(tc[i - W:i], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEVICE)
            if use_feat:
                fin = torch.tensor(feats[TEST_CELL][i - W:i], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                cin = torch.cat([cin, fin], dim=-1)
            preds.append(model(cin).item())
    preds = np.array(preds)
    tv = tc[W:]
    pv = preds[:len(tv)]
    r2 = 1 - np.sum((tv - pv) ** 2) / np.sum((tv - tv.mean()) ** 2)
    ae_vals = []
    for sp in [300, 400, 500]:
        sp_preds = preds[sp - W:]
        pe = -1
        for j in range(len(sp_preds) - 1):
            if sp_preds[j] >= eol_n > sp_preds[j + 1]:
                pe = sp + j + (eol_n - sp_preds[j]) / (sp_preds[j + 1] - sp_preds[j] + 1e-8)
                break
            elif sp_preds[j] < eol_n:
                pe = sp + j
                break
        ae_vals.append(abs(te - pe) if pe >= 0 else -1)
    print(f"[{label}] R2={r2:.4f} AE={ae_vals[0]:.0f}/{ae_vals[1]:.0f}/{ae_vals[2]:.0f}")
    return r2, ae_vals

print("=" * 50)
r2_s, ae_s = train_and_eval("cap-only", use_feat=False)
r2_m, ae_m = train_and_eval("multi-var", use_feat=True)
print("=" * 50)
print(f"cap-only:  R2={r2_s:.4f}  AE={ae_s[0]:.0f}/{ae_s[1]:.0f}/{ae_s[2]:.0f}")
print(f"multi-var: R2={r2_m:.4f}  AE={ae_m[0]:.0f}/{ae_m[1]:.0f}/{ae_m[2]:.0f}")
print(f"PatchFormer: AE=0.8/0.8/1.1 (cap-only)")
