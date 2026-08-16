"""CALCE: single-branch vs multi-scale, cap-only vs multi-var."""
import sys, numpy as np, torch
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

print("Loading...")
caps_all, feats_all, _ = load_calce_cells_multivar()
caps = {c: caps_all[c].copy() for c in caps_all}
# Clean 5ch: V_mean(0), V_min(1), V_max(2), IR(3), dt(9)
keep = [0, 1, 2, 3, 9]
feats = {c: feats_all[c][:, keep].copy() for c in feats_all}
fd = len(keep)

all_tr = np.concatenate([caps[c] for c in TRAIN])
lo, hi = all_tr.min(), all_tr.max()

def scale(seqs):
    return [(s - lo) / (hi - lo) for s in seqs]

def run(label, use_multiscale, use_feat):
    in_dim = 1 + (fd if use_feat else 0)
    feat_list = [feats[c] for c in TRAIN] if use_feat else None
    tr = Seq2VecDataset(scale([caps[c] for c in TRAIN]), W, OUT, 1, feat_list)
    ld = DataLoader(tr, BATCH, shuffle=True)
    model = build_gdn_model(
        patch_size=2,
        multiscale=use_multiscale,
        cross_exchange=False,  # test multi-scale alone first
        input_dim=in_dim,
        window_size=W,
        output_len=OUT,
    ).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    params = sum(p.numel() for p in model.parameters())
    for ep in range(EPOCHS):
        model.train()
        for cap, feat, tgt, msk in ld:
            cap, tgt, msk = cap.to(DEV), tgt.to(DEV), msk.to(DEV)
            x = torch.cat([cap, feat.to(DEV)], dim=-1) if (use_feat and feat.shape[-1] > 1) else cap
            opt.zero_grad()
            loss = masked_mae(model(x), tgt, msk)
            loss.backward()
            opt.step()
        if ep % 50 == 0:
            print(f"  E{ep}", flush=True)

    model.eval()
    tc = scale([caps[TEST]])[0]
    eol_n = (0.77 - lo) / (hi - lo)
    te = int(np.argmax(tc < eol_n))
    preds = []
    with torch.no_grad():
        for i in range(W, len(tc)):
            cin = torch.tensor(tc[i - W:i], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEV)
            if use_feat:
                fin = torch.tensor(feats[TEST][i - W:i], dtype=torch.float32).unsqueeze(0).to(DEV)
                cin = torch.cat([cin, fin], dim=-1)
            preds.append(model(cin).item())
    preds = np.array(preds); tv = tc[W:]; pv = preds[:len(tv)]
    r2 = 1 - np.sum((tv - pv) ** 2) / np.sum((tv - tv.mean()) ** 2)
    aes = []
    for sp in [300, 400, 500]:
        sp_preds = preds[sp - W:]; pe = -1
        for j in range(len(sp_preds) - 1):
            if sp_preds[j] >= eol_n > sp_preds[j + 1]:
                pe = sp + j + (eol_n - sp_preds[j]) / (sp_preds[j + 1] - sp_preds[j] + 1e-8); break
            elif sp_preds[j] < eol_n: pe = sp + j; break
        aes.append(abs(te - pe) if pe >= 0 else -1)
    print(f"[{label}] R2={r2:.4f} AE={aes[0]:.0f}/{aes[1]:.0f}/{aes[2]:.0f} ({params:,} params)")
    return r2, aes

print("=" * 50)
tests = [
    ("single-branch cap-only", False, False),
    ("multi-scale cap-only", True, False),
    ("multi-scale +5ch feat", True, True),
]
for label, ms, uf in tests:
    run(label, ms, uf)
print("=" * 50)
print("Compare: single cap-only: R2=0.993 AE=11 | +5ch: R2=0.990 AE=7")
