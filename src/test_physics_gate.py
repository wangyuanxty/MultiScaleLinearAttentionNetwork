"""CALCE: physics gate ablation — multi-scale with/without IR gate.

Baseline: multi-scale cap+5ch
Physics: multi-scale cap+5ch + IR decay gate on coarse branch (patch 8)

Input: [cap, V_mean, V_min, V_max, IR, dt] -> ir_ch=4, t_ch=None
Physics: decay += gamma_ir * IR -> higher IR = faster state decay
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
W, OUT, BATCH, EPOCHS = 64, 1, 64, 100
TRAIN = ["CS2_36", "CS2_37", "CS2_38"]
TEST = "CS2_35"

print("Loading...")
caps_all, feats_all, _ = load_calce_cells_multivar()
caps = {c: caps_all[c].copy() for c in caps_all}
keep = [0, 1, 2, 3, 9]  # V_mean, V_min, V_max, IR, dt
feats = {c: feats_all[c][:, keep].copy() for c in feats_all}
fd = len(keep)  # 5

all_tr = np.concatenate([caps[c] for c in TRAIN])
lo, hi = all_tr.min(), all_tr.max()


def scale(seqs):
    return [(s - lo) / (hi - lo) for s in seqs]


def run(label, use_physics):
    in_dim = 1 + fd  # cap + 5 feat
    feat_list = [feats[c] for c in TRAIN]
    tr = Seq2VecDataset(scale([caps[c] for c in TRAIN]), W, OUT, 1, feat_list)
    ld = DataLoader(tr, BATCH, shuffle=True)
    model = build_gdn_model(
        multiscale=True, input_dim=in_dim, window_size=W, output_len=OUT,
        use_physics=use_physics, ir_ch=4, t_ch=None, readout="last",
    ).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    params = sum(p.numel() for p in model.parameters())
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
    preds = []
    with torch.no_grad():
        for i in range(W, len(tc)):
            cin = torch.tensor(tc[i - W:i], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEV)
            fin = torch.tensor(ftest[i - W:i], dtype=torch.float32).unsqueeze(0).to(DEV)
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
    print(f"[{label}] R2={r2:.4f} AE={aes[0]:.0f}/{aes[1]:.0f}/{aes[2]:.0f} ({params:,} params)", flush=True)

    if use_physics:
        gamma_ir = model.branches[2].layers[0]['gdn'].gamma_ir.item()
        print(f"  learned gamma_ir={gamma_ir:.4f}", flush=True)
    return r2, aes


print("=" * 50)
run("baseline (no physics)", False)
run("IR gate", True)
print("=" * 50)
print("Compare: multi-scale cap-only AE=3 (no features)")
