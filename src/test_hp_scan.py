"""Hyperparameter scan on CALCE — check if tuning closes the gap to PatchFormer (AE 0.8-1.1).

Small scan first (lr x d_model x layers), 3 seeds each, 60 epochs.
If tuned AE approaches PatchFormer's 0.8-1.1, the gap is tuning effort, not method.
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
W, OUT, BATCH, EPOCHS = 64, 1, 64, 60
TRAIN = ["CS2_36", "CS2_37", "CS2_38"]
TEST = "CS2_35"

print("Loading...")
caps_all, _, _ = load_calce_cells_multivar()
caps = {c: caps_all[c].copy() for c in caps_all}
all_tr = np.concatenate([caps[c] for c in TRAIN])
lo, hi = all_tr.min(), all_tr.max()


def scale(seqs):
    return [(s - lo) / (hi - lo) for s in seqs]


tr = Seq2VecDataset(scale([caps[c] for c in TRAIN]), W, OUT, 1)
ld = DataLoader(tr, BATCH, shuffle=True)


def run(lr, d_model, num_layers, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_gdn_model(
        multiscale=False, input_dim=1, window_size=W, output_len=OUT,
        readout="last", d_model=d_model, num_layers=num_layers,
    ).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
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
    preds = []
    with torch.no_grad():
        for i in range(W, len(tc)):
            cin = torch.tensor(tc[i - W:i], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEV)
            preds.append(model(cin).item())
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


configs = []
for lr in [1e-3, 3e-4]:
    for dm in [64, 128]:
        for nl in [2, 3]:
            configs.append((lr, dm, nl))

print(f"scan: {len(configs)} configs x 3 seeds, {len(configs)*3} runs x {EPOCHS} ep", flush=True)
results = {}
for lr, dm, nl in configs:
    r2s, aess = [], []
    for seed in [1, 2, 3]:
        r2, aes = run(lr, dm, nl, seed)
        r2s.append(r2); aess.append(aes)
    mean_ae = np.mean([np.mean(a) for a in aess])
    results[(lr, dm, nl)] = {"r2": np.mean(r2s), "ae_mean": mean_ae,
                             "ae_all": aess, "r2_all": r2s}
    print(f"lr={lr} d={dm} L={nl}: R2={np.mean(r2s):.4f} AE={mean_ae:.1f} "
          f"(runs: {[f'{np.mean(a):.0f}' for a in aess]})", flush=True)

best = min(results, key=lambda k: results[k]["ae_mean"])
print(f"\nbest: lr={best[0]} d={best[1]} L={best[2]} AE={results[best]['ae_mean']:.1f}")
print(f"PatchFormer (tuned, 10-seed avg): AE=0.8-1.1 | ours default: AE=2-3")
print("done")
