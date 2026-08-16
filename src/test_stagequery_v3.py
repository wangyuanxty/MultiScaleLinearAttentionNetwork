"""StageQuery V3 (GDN state as query) vs scalar gate on PANASONIC.

Quick 60-epoch comparison, 3 seeds, non-recursive R2/MAE per SP.
Scalar gate reference from paper: R2=0.9988.
"""
import sys
import numpy as np
import torch

sys.path.insert(0, '.')
from gdn_model import build_gdn_model, masked_mae
from load_datasets import load_panasonic_cells
from data_pipeline import Seq2VecDataset, collate_seq2vec

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
W, BATCH, EPOCHS, SEEDS = 30, 64, 60, [42, 43, 44]

caps_all = load_panasonic_cells()
cells = sorted(caps_all.keys())
train_cells, test_cell = cells[:-1], cells[-1]
caps = {c: caps_all[c].copy().astype(np.float32) for c in cells}
all_tr = np.concatenate([caps[c] for c in train_cells])
lo, hi = all_tr.min(), all_tr.max()
tc_test = (caps[test_cell] - lo) / (hi - lo + 1e-8)
n_test = len(tc_test)
sps = [300, 500]
print(f"PANASONIC: {len(train_cells)} train, test={test_cell} ({n_test} cycles)")


def train_eval(tag, seed, multiscale=True, cross_exchange=False, stage_query=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_gdn_model(
        multiscale=multiscale,
        cross_exchange=cross_exchange,
        stage_query=stage_query,
        input_dim=1, window_size=W, output_len=1, readout="last",
    ).to(DEV)
    tr = Seq2VecDataset(
        [(caps[c] - lo) / (hi - lo + 1e-8) for c in train_cells],
        W, 1, 1, None)
    ld = torch.utils.data.DataLoader(tr, BATCH, shuffle=True,
                                     collate_fn=collate_seq2vec)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        model.train()
        for cap, feat, tgt, msk in ld:
            cap, tgt, msk = cap.to(DEV), tgt.to(DEV), msk.to(DEV)
            loss = masked_mae(model(cap), tgt, msk)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(W, n_test):
            cin = torch.tensor(tc_test[i - W:i], dtype=torch.float32) \
                .unsqueeze(0).unsqueeze(-1).to(DEV)
            preds.append(model(cin).item())
    pv = np.array(preds)[:n_test - W]
    tv = tc_test[W:]
    r2 = 1 - np.sum((tv - pv) ** 2) / np.sum((tv - tv.mean()) ** 2)
    mae_sp = {}
    for sp in sps:
        seg_p, seg_t = pv[sp - W:], tc_test[sp:]
        n = min(len(seg_p), len(seg_t))
        mae_sp[sp] = np.mean(np.abs(seg_p[:n] - seg_t[:n]))
    print(f"  [{tag}] seed{seed}: R2={r2:.4f} "
          + " ".join(f"SP{sp}={mae_sp[sp]:.4f}" for sp in sps), flush=True)
    return r2, mae_sp


results = {}
for tag, ms, xe, sq in [
    ("single", False, False, False),
    ("multi(no-xchg)", True, False, False),
    ("StageQuery-V3", True, False, True),
]:
    r2s, maes = [], {sp: [] for sp in sps}
    for seed in SEEDS:
        r2, m = train_eval(tag, seed, multiscale=ms, cross_exchange=xe, stage_query=sq)
        r2s.append(r2)
        for sp in sps:
            maes[sp].append(m[sp])
    results[tag] = (r2s, maes)
    print(f"\n=== {tag}: R2 mean={np.mean(r2s):.4f}±{np.std(r2s):.4f} "
          + " ".join(f"SP{sp}={np.mean(maes[sp]):.4f}±{np.std(maes[sp]):.4f}" for sp in sps),
          flush=True)
