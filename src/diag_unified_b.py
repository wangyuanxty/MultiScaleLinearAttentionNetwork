"""Diagnose why UnifiedGDNSeq2Vec (B) doesn't converge.

Single forward/backward, print gradient magnitudes by position segment:
  - window tokens (pos 0-63) vs query tokens (pos 64-95) in the input
  - encoder params (input_proj, pos_embed, layers) vs query param vs head
Also compare grad at GDN layer input vs output (does the block kill gradients?).
"""
import sys, numpy as np, torch, torch.nn as nn
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from gdn_v2 import GDN2Block
from load_datasets import load_calce_cells_multivar
from data_pipeline import Seq2VecDataset
from torch.utils.data import DataLoader
import logging; logging.basicConfig(level=logging.WARNING)

W, OUT, BATCH, STRIDE = 64, 32, 64, 1
TRAIN = ["CS2_36", "CS2_37", "CS2_38"]

src = open(Path(__file__).parent / "test_seq2vec_v2.py", encoding="utf-8").read()
cls_code = src[src.index("class UnifiedGDNSeq2Vec"):src.index("def masked_mae")]
exec(cls_code)

print("Loading...")
caps_all, _, _ = load_calce_cells_multivar()
caps = {c: caps_all[c].copy() for c in caps_all}
all_tr = np.concatenate([caps[c] for c in TRAIN])
lo, hi = all_tr.min(), all_tr.max()


def scale(seqs):
    return [(s - lo) / (hi - lo) for s in seqs]


tr = Seq2VecDataset(scale([caps[c] for c in TRAIN]), W, OUT, STRIDE)
cap, feat, tgt, msk = next(iter(DataLoader(tr, BATCH, shuffle=True)))
x = cap  # (B, 64, 1)


def masked_mae(pred, target, mask=None):
    err = (pred - target).abs()
    if mask is not None:
        err = err * mask
        return err.sum() / (mask.sum() + 1e-8)
    return err.mean()


model = UnifiedGDNSeq2Vec(out_len=OUT)

hq_in_grads, hq_out_grads = [], []
handles = []
for i, l in enumerate(model.layers):
    def mk_hooks(i):
        def bw_in_hook(m, grad_in, grad_out):
            hq_in_grads.append((i, grad_in[0].norm().item()))
            hq_out_grads.append((i, grad_out[0].norm().item()))
        return bw_in_hook
    handles.append(model.layers[i]['gdn'].register_full_backward_hook(mk_hooks(i)))

pred = model(x)
loss = masked_mae(pred, tgt, msk)
loss.backward()
for h in handles:
    h.remove()

print("--- GDN layer backprop grads (norm at input vs output) ---")
for i, g in hq_in_grads:
    print(f"layer{i}: grad_in_norm={g:.5f}")
for i, g in hq_out_grads:
    print(f"layer{i}: grad_out_norm={g:.5f}")

print("\n--- parameter gradients ---")
for name, p in model.named_parameters():
    if p.grad is not None:
        print(f"{name:20s} grad_norm={p.grad.norm().item():.6f}")

# gradient through the head output back to the 96-token input, split by segment
hq = torch.cat([
    model.input_proj(x) + model.pos_embed,
    model.queries.expand(x.size(0), -1, -1),
], dim=1)
hq.retain_grad()
pred2 = model(x)
loss2 = masked_mae(pred2, tgt, msk)
loss2.backward()
g_in = hq.grad if hq.grad is not None else None
if g_in is not None:
    g_win = g_in[:, :64, :].norm().item()
    g_qry = g_in[:, 64:, :].norm().item()
    print(f"\ninput grads: window(0-63) norm={g_win:.6f}  query(64-95) norm={g_qry:.6f}")

with torch.no_grad():
    preds = pred.squeeze(1).numpy()
    print(f"\npred on batch: mean={preds.mean():.4f} std={preds.std():.4f} "
          f"first-sample first6={np.round(preds[0, :6], 4)}")
    print(f"targets      : mean={tgt.numpy().mean():.4f} std={tgt.numpy().std():.4f} "
          f"first-sample first6={np.round(tgt[0, :6].numpy(), 4)}")
