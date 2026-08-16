"""Mechanism validation for Direction 2 (stage-aware cross-scale interaction).
Check on trained xchg model:
(1) cross-exchange gate values differ between regeneration vs monotonic segments
(2) gate values differ across degradation stages (early/mid/late)
Uses PANASONIC (has regeneration)."""
import sys, numpy as np, torch
sys.path.insert(0, '.')
from gdn_model import build_gdn_model, masked_mae
from load_datasets import load_panasonic_cells
from data_pipeline import Seq2VecDataset, collate_seq2vec

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
W, SEED = 30, 42

# ─── Load PANASONIC data ─────────────────────────────────────
caps_all = load_panasonic_cells()
cells = sorted(caps_all.keys())
caps = {c: caps_all[c].copy().astype(np.float32) for c in cells}
train_cells, test_cell = cells[:-1], cells[-1]
all_tr = np.concatenate([caps[c] for c in train_cells])
lo, hi = all_tr.min(), all_tr.max()
tc = (caps[test_cell] - lo) / (hi - lo + 1e-8)
raw = caps[test_cell]
print(f"PANASONIC test={test_cell} len={len(tc)}")

# Regeneration points (2-step recovery in raw capacity)
regen = set()
for i in range(2, len(raw)):
    if raw[i] - raw[i-2] > 0.005:
        regen.add(i)
print(f"Regeneration points: {len(regen)}/{len(raw)} ({100*len(regen)/len(raw):.1f}%)")

n = len(tc)
stages = {'early': set(range(0, n//3)), 'mid': set(range(n//3, 2*n//3)), 'late': set(range(2*n//3, n))}

# ─── Train xchg model (no compatible checkpoint exists) ──────
print("Training xchg model 100ep...")
torch.manual_seed(SEED); np.random.seed(SEED)
model = build_gdn_model(multiscale=True, cross_exchange=True, input_dim=1, window_size=W,
                        output_len=1, readout="last").to(DEV)
tr = Seq2VecDataset([(caps[c]-lo)/(hi-lo+1e-8) for c in train_cells], W, 1, 1, None)
ld = torch.utils.data.DataLoader(tr, 64, shuffle=True, collate_fn=collate_seq2vec)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for ep in range(100):
    model.train()
    for cap, feat, tgt, msk in ld:
        cap, tgt, msk = cap.to(DEV), tgt.to(DEV), msk.to(DEV)
        loss = masked_mae(model(cap), tgt, msk)
        opt.zero_grad(); loss.backward(); opt.step()
    if ep % 25 == 0: print(f"  E{ep} L={loss.item():.4f}", flush=True)
torch.save(model.state_dict(), '../checkpoints/panasonic_xchg_analysis.pt')
model.eval()

# ─── Hook: capture gate from cross[0] on each forward ────────
gate_holder = {}
def make_hook(layer_idx):
    def hook(module, inp, out):
        h_a, h_b = inp
        with torch.no_grad():
            p_a = h_a.mean(dim=1)
            p_b = h_b.mean(dim=1)
            g = module.gate_fn(torch.cat([p_a, p_b], dim=-1))
        gate_holder[layer_idx] = g.detach().cpu().item()
    return hook

model.cross[0].register_forward_hook(make_hook(0))

# ─── Sweep windows, collect gate at window end cycle ─────────
results = []
with torch.no_grad():
    for i in range(W, n - 2):
        cin = torch.tensor(tc[i-W:i], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEV)
        model(cin)
        if 0 in gate_holder:
            results.append((i, gate_holder[0]))
        gate_holder.clear()

print(f"\nCollected {len(results)} window gate values")

# ─── Analysis ────────────────────────────────────────────────
g_regen = [v for c, v in results if c in regen]
g_mono = [v for c, v in results if c not in regen]
print(f"\n=== Gate value analysis (cross[0]) ===")
if g_regen and g_mono:
    print(f"Regeneration: mean={np.mean(g_regen):.4f} std={np.std(g_regen):.4f} (n={len(g_regen)})")
    print(f"Monotonic:    mean={np.mean(g_mono):.4f} std={np.std(g_mono):.4f} (n={len(g_mono)})")
    d = np.mean(g_regen) - np.mean(g_mono)
    print(f"Delta = {d:+.4f}  ({100*d/np.mean(g_mono):+.1f}% relative)")
for sname, sset in stages.items():
    gs = [v for c, v in results if c in sset]
    if gs:
        print(f"Stage {sname:5s}: gate mean={np.mean(gs):.4f} std={np.std(gs):.4f} (n={len(gs)})")
