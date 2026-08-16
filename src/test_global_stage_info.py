"""Direction 2 premise: does the global branch (coarse, patch=8) pooled
vector carry degradation-stage information (early/mid/late)?
Uses multi-scale model (no cross-exchange needed — just check if global
branch alone separates stages)."""
import sys, numpy as np, torch
sys.path.insert(0, '.')
from gdn_model import build_gdn_model, masked_mae
from load_datasets import load_panasonic_cells
from data_pipeline import Seq2VecDataset, collate_seq2vec

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
W, SEED = 30, 42

caps_all = load_panasonic_cells()
cells = sorted(caps_all.keys())
caps = {c: caps_all[c].copy().astype(np.float32) for c in cells}
train_cells, test_cell = cells[:-1], cells[-1]
all_tr = np.concatenate([caps[c] for c in train_cells])
lo, hi = all_tr.min(), all_tr.max()
tc = (caps[test_cell] - lo) / (hi - lo + 1e-8)
n = len(tc)
print(f"PANASONIC test={test_cell} len={n}")

# Stage labels
stages = np.array([0]* (n//3) + [1]*(n//3) + [2]*(n - 2*(n//3)))[:n]

# Train multi-scale model
print("Training multi-scale model (no xchg) 100ep...")
torch.manual_seed(SEED); np.random.seed(SEED)
model = build_gdn_model(multiscale=True, cross_exchange=False, input_dim=1,
                        window_size=W, output_len=1, readout="last").to(DEV)
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
model.eval()

# ─── Extract global branch (coarse, idx=2) pooled vectors ─────
print("Extracting global branch pooled vectors...")
global_vecs = []
stage_labels = []

with torch.no_grad():
    for i in range(W, n - 2):
        cin = torch.tensor(tc[i-W:i], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEV)
        phys_mod = model.branches[2].make_phys_mod(cin, None)
        h = model.branches[2].init_proj(cin)
        for layer in model.branches[2].layers:
            h = layer['norm'](layer['gdn'](h, phys_mod) + h)
        gv = h.mean(dim=1).cpu().numpy()[0]
        global_vecs.append(gv)
        stage_labels.append(stages[i])

global_vecs = np.array(global_vecs)
stage_labels = np.array(stage_labels)
print(f"Global vectors: {global_vecs.shape}")

# ─── Analysis: linear separability ────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

clf = LogisticRegression(max_iter=2000, solver='lbfgs')
scores = cross_val_score(clf, global_vecs, stage_labels, cv=5, scoring='accuracy')
print(f"\n=== Stage separability from global branch ===")
print(f"3-class accuracy: {scores.mean():.3f} +/- {scores.std():.3f}  (chance: 0.333)")
separable = scores.mean() > 0.5
print(f"Stage-separable: {'YES' if separable else 'NO'}")

# Pairwise distance ratios
for a, b, an, bn in [(0,1,'early','mid'), (1,2,'mid','late'), (0,2,'early','late')]:
    ma = global_vecs[stage_labels==a].mean(axis=0)
    mb = global_vecs[stage_labels==b].mean(axis=0)
    dist = np.linalg.norm(ma - mb)
    v_a = global_vecs[stage_labels==a].var(axis=0).sum()
    v_b = global_vecs[stage_labels==b].var(axis=0).sum()
    sep = dist / np.sqrt(v_a + v_b + 1e-8)
    print(f"  {an} vs {bn}: sep={sep:.3f} (dist={dist:.4f})")

# Also: can global branch distinguish regen vs non-regen?
regen_set = set()
raw = caps[test_cell]
for i in range(2, len(raw)):
    if raw[i] - raw[i-2] > 0.005:
        regen_set.add(i)
regen_labels = np.array([1 if i in regen_set else 0 for i in range(W, n-2)])
clf2 = LogisticRegression(max_iter=2000, solver='lbfgs')
scores2 = cross_val_score(clf2, global_vecs, regen_labels, cv=5, scoring='accuracy')
n_regen = regen_labels.sum()
print(f"\nRegen detection from global: {scores2.mean():.3f} +/- {scores2.std():.3f} (chance: {n_regen/len(regen_labels):.3f})")
