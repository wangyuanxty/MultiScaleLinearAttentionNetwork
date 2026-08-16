"""I2 diagnosis: does multi-scale (patch 2/4/8) improve local MAE on
capacity-regeneration segments vs single-scale (patch 1)?
Global R2 may be flat/negative, but multi-scale's design purpose is
capturing regeneration — compare per-segment MAE."""
import sys, numpy as np, torch
sys.path.insert(0, '.')
from gdn_model import build_gdn_model, masked_mae
from load_datasets import load_calce_cells_multivar
from data_pipeline import Seq2VecDataset, collate_seq2vec

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH, SEED, EPOCHS = 64, 42, 100
TRAIN, TEST = ["CS2_36","CS2_37","CS2_38"], "CS2_35"
W = 64

caps_all, _, _ = load_calce_cells_multivar()
caps = {c: caps_all[c].copy().astype(np.float32) for c in caps_all}
all_tr = np.concatenate([caps[c] for c in TRAIN])
lo, hi = all_tr.min(), all_tr.max()
def scale(seqs): return [(s-lo)/(hi-lo) for s in seqs]

tc = scale([caps[TEST]])[0]
print(f"Test {TEST}: {len(tc)} cycles")

# ─── detect regeneration segments ────────────────────────────
def find_regen(cap, min_jump=0.005, min_len=2):
    """Segments where capacity locally increases (regeneration)."""
    segs = []
    cur = []
    for i in range(1, len(cap)):
        if cap[i] - cap[i-1] > min_jump:
            cur.append(i)
        else:
            if len(cur) >= min_len:
                segs.append((cur[0]-1, cur[-1]))
            cur = []
    if len(cur) >= min_len:
        segs.append((cur[0]-1, cur[-1]))
    return segs

regen = find_regen(tc)
n_regen_pts = sum(e-s+1 for s, e in regen)
print(f"Regeneration segments: {len(regen)}, points: {n_regen_pts}/{len(tc)} ({100*n_regen_pts/len(tc):.1f}%)")
for s, e in regen[:8]:
    print(f"  [{s}-{e}] jump={tc[e]-tc[s]:.4f}")

# ─── train and eval ──────────────────────────────────────────
def train_eval(tag, multiscale):
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = build_gdn_model(multiscale=multiscale, input_dim=1, window_size=W,
                            output_len=1, readout="last").to(DEV)
    tr = Seq2VecDataset(scale([caps[c] for c in TRAIN]), W, 1, 1, None)
    ld = torch.utils.data.DataLoader(tr, BATCH, shuffle=True, collate_fn=collate_seq2vec)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        model.train()
        for cap, feat, tgt, msk in ld:
            cap, tgt, msk = cap.to(DEV), tgt.to(DEV), msk.to(DEV)
            loss = masked_mae(model(cap), tgt, msk)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 50 == 0:
            print(f"  [{tag}] E{ep} L={loss.item():.4f}", flush=True)

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(W, len(tc)):
            cin = torch.tensor(tc[i-W:i], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEV)
            preds.append(model(cin).item())
    pv = np.array(preds)[:len(tc)-W]; tv = tc[W:]
    r2 = 1 - np.sum((tv-pv)**2)/np.sum((tv-tv.mean())**2)

    mae_global = np.mean(np.abs(pv - tv))
    mask = np.zeros(len(tv), dtype=bool)
    for s, e in regen:
        s2, e2 = s - W, e - W
        if e2 >= 0:
            mask[max(0,s2):min(len(tv),e2+1)] = True
    if mask.sum() > 0:
        mae_regen = np.mean(np.abs(pv[mask] - tv[mask]))
        mae_nonregen = np.mean(np.abs(pv[~mask] - tv[~mask]))
    else:
        mae_regen = mae_nonregen = float('nan')
    return {"r2": r2, "mae_global": mae_global,
            "mae_regen": mae_regen, "mae_nonregen": mae_nonregen,
            "n_regen": int(mask.sum())}

print("\n=== Training single-cap ===", flush=True)
s = train_eval("single", False)
print(f"single: R2={s['r2']:.4f} MAE_global={s['mae_global']:.4f} "
      f"MAE_regen={s['mae_regen']:.4f} ({s['n_regen']}pts) MAE_nonregen={s['mae_nonregen']:.4f}", flush=True)

print("\n=== Training multi-cap ===", flush=True)
m = train_eval("multi", True)
print(f"multi:  R2={m['r2']:.4f} MAE_global={m['mae_global']:.4f} "
      f"MAE_regen={m['mae_regen']:.4f} ({m['n_regen']}pts) MAE_nonregen={m['mae_nonregen']:.4f}", flush=True)

print("\n=== I2 diagnosis ===")
print(f"Global:    single {s['mae_global']:.4f} vs multi {m['mae_global']:.4f} "
      f"({(s['mae_global']-m['mae_global'])*1000:+.1f}e-3)")
print(f"Regen:     single {s['mae_regen']:.4f} vs multi {m['mae_regen']:.4f} "
      f"({(s['mae_regen']-m['mae_regen'])*1000:+.1f}e-3)")
print(f"Non-regen: single {s['mae_nonregen']:.4f} vs multi {m['mae_nonregen']:.4f} "
      f"({(s['mae_nonregen']-m['mae_nonregen'])*1000:+.1f}e-3)")
if m['mae_regen'] < s['mae_regen'] * 0.95:
    print("VERDICT: multi-scale wins on regeneration segments -> I2 holds (local gain)")
else:
    print("VERDICT: multi-scale does NOT win on regeneration -> I2 not supported")
