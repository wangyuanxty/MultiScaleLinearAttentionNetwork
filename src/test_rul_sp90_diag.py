"""NASA RUL anomaly diagnosis: why is SP90 error (50) > SP50 (10)?
Inspect the rollout trajectory around EOL crossing for each SP."""
import sys, numpy as np, torch
sys.path.insert(0, '.')
from gdn_model import build_gdn_model, masked_mae
from load_datasets import load_nasa_capacity
from data_pipeline import Seq2VecDataset, collate_seq2vec

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH, SEED, EPOCHS = 64, 42, 100
TRAIN, TEST = ["B0006","B0007","B0018"], "B0005"
W, EOL_RATIO = 30, 0.70

caps_all = {c: load_nasa_capacity(c).astype(np.float32) for c in TRAIN + [TEST]}
all_tr = np.concatenate([caps_all[c] for c in TRAIN])
lo, hi = all_tr.min(), all_tr.max()
def scale(seqs): return [(s-lo)/(hi-lo) for s in seqs]
tc = scale([caps_all[TEST]])[0]
eol_n = (EOL_RATIO - lo) / (hi - lo)
te = int(np.argmax(tc < eol_n))
print(f"Test {TEST}: {len(tc)} cycles, normalized EOL={eol_n:.4f} at cycle {te}")

# Train K=32 model
torch.manual_seed(SEED); np.random.seed(SEED)
K = 32
model = build_gdn_model(multiscale=True, input_dim=1, window_size=W,
                        output_len=K, readout="last").to(DEV)
tr = Seq2VecDataset(scale([caps_all[c] for c in TRAIN]), W, K, 1, None)
ld = torch.utils.data.DataLoader(tr, BATCH, shuffle=True, collate_fn=collate_seq2vec)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for ep in range(EPOCHS):
    model.train()
    for cap, feat, tgt, msk in ld:
        cap, tgt, msk = cap.to(DEV), tgt.to(DEV), msk.to(DEV)
        loss = masked_mae(model(cap), tgt, msk)
        opt.zero_grad(); loss.backward(); opt.step()
    if ep % 50 == 0:
        print(f"E{ep} L={loss.item():.4f}", flush=True)
model.eval()

def rollout(sp, verbose=True):
    window = tc[sp-W:sp].copy()
    traj = []
    while len(traj) < 800:
        cin = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEV)
        pred = model(cin).squeeze(0).cpu().numpy()
        traj.extend(pred.tolist())
        window = np.concatenate([window[K:], pred])
        if pred[-1] < eol_n:
            break
    traj = np.array(traj)
    pe = -1
    for j in range(len(traj)-1):
        if traj[j] >= eol_n > traj[j+1]:
            frac = (eol_n - traj[j]) / (traj[j+1] - traj[j] + 1e-8)
            pe = sp + j + frac
            if verbose:
                print(f"\nSP={sp}: crossing at j={j} (pred cycle {sp+j})")
                print(f"  traj[j]={traj[j]:.4f} traj[j+1]={traj[j+1]:.4f} eol_n={eol_n:.4f}")
                print(f"  frac={frac:.4f} -> pred_EOL={pe:.2f} (true {te}) err={abs(te-pe):.1f}")
                print(f"  true cap at {sp+j}: {tc[sp+j]:.4f} vs pred {traj[j]:.4f}")
                print(f"  true cap at {sp+j+1}: {tc[sp+j+1]:.4f} vs pred {traj[j+1]:.4f}")
                print(f"  rollout length: {len(traj)} steps")
            break
        elif traj[j] < eol_n:
            pe = sp + j
            if verbose:
                print(f"\nSP={sp}: already below EOL at j={j} (pred cycle {sp+j})")
                print(f"  traj[j]={traj[j]:.4f} < eol_n={eol_n:.4f}")
                print(f"  pred_EOL={pe:.1f} (true {te}) err={abs(te-pe):.1f}")
            break
    if pe < 0:
        print(f"\nSP={sp}: never crossed EOL in {len(traj)} steps (max {traj.max():.4f})")
    return pe

for sp in [50, 70, 90]:
    rollout(sp)

print(f"\nTrue capacity near EOL (cycles {te-15}..{te+5}):")
for i in range(te-15, min(te+6, len(tc))):
    marker = " <-- EOL" if i == te else ""
    print(f"  {i}: {tc[i]:.4f} (raw {caps_all[TEST][i]:.3f}){marker}")
