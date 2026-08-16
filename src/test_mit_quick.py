"""MIT subset: single vs multi vs StageQuery xchg (I2 validation)."""
import sys, numpy as np, torch
sys.path.insert(0, '.')
from gdn_model import build_gdn_model, masked_mae
from load_datasets import load_mit_stanford
from data_pipeline import Seq2VecDataset, collate_seq2vec

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
W, BATCH, SEED, EPOCHS = 64, 64, 42, 60

caps_all = load_mit_stanford()
train_cells = ['batch2_cell11','batch2_cell26','batch2_cell32','batch2_cell36',
               'batch2_cell37','batch2_cell42','batch2_cell46']
test_cell = 'batch2_cell5'
caps = {c: caps_all[c].copy().astype(np.float32) for c in train_cells + [test_cell]}
all_tr = np.concatenate([caps[c] for c in train_cells])
lo, hi = all_tr.min(), all_tr.max()
def scale(seqs): return [(s-lo)/(hi-lo) for s in seqs]
tc_test = scale([caps[test_cell]])[0]
n_test = len(tc_test)
sps = [100, 200, 300]
print(f"MIT subset: {len(train_cells)} train, test={test_cell} ({n_test} cycles)")

def train_eval(tag, multiscale, cross_exchange=False):
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = build_gdn_model(multiscale=multiscale, cross_exchange=cross_exchange,
                            input_dim=1, window_size=W, output_len=1,
                            readout="last").to(DEV)
    tr = Seq2VecDataset(scale([caps[c] for c in train_cells]), W, 1, 1, None)
    ld = torch.utils.data.DataLoader(tr, BATCH, shuffle=True, collate_fn=collate_seq2vec)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        model.train()
        for cap, feat, tgt, msk in ld:
            cap, tgt, msk = cap.to(DEV), tgt.to(DEV), msk.to(DEV)
            loss = masked_mae(model(cap), tgt, msk)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(W, n_test):
            cin = torch.tensor(tc_test[i-W:i], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEV)
            preds.append(model(cin).item())
    pv = np.array(preds)[:n_test-W]; tv = tc_test[W:]
    r2 = 1 - np.sum((tv-pv)**2)/np.sum((tv-tv.mean())**2)
    mae_sp = {}
    for sp in sps:
        seg = pv[sp-W:]; seg_t = tc_test[sp:]
        n = min(len(seg), len(seg_t))
        mae_sp[sp] = np.mean(np.abs(seg[:n]-seg_t[:n]))
    print(f"  {tag}: R2={r2:.4f} " + " ".join(f"SP{sp}={mae_sp[sp]:.4f}" for sp in sps), flush=True)
    return r2, mae_sp

print("\n--- MIT subset I2 validation ---", flush=True)
r2_s, m_s = train_eval("single", False, False)
r2_m, m_m = train_eval("multi(no-xchg)", True, False)
r2_x, m_x = train_eval("StageQuery(xchg)", True, True)

print(f"\n=== MIT subset comparison ===")
for tag, r2, m in [("single",r2_s,m_s),("multi(no-x)",r2_m,m_m),("StageQuery",r2_x,m_x)]:
    mae_str = " ".join(f"SP{sp}={m[sp]:.4f}" for sp in sps)
    print(f"  {tag:16s}: R2={r2:.4f} {mae_str}")
