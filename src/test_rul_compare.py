"""K=1 single-step AR vs K=32 chunked rollout: which gives lower RUL MAE?
Uses fixed EOL thresholds in original Ah (NASA 1.40, CALCE 0.87)."""
import sys, numpy as np, torch
sys.path.insert(0, '.')
from gdn_model import build_gdn_model, masked_mae
from load_datasets import load_nasa_capacity, load_calce_cells_multivar
from data_pipeline import Seq2VecDataset, collate_seq2vec

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH, SEED, EPOCHS = 64, 42, 100

def run_dataset(ds, train_cells, test_cell, W, sps, eol_ah):
    if ds == 'nasa':
        caps = {c: load_nasa_capacity(c).astype(np.float32) for c in train_cells + [test_cell]}
    else:
        caps_all, _, _ = load_calce_cells_multivar()
        caps = {c: caps_all[c].copy().astype(np.float32) for c in caps_all}
    all_tr = np.concatenate([caps[c] for c in train_cells])
    lo, hi = all_tr.min(), all_tr.max()
    def scale(seqs): return [(s-lo)/(hi-lo) for s in seqs]
    raw_test = caps[test_cell]
    tc = scale([raw_test])[0]
    te = int(np.argmax(raw_test < eol_ah)) if (raw_test < eol_ah).any() else len(raw_test)
    eol_n = (eol_ah - lo) / (hi - lo + 1e-8)
    print(f"\n=== {ds}: test={test_cell} len={len(tc)} EOL@{te} (raw {raw_test[te]:.3f}Ah, norm {eol_n:.4f}) ===")

    def train(K):
        torch.manual_seed(SEED); np.random.seed(SEED)
        model = build_gdn_model(multiscale=True, input_dim=1, window_size=W,
                                output_len=K, readout="last").to(DEV)
        tr = Seq2VecDataset(scale([caps[c] for c in train_cells]), W, K, 1, None)
        ld = torch.utils.data.DataLoader(tr, BATCH, shuffle=True, collate_fn=collate_seq2vec)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for ep in range(EPOCHS):
            model.train()
            for cap, feat, tgt, msk in ld:
                cap, tgt, msk = cap.to(DEV), tgt.to(DEV), msk.to(DEV)
                loss = masked_mae(model(cap), tgt, msk)
                opt.zero_grad(); loss.backward(); opt.step()
            if ep % 50 == 0:
                print(f"  [K={K}] E{ep} L={loss.item():.4f}", flush=True)
        return model

    def eval_rul(model, K):
        model.eval()
        out = {}
        with torch.no_grad():
            for sp in sps:
                if sp >= te: continue
                window = tc[sp-W:sp].copy()
                traj = []
                while len(traj) < 800:
                    cin = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEV)
                    pred = model(cin).squeeze(0).cpu().numpy()
                    traj.extend(pred.tolist())
                    window = np.concatenate([window[K:], pred])
                    if pred[-1] < eol_n: break
                traj = np.array(traj)
                pe = -1
                for j in range(len(traj)-1):
                    if traj[j] >= eol_n > traj[j+1]:
                        pe = sp + j + (eol_n - traj[j]) / (traj[j+1] - traj[j] + 1e-8)
                        break
                    elif traj[j] < eol_n:
                        pe = sp + j; break
                out[sp] = abs(te - pe) if pe >= 0 else None
        return out

    r1 = eval_rul(train(1), 1)
    r32 = eval_rul(train(32), 32)
    print(f"\n  K=1  (single-step AR): { {sp: round(v,1) if v else None for sp,v in r1.items()} }")
    print(f"  K=32 (chunked):        { {sp: round(v,1) if v else None for sp,v in r32.items()} }")
    return r1, r32

for ds, train_cells, test_cell, W, sps, eol_ah in [
    ('nasa', ['B0006','B0007','B0018'], 'B0005', 30, [50,70,90], 1.40),
    ('calce', ['CS2_36','CS2_37','CS2_38'], 'CS2_35', 64, [300,400,500], 0.87),
]:
    run_dataset(ds, train_cells, test_cell, W, sps, eol_ah)
