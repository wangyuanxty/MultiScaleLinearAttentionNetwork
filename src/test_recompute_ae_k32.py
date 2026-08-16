"""K=32 non-recursive AE evaluation.
每 32 步用真实窗口预测 32 步;若段末 pred[-1] 低于 EOL → 段内插值穿越 → AE。
无误差累积(窗口始终真实),一次看 32 步 = 更早感知 EOL 走向。"""
import sys, numpy as np, torch, os
sys.path.insert(0, '.')
from gdn_model import build_gdn_model
from load_datasets import load_calce_cells_multivar, load_nasa_multivar, load_mit_stanford, load_panasonic_cells, load_gotion_cells

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = 'D:/research/degradation_prognostics/Transformer_and_Multi_Scale_Models/checkpoints'
K = 32

def load_data(ds):
    if ds == 'calce':
        caps_all, _, _ = load_calce_cells_multivar()
        caps = {c: caps_all[c].copy().astype(np.float32) for c in caps_all}
        return caps, ['CS2_36','CS2_37','CS2_38'], 'CS2_35', 64, [300,400,500], 0.77
    elif ds == 'nasa':
        caps = {}
        for b in ['B0005','B0006','B0007','B0018']:
            caps[b] = load_nasa_multivar(b)['capacity'].astype(np.float32)
        return caps, ['B0006','B0007','B0018'], 'B0005', 30, [50,70,90], 1.40
    elif ds == 'mit':
        caps_all = load_mit_stanford()
        tcells = ['batch2_cell11','batch2_cell26','batch2_cell32','batch2_cell36','batch2_cell37','batch2_cell42','batch2_cell46']
        caps = {c: caps_all[c].copy().astype(np.float32) for c in tcells + ['batch2_cell5']}
        return caps, tcells, 'batch2_cell5', 64, [200,300,400], 0.86
    elif ds == 'panasonic':
        caps_all = load_panasonic_cells()
        caps = {c: caps_all[c].copy().astype(np.float32) for c in caps_all}
        cells = sorted(caps.keys())
        return caps, cells[:-1], cells[-1], 30, [300,500,700], 2.12
    elif ds == 'gotion':
        caps_all = load_gotion_cells()
        caps = {c: caps_all[c].copy().astype(np.float32) for c in ['Cell01','Cell02','Cell03']}
        return caps, ['Cell02','Cell03'], 'Cell01', 30, [500,800,1100], 21.60

def recompute(ds):
    caps, train_cells, test_cell, W, sps, eol_ah = load_data(ds)
    all_tr = np.concatenate([caps[c] for c in train_cells])
    lo, hi = all_tr.min(), all_tr.max()
    tc = (caps[test_cell] - lo) / (hi - lo + 1e-8)
    threshold_n = (eol_ah - lo) / (hi - lo + 1e-8)
    te = int(np.argmax(caps[test_cell] < eol_ah)) if (caps[test_cell] < eol_ah).any() else len(tc)

    ckpt = f'{CKPT}/unified_{ds}_K32.pt'
    if not os.path.exists(ckpt):
        print(f'{ds}: no checkpoint {ckpt}')
        return
    model = build_gdn_model(multiscale=True, cross_exchange=True, input_dim=1,
                            window_size=W, output_len=K, readout="last").to(DEV)
    model.load_state_dict(torch.load(ckpt, map_location=DEV, weights_only=True))
    model.eval()

    print(f'\n=== {ds} (K=32 non-recursive stride=1, EOL@{te}) ===')
    print('  {:>5} {:>6} {:>6} {:>5} {:>6}'.format('SP','TRUL','PRUL','AE','RE'))
    for sp in sps:
        if sp >= te: continue
        tru = te - sp
        pe = -1
        with torch.no_grad():
            for t in range(sp, te):
                if t - W < 0: break
                window = tc[t-W:t]
                cin = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEV)
                pred = model(cin).squeeze(0).cpu().numpy()
                if pred[-1] < threshold_n:
                    for j in range(len(pred)-1):
                        if pred[j] >= threshold_n > pred[j+1]:
                            frac = (threshold_n - pred[j]) / (pred[j+1] - pred[j] + 1e-8)
                            pe = t + j + frac
                            break
                    else:
                        pe = t + len(pred) - 1
                    break
        if pe < 0:
            print(f'  {sp:>5} {tru:>6} {"N/A":>6} {"N/A":>5} {"N/A":>6}')
        else:
            pru = pe - sp
            ae = abs(tru - pru)
            re = ae / tru if tru > 0 else 0
            print(f'  {sp:>5} {tru:>6} {pru:>6} {ae:>5} {re:>6.4f}')

for ds in ['calce', 'nasa', 'mit', 'panasonic', 'gotion']:
    recompute(ds)
